"""Evaluate learned identities against a frozen checkpoint under fixed conditions.

No learning, curriculum, or reward shaping is active during evaluation. By
default actions are sampled from the learned policy without the training-only
exploration floor. ``--behavior-mix`` explicitly restores each model's saved
exploration mix, while ``--greedy`` measures argmax behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from flyfight.environment import Arena, DEFAULT_MAP
from flyfight.model import FlyPolicy, action_stats
from flyfight.training import choose_device


def wilson(wins: int, total: int) -> list[float]:
    if not total:
        return [0.0, 1.0]
    z = 1.96
    p = wins / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _checkpoint_model(checkpoint: dict, identity: str) -> dict[str, torch.Tensor]:
    return checkpoint["models"][0 if identity == "A" else 1]


def _saved_mix(checkpoint: dict, *, enabled: bool) -> float:
    return float(checkpoint.get("config", {}).get("exploration_mix", 0.0)) if enabled else 0.0


def _select_actions(logits: torch.Tensor, model: FlyPolicy, *, greedy: bool,
                    exploration_mix: float) -> torch.Tensor:
    if greedy:
        return torch.stack([head.argmax(-1) for head in logits.split(model.heads, dim=-1)], -1)
    return action_stats(logits, heads=model.heads, exploration_mix=exploration_mix)[0]


@torch.inference_mode()
def evaluate(checkpoint: str, opponent: str, *, games: int = 128, envs: int = 32,
             device: str = "auto", seed: int = 424242, identity: str = "A",
             map_path: str = str(DEFAULT_MAP), behavior_mix: bool = False,
             greedy: bool = False) -> dict:
    """Evaluate one checkpoint identity against the opposite frozen identity."""
    if min(games, envs) < 1:
        raise ValueError("games and envs must be positive")
    if identity not in ("A", "B"):
        raise ValueError("identity must be A or B")
    if behavior_mix and greedy:
        raise ValueError("behavior_mix and greedy are mutually exclusive")

    dev = choose_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    candidate_checkpoint = torch.load(checkpoint, map_location=dev, weights_only=True)
    opponent_checkpoint = torch.load(opponent, map_location=dev, weights_only=True)
    if candidate_checkpoint["map_hash"] != opponent_checkpoint["map_hash"]:
        raise ValueError("Different maps in checkpoints")
    supplied_hash = hashlib.sha256(Path(map_path).read_bytes()).hexdigest()
    if candidate_checkpoint["map_hash"] != supplied_hash:
        raise ValueError("Supplied map does not match checkpoint")

    cfg = candidate_checkpoint["config"]
    other_cfg = opponent_checkpoint["config"]
    action_steps = int(cfg.get("action_steps", 9))
    speeds = {key: cfg.get(key, default) for key, default in
              (("turn_speed", 2.0), ("pitch_speed", 1.0))}
    for key, value in speeds.items():
        if other_cfg.get(key, 2.0 if key == "turn_speed" else 1.0) != value:
            raise ValueError(f"Different {key}")
    if int(other_cfg.get("action_steps", 9)) != action_steps:
        raise ValueError("Different action_steps")
    for key in ("width", "height", "hidden", "dt", "episode_seconds"):
        if cfg[key] != other_cfg[key]:
            raise ValueError(f"Different {key}")

    models = [FlyPolicy(cfg["width"], cfg["height"], cfg["hidden"], action_steps).to(dev).eval()
              for _ in range(2)]
    opponent_identity = "B" if identity == "A" else "A"
    models[0].load_state_dict(_checkpoint_model(candidate_checkpoint, identity))
    models[1].load_state_dict(_checkpoint_model(opponent_checkpoint, opponent_identity))
    mixes = [_saved_mix(candidate_checkpoint, enabled=behavior_mix),
             _saved_mix(opponent_checkpoint, enabled=behavior_mix)]

    lane_count = min(envs, games)
    arena = Arena(lane_count, dev, width=cfg["width"], height=cfg["height"],
                  dt=cfg["dt"], episode_seconds=cfg["episode_seconds"], seed=seed,
                  map_path=map_path, action_steps=action_steps, **speeds,
                  headshot_bonus=0.0, damage_reward=0.0, aim_reward=0.0,
                  curriculum_fraction=0.0)
    hidden = [torch.zeros(lane_count, cfg["hidden"], device=dev) for _ in range(2)]
    batch = torch.arange(lane_count, device=dev)
    counts = [0] * lane_count
    quotas = [games // lane_count + (i < games % lane_count) for i in range(lane_count)]
    roles = torch.arange(lane_count, device=dev) % 2
    wins = losses = draws = 0
    active_steps = shots = hits = headshots = 0
    slot_steps = [0, 0]
    slot_games = [0, 0]

    while any(count < quota for count, quota in zip(counts, quotas)):
        active = torch.tensor([count < quota for count, quota in zip(counts, quotas)],
                              dtype=torch.bool, device=dev)
        rgb = arena.observe()
        actions = torch.zeros(lane_count, 2, 4, dtype=torch.long, device=dev)
        for model_index, model in enumerate(models):
            slot = roles if model_index == 0 else 1 - roles
            logits, _, hidden[model_index] = model(
                rgb[batch, slot], arena.last_action[batch, slot], hidden[model_index])
            actions[batch, slot] = _select_actions(
                logits, model, greedy=greedy, exploration_mix=mixes[model_index])
        _, done, info = arena.step(actions)

        candidate_fired = info["fired"][batch, roles]
        candidate_hits = info["hits"][batch, roles]
        candidate_headshots = info["headshots"][batch, roles]
        active_steps += int(active.sum())
        shots += int(candidate_fired[active].sum())
        hits += int(candidate_hits[active].sum())
        headshots += int(candidate_headshots[active].sum())
        for slot in (0, 1):
            slot_steps[slot] += int((active & (roles == slot)).sum())

        outcome = info["outcome"][batch, roles]
        completed = done & active
        for index, result in zip(torch.where(completed)[0].cpu().tolist(),
                                 outcome[completed].cpu().tolist()):
            wins += int(result > 0)
            losses += int(result < 0)
            draws += int(result == 0)
            slot_games[int(roles[index])] += 1
            counts[index] += 1
        roles = torch.where(done, 1 - roles, roles)
        hidden = [state * (~done)[:, None] for state in hidden]

    scale = 1000.0 / max(active_steps, 1)
    return {
        "checkpoint": checkpoint,
        "opponent": opponent,
        "candidate_identity": identity,
        "opponent_identity": opponent_identity,
        "seed": seed,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate_all": wins / games,
        "win_rate_wilson95": wilson(wins, games),
        "decisive_win_rate": wins / (wins + losses) if wins + losses else None,
        "candidate_active_steps": active_steps,
        "candidate_shots": shots,
        "candidate_hits": hits,
        "candidate_headshots": headshots,
        "candidate_shots_per_1000_active_steps": shots * scale,
        "candidate_hits_per_1000_active_steps": hits * scale,
        "candidate_headshots_per_1000_active_steps": headshots * scale,
        "candidate_hit_rate": hits / shots if shots else None,
        "candidate_headshot_rate_per_hit": headshots / hits if hits else None,
        "candidate_arena_slot_steps": slot_steps,
        "candidate_arena_slot_games": slot_games,
        "action_mode": "greedy" if greedy else ("saved_behavior_mix" if behavior_mix else "raw_policy_sample"),
        "applied_exploration_mix": {"candidate": mixes[0], "opponent": mixes[1]},
        "reward_shaping": "off",
        "curriculum": "off_full_map",
        "note": "Frozen-opponent evaluation with alternating arena slots and no learning. Combat activity alone does not prove learned skill.",
    }


def evaluate_both(checkpoint: str, opponent: str, **kwargs) -> dict:
    """Evaluate both candidate identities under the same frozen conditions."""
    if "identity" in kwargs:
        raise TypeError("evaluate_both selects both identities; do not pass identity")
    results = {identity: evaluate(checkpoint, opponent, identity=identity, **kwargs)
               for identity in ("A", "B")}
    totals = {key: sum(result[key] for result in results.values())
              for key in ("games", "wins", "losses", "draws", "candidate_active_steps",
                          "candidate_shots", "candidate_hits", "candidate_headshots")}
    steps = max(totals["candidate_active_steps"], 1)
    decisive = totals["wins"] + totals["losses"]
    return {
        "checkpoint": checkpoint,
        "opponent": opponent,
        "candidate_identity": "both",
        "seed": kwargs.get("seed", 424242),
        **totals,
        "win_rate_all": totals["wins"] / totals["games"],
        "win_rate_wilson95": wilson(totals["wins"], totals["games"]),
        "decisive_win_rate": totals["wins"] / decisive if decisive else None,
        "candidate_shots_per_1000_active_steps": totals["candidate_shots"] * 1000 / steps,
        "candidate_hits_per_1000_active_steps": totals["candidate_hits"] * 1000 / steps,
        "candidate_headshots_per_1000_active_steps": totals["candidate_headshots"] * 1000 / steps,
        "identities": results,
        "note": "Aggregate of separately evaluated checkpoint identities A and B under matched frozen conditions.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--identity", choices=["A", "B", "both"], default="A")
    parser.add_argument("--map-path", default=str(DEFAULT_MAP))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--behavior-mix", action="store_true",
                      help="Opt in to each selected model's saved exploration_mix.")
    mode.add_argument("--greedy", action="store_true", help="Use argmax actions instead of sampling.")
    parser.add_argument("--output", default="evaluation.json")
    args = vars(parser.parse_args())
    output = Path(args.pop("output"))
    identity = args.pop("identity")
    result = evaluate_both(**args) if identity == "both" else evaluate(**args, identity=identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
