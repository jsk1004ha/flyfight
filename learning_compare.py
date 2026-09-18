"""Run a seeded, equal-environment-step baseline/improved learning comparison.

The defaults are a small smoke experiment (hidden=64), not a production skill
claim. Each final policy is evaluated against its own frozen initial snapshot
on the full map with shaping and curriculum disabled by ``evaluate.py``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch

from evaluate import evaluate_both
from flyfight.environment import DEFAULT_MAP
from flyfight.training import Config, Trainer


PROFILE_NAMES = ("baseline", "improved")


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def profile_config(name: str, *, seed: int, hidden: int, env_steps: int,
                   device: str, run_dir: Path, action_steps: int,
                   map_path: str) -> Config:
    """Build an explicit profile while keeping the environment-step budget exact."""
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {name!r}")
    horizon = 16 if name == "baseline" else 64
    steps_per_update = 8 * horizon
    if env_steps <= 0 or env_steps % steps_per_update:
        raise ValueError(
            f"env_steps must be a positive multiple of {steps_per_update} for {name}")
    updates = env_steps // steps_per_update
    common = dict(
        envs=8,
        hidden=hidden,
        horizon=horizon,
        epochs=2,
        minibatch_envs=8,
        gamma=.999,
        gae_lambda=.97,
        clip=.2,
        episode_seconds=60.0,
        dt=.1,
        seed=seed,
        device=device,
        threads=2,
        updates=updates,
        save_every=max(1, updates),
        publish_every=max(1, updates),
        run_dir=str(run_dir),
        map_path=map_path,
        action_steps=action_steps,
        turn_speed=2.0,
        pitch_speed=1.0,
        headshot_bonus=.25,
    )
    if name == "baseline":
        return Config(
            **common,
            learning_rate=3e-4,
            entropy=.01,
            exploration_mix=0.0,
            target_kl=0.0,
            advantage_floor=1e-8,
            exploration_prior=0.0,
            damage_reward=0.0,
            aim_reward=0.0,
            curriculum_fraction=0.0,
            curriculum_updates=0,
        )
    curriculum_updates = max(1, round(updates * .8))
    return Config(
        **common,
        learning_rate=1e-4,
        entropy=.03,
        exploration_mix=.1,
        target_kl=.02,
        advantage_floor=.1,
        exploration_prior=.002,
        damage_reward=.5,
        aim_reward=.05,
        curriculum_fraction=.8,
        curriculum_updates=curriculum_updates,
    )


def _weights_fingerprint(trainer: Trainer) -> str:
    digest = hashlib.sha256()
    for model in trainer.models:
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _run_profile(cfg: Config, *, games: int, eval_envs: int, eval_seed: int) -> dict:
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(cfg)
    initial_path = run / "initial.pt"
    final_path = run / "final.pt"
    trainer.save(initial_path)
    fingerprint = _weights_fingerprint(trainer)
    last_stats = None
    combat_totals = {"shots": 0, "hits": 0, "headshots": 0}
    progress_path = run / "progress.json"
    for index in range(cfg.updates):
        last_stats = trainer.iteration()
        combat_totals["shots"] += int(last_stats["update_shots"])
        combat_totals["hits"] += int(last_stats["update_hits"])
        combat_totals["headshots"] += int(last_stats["update_headshots"])
        progress = {
            "state": "training" if index + 1 < cfg.updates else "training_complete",
            "update": index + 1,
            "updates": cfg.updates,
            "consumed_env_steps": trainer.env_steps,
            "requested_env_steps": cfg.envs * cfg.horizon * cfg.updates,
            "training_combat_totals": combat_totals,
            "last_training_stats": last_stats,
        }
        _write_json_atomic(progress_path, progress)
        if (index + 1) % 16 == 0 or index + 1 == cfg.updates:
            print(f"[{Path(cfg.run_dir).name}] update {index + 1}/{cfg.updates} "
                  f"env_steps={trainer.env_steps} shots={combat_totals['shots']} "
                  f"hits={combat_totals['hits']} headshots={combat_totals['headshots']}",
                  flush=True)
    if trainer.env_steps != cfg.envs * cfg.horizon * cfg.updates:
        raise RuntimeError("Trainer did not consume the requested environment-step budget")
    trainer.save(final_path)
    evaluation_args = dict(games=games, envs=eval_envs, device=cfg.device,
                           seed=eval_seed, map_path=cfg.map_path)
    initial_eval = evaluate_both(str(initial_path), str(initial_path), **evaluation_args)
    final_eval = evaluate_both(str(final_path), str(initial_path), **evaluation_args)
    return {
        "config": asdict(cfg),
        "initial_checkpoint": str(initial_path),
        "final_checkpoint": str(final_path),
        "initial_weights_fingerprint": fingerprint,
        "consumed_env_steps": trainer.env_steps,
        "training_combat_totals": combat_totals,
        "last_training_stats": last_stats,
        "evaluation_seed": eval_seed,
        "initial_evaluation": initial_eval,
        "final_evaluation": final_eval,
    }


def compare(*, seeds: list[int], env_steps: int, games: int, eval_envs: int,
            hidden: int, device: str, action_steps: int, map_path: str,
            run_dir: str, output: str) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if min(games, eval_envs) <= 0:
        raise ValueError("games and eval_envs must be positive")
    if env_steps % 512:
        raise ValueError("env_steps must be a positive multiple of 512 for equal exact budgets")
    root = Path(run_dir)
    report_path = Path(output)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite comparison run directory: {root}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    root.mkdir(parents=True)

    report = {
        "experiment": "equal_env_steps_frozen_initial_opponent",
        "scale_label": "smoke_only_not_production" if hidden == 64 else "custom_scale",
        "warning": "Activity, entropy, or hit mechanics alone do not establish learned combat skill.",
        "env_steps_per_profile_seed": env_steps,
        "games_per_identity_evaluation": games,
        "evaluation": "same seed before/after; raw learned-policy sampling; full map; shaping and curriculum off",
        "seeds": seeds,
        "runs": {},
    }
    _write_json_atomic(report_path, report)
    for seed in seeds:
        order = list(PROFILE_NAMES)
        random.Random(seed).shuffle(order)
        seed_result = {"execution_order": order, "profiles": {}}
        fingerprints = []
        for name in order:
            cfg = profile_config(
                name, seed=seed, hidden=hidden, env_steps=env_steps, device=device,
                run_dir=root / f"seed-{seed}" / name, action_steps=action_steps,
                map_path=map_path)
            result = _run_profile(cfg, games=games, eval_envs=eval_envs,
                                  eval_seed=seed + 1_000_000)
            seed_result["profiles"][name] = result
            fingerprints.append(result["initial_weights_fingerprint"])
            report["runs"][str(seed)] = seed_result
            _write_json_atomic(report_path, report)
        if len(set(fingerprints)) != 1:
            raise RuntimeError("Profiles did not start from identical model weights")
        seed_result["shared_initial_weights_fingerprint"] = fingerprints[0]
        report["runs"][str(seed)] = seed_result
        _write_json_atomic(report_path, report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29])
    parser.add_argument("--env-steps", type=int, default=4096,
                        help="Exact budget per profile and seed; positive multiple of 512.")
    parser.add_argument("--games", type=int, default=16,
                        help="Games per candidate identity for each initial/final evaluation.")
    parser.add_argument("--eval-envs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64,
                        help="64 is smoke-only; pass 768 for the production architecture.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--action-steps", type=int, default=201)
    parser.add_argument("--map-path", default=str(DEFAULT_MAP))
    parser.add_argument("--run-dir", default="runs/learning-comparison")
    parser.add_argument("--output", default="learning-comparison.json")
    args = vars(parser.parse_args())
    result = compare(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
