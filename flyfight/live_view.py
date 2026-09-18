"""Serialize one real PPO training arena for the browser viewer.

The serializer never advances an environment or runs a policy.  Every value in
a frame is copied from environment 0 of the rollout that is being used for PPO.
"""
from __future__ import annotations

import base64
import math

import numpy as np
import torch

from .environment import load_map


class LiveTrainingView:
    """Stable viewer metadata plus serialization of sampled training tensors."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.map = load_map(cfg.map_path)
        self.display_indices = np.linspace(
            0, cfg.hidden - 1, min(cfg.hidden, 4096), dtype=int
        )
        random = np.random.default_rng(100)
        lobes = [
            ((-1.67, .48, -.08), (1.08, .88, .68), .195),
            ((1.67, .48, .08), (1.08, .88, .68), .195),
            ((-.57, .73, .04), (.86, .73, .56), .18),
            ((.57, .73, -.04), (.86, .73, .56), .18),
            ((0, -1.05, .10), (.86, 1.16, .52), .25),
        ]
        self.neurons = []
        for shown, index in enumerate(self.display_indices):
            ratio = shown / max(len(self.display_indices), 1)
            cumulative, location_group = 0., len(lobes) - 1
            for candidate, (_, _, share) in enumerate(lobes):
                cumulative += share
                if ratio < cumulative:
                    location_group = candidate
                    break
            center, scale, _ = lobes[location_group]
            direction = random.normal(size=3)
            direction /= np.linalg.norm(direction)
            point = np.asarray(center) + direction * (random.random() ** (1 / 3)) * np.asarray(scale)
            if location_group == 4:
                point[0] *= .76 + .24 * np.clip((.25 - point[1]) / 1.5, 0, 1)
            if location_group < 2:
                color_group = 0 if shown % 5 == 0 else 1
            elif location_group < 4:
                color_group = 1 if shown % 6 == 0 else 0
            else:
                color_group = 1 if shown % 4 == 0 else 4 if shown % 7 == 0 else 0
            self.neurons.append(dict(
                id=f"syn-{index:04d}", x=float(point[0]), y=float(point[1]),
                z=float(point[2]), group=color_group,
            ))
        count = len(self.neurons)
        self.edges = random.integers(0, count, (min(8192, count * 2), 2)).tolist()

    def hello(self) -> dict:
        duration = math.ceil(self.cfg.episode_seconds / self.cfg.dt) * self.cfg.dt
        return {
            "type": "hello", "version": 1,
            "dataset": "FlyFight Native / live PPO training environment 0",
            "activity_source": "synthetic", "neurons": self.neurons, "edges": self.edges,
            "round_duration": duration, "map": self.map, "native": True,
            "observation": {
                "width": self.cfg.width, "height": self.cfg.height,
                "renderer": "analytic RGB raycaster",
            },
            "edge_note": "fixed subset of dense recurrent edges; positions are schematic",
            "match_source": "actual PPO rollout sampled from training environment 0",
            "view_source": "training_live", "environment_index": 0,
        }

    @torch.inference_mode()
    def frame(self, env, rgb: torch.Tensor, hidden: list[torch.Tensor], info: dict, *,
              update: int, env_step: int) -> dict:
        """Capture the post-action state before terminal auto-reset.

        ``rgb`` and ``hidden`` are the exact policy inputs/state from the action
        that produced this transition; ``env`` is the corresponding post-step
        physical state.  The caller must reset terminal arenas only afterwards.
        """
        observations = rgb[0].detach().cpu().numpy()
        positions = env.pos[0].detach().cpu().tolist()
        yaws = env.yaw[0].detach().cpu().tolist()
        pitches = env.pitch[0].detach().cpu().tolist()
        hp = env.hp[0].detach().cpu().tolist()
        actions = env.last_action[0].detach().cpu().tolist()
        score = env.score[0].detach().cpu().tolist()
        activities = []
        signed_activities = []
        indices = torch.as_tensor(self.display_indices, device=hidden[0].device)
        for state in hidden:
            selected = state[0].index_select(0, indices).detach().float().cpu()
            activities.append(selected.abs().clamp(0, 1).tolist())
            signed_activities.append(selected.clamp(-1, 1).tolist())

        events = []
        for i in range(2):
            if bool(info["fired"][0, i]):
                events.append({
                    "type": "shot", "agent": "AB"[i],
                    "start": info["ray_start"][0, i].detach().cpu().tolist(),
                    "end": info["ray_end"][0, i].detach().cpu().tolist(),
                    "hit": bool(info["hits"][0, i]),
                    "headshot": bool(info["headshots"][0, i]),
                    "damage": float(info["damage"][0, i]),
                })

        transition = env_step // env.count
        frame = {
            "type": "frame", "t": transition * env.dt,
            "input_t": max(0, transition - 1) * env.dt,
            "round": int(env.rounds[0]), "round_time": float(env.steps[0]) * env.dt,
            "score": score, "policy_version": update, "intermission": False,
            "events": events, "agents": [], "view_source": "training_live",
            "environment_index": 0, "env_step": env_step, "update": update,
        }
        for i in range(2):
            frame["agents"].append({
                "id": "AB"[i], "position": positions[i], "yaw": yaws[i],
                "pitch": pitches[i], "hp": hp[i], "action": actions[i],
                "activity": activities[i], "signed_activity": signed_activities[i],
                "observation": {
                    "width": env.width, "height": env.height,
                    "rgb_base64": base64.b64encode(observations[i].tobytes()).decode("ascii"),
                },
            })
        done = bool(info["terminal_hp"][0].min() <= 0 or env.steps[0] >= env.max_steps)
        if done:
            outcome = info["outcome"][0]
            frame["result"] = "A" if outcome[0] > 0 else "B" if outcome[1] > 0 else "DRAW"
        return frame
