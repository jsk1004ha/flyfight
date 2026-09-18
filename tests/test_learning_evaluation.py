from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import evaluate as evaluation
from flyfight.environment import DEFAULT_MAP
from flyfight.model import FlyPolicy
from learning_compare import profile_config


class _OneStepArena:
    def __init__(self, count, device, *, width, height, **kwargs):
        self.count = count
        self.device = torch.device(device)
        self.width = width
        self.height = height
        self.last_action = torch.zeros(count, 2, 5, device=self.device)

    def observe(self):
        return torch.zeros(self.count, 2, 3, self.height, self.width,
                           dtype=torch.uint8, device=self.device)

    def step(self, actions):
        done = torch.ones(self.count, dtype=torch.bool, device=self.device)
        fired = torch.ones(self.count, 2, dtype=torch.bool, device=self.device)
        hits = fired.clone()
        headshots = torch.zeros_like(fired)
        headshots[:, 1] = True
        outcome = torch.tensor([[1.0, -1.0]], device=self.device).repeat(self.count, 1)
        return torch.zeros_like(outcome), done, {
            "fired": fired,
            "hits": hits,
            "headshots": headshots,
            "outcome": outcome,
        }


def _checkpoint(path: Path, *, exploration_mix: float) -> None:
    config = {
        "width": 8,
        "height": 6,
        "hidden": 16,
        "dt": .1,
        "episode_seconds": 1.0,
        "action_steps": 9,
        "turn_speed": 2.0,
        "pitch_speed": 1.0,
        "exploration_mix": exploration_mix,
    }
    torch.manual_seed(7)
    models = [FlyPolicy(8, 6, 16, 9).state_dict() for _ in range(2)]
    torch.save({
        "schema": 3,
        "config": config,
        "map_hash": hashlib.sha256(Path(DEFAULT_MAP).read_bytes()).hexdigest(),
        "models": models,
    }, path)


def test_evaluate_reports_balanced_candidate_activity(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.pt"
    opponent = tmp_path / "opponent.pt"
    _checkpoint(candidate, exploration_mix=.1)
    _checkpoint(opponent, exploration_mix=.2)
    monkeypatch.setattr(evaluation, "Arena", _OneStepArena)

    result = evaluation.evaluate(str(candidate), str(opponent), games=4, envs=2,
                                 device="cpu", seed=123, identity="A")

    assert (result["wins"], result["losses"], result["draws"]) == (2, 2, 0)
    assert result["candidate_active_steps"] == 4
    assert result["candidate_shots"] == 4
    assert result["candidate_hits"] == 4
    assert result["candidate_headshots"] == 2
    assert result["candidate_shots_per_1000_active_steps"] == 1000
    assert result["candidate_headshots_per_1000_active_steps"] == 500
    assert result["candidate_arena_slot_steps"] == [2, 2]
    assert result["candidate_arena_slot_games"] == [2, 2]
    assert result["action_mode"] == "raw_policy_sample"
    assert result["applied_exploration_mix"] == {"candidate": 0.0, "opponent": 0.0}
    assert result["reward_shaping"] == "off"
    assert result["curriculum"] == "off_full_map"


def test_behavior_mix_is_explicit_and_uses_each_checkpoint(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.pt"
    opponent = tmp_path / "opponent.pt"
    _checkpoint(candidate, exploration_mix=.1)
    _checkpoint(opponent, exploration_mix=.2)
    monkeypatch.setattr(evaluation, "Arena", _OneStepArena)

    result = evaluation.evaluate(str(candidate), str(opponent), games=2, envs=2,
                                 device="cpu", behavior_mix=True)

    assert result["action_mode"] == "saved_behavior_mix"
    assert result["applied_exploration_mix"] == {"candidate": .1, "opponent": .2}
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluation.evaluate(str(candidate), str(opponent), games=2, envs=2,
                            device="cpu", behavior_mix=True, greedy=True)


def test_evaluate_both_keeps_identity_diagnostics(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    opponent = tmp_path / "opponent.pt"
    _checkpoint(checkpoint, exploration_mix=0)
    _checkpoint(opponent, exploration_mix=0)
    monkeypatch.setattr(evaluation, "Arena", _OneStepArena)

    result = evaluation.evaluate_both(str(checkpoint), str(opponent), games=4,
                                      envs=2, device="cpu", seed=9)

    assert result["candidate_identity"] == "both"
    assert result["games"] == 8
    assert result["candidate_active_steps"] == 8
    assert set(result["identities"]) == {"A", "B"}
    assert result["identities"]["A"]["candidate_identity"] == "A"
    assert result["identities"]["B"]["candidate_identity"] == "B"


def test_comparison_profiles_are_explicit_and_equal_budget(tmp_path):
    baseline = profile_config("baseline", seed=17, hidden=64, env_steps=4096,
                              device="cpu", run_dir=tmp_path / "baseline",
                              action_steps=201, map_path=str(DEFAULT_MAP))
    improved = profile_config("improved", seed=17, hidden=64, env_steps=4096,
                              device="cpu", run_dir=tmp_path / "improved",
                              action_steps=201, map_path=str(DEFAULT_MAP))

    assert baseline.envs * baseline.horizon * baseline.updates == 4096
    assert improved.envs * improved.horizon * improved.updates == 4096
    assert (baseline.horizon, baseline.learning_rate, baseline.entropy) == (16, 3e-4, .01)
    assert (improved.horizon, improved.learning_rate, improved.entropy) == (64, 1e-4, .03)
    assert improved.exploration_mix == .1
    assert improved.target_kl == .02
    assert improved.advantage_floor == .1
    assert improved.exploration_prior == .002
    assert improved.damage_reward == .5
    assert improved.aim_reward == .05
    assert improved.curriculum_fraction == .8
    assert improved.curriculum_updates == round(improved.updates * .8)
    with pytest.raises(ValueError, match="multiple"):
        profile_config("improved", seed=17, hidden=64, env_steps=128,
                       device="cpu", run_dir=tmp_path / "bad",
                       action_steps=201, map_path=str(DEFAULT_MAP))
