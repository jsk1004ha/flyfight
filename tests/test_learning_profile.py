from dataclasses import replace
import hashlib
import json

import pytest
import torch

from flyfight.learning_profile import prepare_combat_run
from flyfight.training import Config, Trainer


def test_upgrade_preserves_source_and_transfers_without_old_optimizer(tmp_path):
    base = Config(envs=2, hidden=24, width=12, height=8, horizon=4,
                  minibatch_envs=2, device="cpu", run_dir=str(tmp_path))
    source = tmp_path / "latest.pt"
    trainer = Trainer(base)
    trainer.iteration()
    trainer.save(source)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    cfg = prepare_combat_run(tmp_path, replace(base, resume=str(source)))
    upgraded = Trainer(cfg)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    assert cfg.action_steps == base.action_steps and cfg.horizon == 64
    assert upgraded.update == 0 and upgraded.env_steps == 0
    for old, new, optimizer in zip(trainer.models, upgraded.models, upgraded.optimizers):
        assert torch.equal(old.encoder.weight, new.encoder.weight)
        assert torch.equal(old.recurrent.weight, new.recurrent.weight)
        assert torch.allclose(old.actor.weight * .05, new.actor.weight)
        assert torch.count_nonzero(new.critic.weight) == 0
        assert not optimizer.state
    record = json.loads((tmp_path / "combat-v1" / "transfer.json").read_text())
    assert record["source_sha256"] == original and record["mode"] == "transfer"
    again = prepare_combat_run(tmp_path, base)
    assert again.resume == cfg.resume  # idempotent, never re-soften the actor


def test_upgrade_refuses_partial_destination(tmp_path):
    (tmp_path / "combat-v1").mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        prepare_combat_run(tmp_path, Config(device="cpu"))


def test_fresh_combat_run_has_precise_actions_and_no_source(tmp_path):
    base = Config(envs=2, hidden=24, width=12, height=8, minibatch_envs=2,
                  device="cpu", action_steps=201)
    cfg = prepare_combat_run(tmp_path, base)
    assert cfg.action_steps == 201 and cfg.exploration_mix == .1
    assert cfg.headshot_bonus == .25 and cfg.damage_reward == .5
    assert cfg.curriculum_fraction == .8
    Trainer(cfg).iteration()
