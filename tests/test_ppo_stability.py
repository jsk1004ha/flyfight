import math

import pytest
import torch

from flyfight.model import action_stats, exploration_regularizer
from flyfight.training import Config, Trainer


def small_config(**kwargs):
    values = dict(envs=2, hidden=24, width=12, height=8, horizon=2, epochs=1,
                  minibatch_envs=2, threads=1, episode_seconds=.2)
    values.update(kwargs)
    return Config(**values)


def test_uniform_mixture_uses_same_exact_logprob_for_extreme_logits():
    logits = torch.tensor([[1000., -1000., -1000.]]).expand(3, -1)
    actions = torch.tensor([[0], [1], [2]])
    _, logp, entropy = action_stats(
        logits, actions, heads=(3,), exploration_mix=.3)
    expected = torch.tensor([.8, .1, .1]).log()
    torch.testing.assert_close(logp, expected)
    assert torch.isfinite(entropy).all()
    assert entropy[0].item() == pytest.approx(-(.8*math.log(.8)+.2*math.log(.1)))


def test_uniform_prior_recovers_saturated_logits_with_finite_gradient():
    logits = torch.tensor([[1000., -1000., -1000.]], requires_grad=True)
    regularizer = exploration_regularizer(logits, heads=(3,)).mean()
    regularizer.backward()
    assert torch.isfinite(regularizer)
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, torch.tensor([[2/3, -1/3, -1/3]]))


def test_advantage_floor_does_not_amplify_tiny_policy_signal():
    trainer = Trainer(small_config(entropy=0, advantage_floor=.1))
    data = trainer.collect()
    tiny = torch.linspace(-1e-12, 1e-12, data["advantages"].numel(),
                          device=trainer.device).view_as(data["advantages"])
    data["advantages"] = tiny
    data["returns"] = data["values"].clone()
    before = [model.actor.weight.detach().clone() for model in trainer.models]
    trainer.optimize(data)
    for model, old in zip(trainer.models, before):
        assert (model.actor.weight-old).abs().max().item() < 1e-8


def test_target_kl_stops_before_mutating_parameters():
    trainer = Trainer(small_config(entropy=0, target_kl=.001))
    data = trainer.collect()
    data["logps"] = data["logps"]-10
    before = [{name:value.detach().clone() for name,value in model.state_dict().items()}
              for model in trainer.models]
    stats = trainer.optimize(data)
    assert stats["kl_early_stops"] == 2 and stats["approx_kl"] > 1
    assert stats["loss"] == 0 and stats["grad_norm"] == 0
    for model, old in zip(trainer.models, before):
        assert all(torch.equal(value, old[name]) for name,value in model.state_dict().items())


def test_telemetry_separates_raw_and_behavior_entropy():
    trainer = Trainer(small_config(exploration_mix=.1, target_kl=.001))
    for model in trainer.models:
        model.actor.weight.data.zero_()
        model.actor.bias.data.fill_(-1000)
        offset = 0
        for size in model.heads:
            model.actor.bias.data[offset] = 1000
            offset += size
    data = trainer.collect()
    data["logps"] = data["logps"]-10  # stop before changing the deliberately collapsed actor
    stats = trainer.optimize(data)
    assert stats["raw_entropy"] < 1e-5
    assert stats["entropy"] > 1
    assert stats["raw_entropy_normalized"] < stats["entropy_normalized"]
    assert len(stats["raw_entropy_heads"]) == len(stats["entropy_heads"]) == 4
    assert stats["exploration_mix"] == .1


def test_nonbatched_sequence_matches_batched_with_exploration_prior():
    shared = dict(exploration_mix=.1, exploration_prior=.02, entropy=.01)
    batched = Trainer(small_config(batch_sequence_encoder=True, **shared))
    per_step = Trainer(small_config(batch_sequence_encoder=False, **shared))
    data = batched.collect()
    torch.manual_seed(1234)
    batched_stats = batched.optimize(data)
    torch.manual_seed(1234)
    per_step_stats = per_step.optimize(data)
    for left, right in zip(batched.models, per_step.models):
        for name, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[name], rtol=2e-5, atol=2e-6)
    assert batched_stats["raw_entropy"] == pytest.approx(per_step_stats["raw_entropy"], rel=1e-6)
    assert batched_stats["entropy"] == pytest.approx(per_step_stats["entropy"], rel=1e-6)


@pytest.mark.parametrize("changed", [
    {"exploration_mix": .1}, {"exploration_prior": .01}, {"target_kl": .02},
    {"advantage_floor": .1}, {"damage_reward": .01}, {"aim_reward": .01},
    {"curriculum_fraction": .5}, {"curriculum_updates": 100},
])
def test_resume_rejects_changed_learning_semantics(tmp_path, changed):
    checkpoint = tmp_path/"legacy-semantics.pt"
    Trainer(small_config()).save(checkpoint)
    with pytest.raises(ValueError, match="Resume requires same"):
        Trainer(small_config(resume=str(checkpoint), **changed))


def test_curriculum_schedule_restores_from_resumed_update(tmp_path):
    cfg = small_config(curriculum_fraction=.8, curriculum_updates=10)
    trainer = Trainer(cfg)
    trainer.update = 5
    checkpoint = tmp_path/"curriculum.pt"
    trainer.save(checkpoint)
    restored = Trainer(small_config(curriculum_fraction=.8, curriculum_updates=10,
                                    resume=str(checkpoint)))
    restored.collect()
    assert restored.env.curriculum_fraction == pytest.approx(.4)
