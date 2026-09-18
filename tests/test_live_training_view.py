import base64
import queue
import threading

import pytest
import torch

from flyfight.environment import Arena
from flyfight.training import Config, Trainer, bounded_put_latest


def live_config(**values):
    config = dict(envs=2, hidden=24, width=12, height=8, horizon=1, epochs=1,
                  minibatch_envs=2, threads=2, episode_seconds=10)
    config.update(values)
    return Config(**config)


def enable_live(trainer):
    trainer.live_queue = queue.Queue(maxsize=1)
    trainer.live_subscribers = threading.Event()
    trainer.live_subscribers.set()
    trainer.live_interval = 0


def test_live_frame_is_exact_sample_from_training_environment_zero():
    trainer = Trainer(live_config())
    enable_live(trainer)
    initial_hidden = [state.clone() for state in trainer.h]

    torch.manual_seed(1234)
    data = trainer.collect()
    frame = trainer.live_queue.get_nowait()

    assert frame["view_source"] == "training_live"
    assert frame["environment_index"] == 0
    assert frame["env_step"] == trainer.cfg.envs
    assert frame["update"] == trainer.update == 0
    decoded = Arena.decode(data["actions"][0, 0], trainer.cfg.action_steps)
    for i, agent in enumerate(frame["agents"]):
        assert agent["action"] == pytest.approx(decoded[i].tolist())
        raw = base64.b64decode(agent["observation"]["rgb_base64"])
        assert raw == data["rgb"][0, 0, i].cpu().numpy().tobytes()
        _, _, hidden = trainer.models[i](data["rgb"][0, :, i], data["prev"][0, :, i], initial_hidden[i])
        expected = hidden[0, trainer.live_view.display_indices]
        assert agent["signed_activity"] == pytest.approx(expected.clamp(-1, 1).tolist())
        assert agent["activity"] == pytest.approx(expected.abs().clamp(0, 1).tolist())
        assert agent["position"] == pytest.approx(trainer.env.pos[0, i].tolist())


def test_terminal_frame_precedes_environment_autoreset():
    trainer = Trainer(live_config(episode_seconds=.1))
    enable_live(trainer)
    trainer.collect()
    frame = trainer.live_queue.get_nowait()

    assert frame["result"] == "DRAW"
    assert frame["round"] == 1 and frame["round_time"] == pytest.approx(.1)
    assert int(trainer.env.rounds[0]) == 2 and int(trainer.env.steps[0]) == 0


def test_streaming_is_inactive_without_a_subscriber_and_does_not_change_rollout():
    baseline = Trainer(live_config(horizon=3))
    streamed = Trainer(live_config(horizon=3))
    enable_live(streamed)
    baseline.live_queue = queue.Queue(maxsize=1)
    baseline.live_subscribers = threading.Event()

    torch.manual_seed(99)
    expected = baseline.collect()
    torch.manual_seed(99)
    actual = streamed.collect()

    assert baseline.live_queue.empty()
    for key in ("rgb", "prev", "actions", "rewards", "dones", "outcomes"):
        assert torch.equal(expected[key], actual[key]), key
    assert torch.equal(baseline.env.pos, streamed.env.pos)
    assert torch.equal(baseline.env.hp, streamed.env.hp)


def test_latest_only_queue_replaces_stale_frame():
    output = queue.Queue(maxsize=1)
    bounded_put_latest(output, {"env_step": 1})
    bounded_put_latest(output, {"env_step": 2})
    assert output.get_nowait() == {"env_step": 2}


def test_live_hello_identifies_actual_training_source():
    hello = Trainer(live_config()).live_view.hello()
    assert hello["view_source"] == "training_live"
    assert hello["environment_index"] == 0
    assert "actual PPO rollout" in hello["match_source"]
    assert len(hello["neurons"]) == 24
