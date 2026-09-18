"""Equivalence and telemetry checks for batched PPO sequence encoding."""
import copy
import json
from pathlib import Path
import statistics
import time

import torch

from flyfight.model import FlyPolicy
from flyfight.training import Config,Trainer


def old_step_loop(model, rgb, previous, hidden, dones):
    logits,values,states = [],[],[]
    for t in range(len(rgb)):
        logit,value,hidden = model(rgb[t],previous[t],hidden)
        logits.append(logit); values.append(value); states.append(hidden)
        hidden = hidden*(~dones[t])[:,None]
    return torch.stack(logits),torch.stack(values),torch.stack(states)


def test_batched_encoder_matches_step_loop_outputs_and_gradients():
    torch.manual_seed(41)
    steps,batch,width,height,hidden_size = 5,4,12,8,32
    reference = FlyPolicy(width,height,hidden_size)
    batched = copy.deepcopy(reference)
    rgb = torch.randint(0,256,(steps,batch,height,width,3),dtype=torch.uint8)
    previous = torch.randn(steps,batch,5)
    hidden = torch.randn(batch,hidden_size)
    dones = torch.tensor([
        [False,False,False,False],
        [False,True,False,False],
        [False,False,False,True],
        [True,False,False,False],
        [False,False,True,False],
    ])

    expected = old_step_loop(reference,rgb,previous,hidden,dones)
    actual = batched.forward_sequence(rgb,previous,hidden,dones)
    for left,right in zip(expected,actual):
        torch.testing.assert_close(left,right,rtol=2e-6,atol=2e-6)

    expected_loss = sum(t.square().mean() for t in expected)
    actual_loss = sum(t.square().mean() for t in actual)
    expected_loss.backward(); actual_loss.backward()
    assert reference.state_dict().keys() == batched.state_dict().keys()
    for (name,left),(other_name,right) in zip(reference.named_parameters(),batched.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(left.grad,right.grad,rtol=3e-5,atol=3e-6)


def test_iteration_reports_collection_and_optimization_timings():
    cfg = Config(envs=4,hidden=32,width=12,height=8,horizon=3,epochs=1,
                 minibatch_envs=4,threads=2,episode_seconds=.3)
    stats = Trainer(cfg).iteration()
    assert stats["collection_seconds"] >= 0
    assert stats["optimization_seconds"] >= 0
    assert stats["collection_env_steps_s"] > 0
    assert abs(stats["seconds"]-(stats["collection_seconds"]+stats["optimization_seconds"])) < 1e-9
    assert stats["update_shots"] >= stats["update_hits"] >= stats["update_headshots"] >= 0
    assert 0 <= stats["update_hit_rate"] <= 1


def benchmark_sequence_encoder():
    """Run a controlled old/new optimize benchmark and write reviewable evidence."""
    common = dict(envs=32,hidden=128,width=24,height=16,horizon=16,epochs=2,
                  minibatch_envs=16,threads=4,episode_seconds=1.6,seed=73)
    source = Trainer(Config(**common))
    data = source.collect()
    initial = [copy.deepcopy(model.state_dict()) for model in source.models]

    def run(batch_sequence_encoder):
        trainer = Trainer(Config(**common,batch_sequence_encoder=batch_sequence_encoder))
        for model,state in zip(trainer.models,initial):
            model.load_state_dict(state)
        torch.manual_seed(991)
        started = time.perf_counter()
        trainer.optimize(data)
        return time.perf_counter()-started

    for _ in range(2):
        run(False); run(True)
    old_seconds,new_seconds = [],[]
    for _ in range(7):
        old_seconds.append(run(False)); new_seconds.append(run(True))
    old_median = statistics.median(old_seconds)
    new_median = statistics.median(new_seconds)
    result = {
        "device":"cpu","torch_version":torch.__version__,"warmups":2,"trials":7,
        "config":common,"old_step_loop_seconds":old_seconds,
        "batched_encoder_seconds":new_seconds,"old_step_loop_median_seconds":old_median,
        "batched_encoder_median_seconds":new_median,"speedup":old_median/new_median,
    }
    destination = Path(__file__).parents[1]/"evidence"/"training_sequence_benchmark.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    benchmark_sequence_encoder()
