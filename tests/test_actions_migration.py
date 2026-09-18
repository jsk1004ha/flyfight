from pathlib import Path

import pytest
import torch

from flyfight.actions import ActionSpec, LEGACY_HEADS, migrate_actor_tensor
from flyfight.environment import Arena
from flyfight.model import FlyPolicy, action_stats
from flyfight.training import Config, Trainer, convert_schema2_checkpoint


def small_config(**kwargs):
    values = dict(envs=2,hidden=24,width=12,height=8,horizon=2,epochs=1,
                  minibatch_envs=2,threads=1,episode_seconds=.2)
    values.update(kwargs)
    return Config(**values)


def test_configurable_precision_decodes_full_symmetric_range():
    arena = Arena(1,width=8,height=6,action_steps=17)
    actions = torch.tensor([[[0,0,8,0],[0,16,12,1]]])
    decoded = arena.decode(actions,arena.action_spec.precision_steps)
    assert decoded[0,0,2:].tolist() == [-1.,0.,0.]
    assert decoded[0,1,2:].tolist() == [1.,.5,1.]
    model = FlyPolicy(8,6,24,action_steps=17)
    logits = model.actor(torch.zeros(2,24))
    sampled,_,_ = action_stats(logits,heads=model.heads)
    assert model.heads == (5,17,17,2) and sampled.shape == (2,4)


@pytest.mark.parametrize("steps",[2,4,66,2003,True,201.0])
def test_action_steps_reject_invalid_values(steps):
    with pytest.raises(ValueError,match="action_steps"):
        ActionSpec(steps)


def legacy_checkpoint(path: Path) -> dict:
    model = FlyPolicy(12,8,24)
    state = model.state_dict()
    old_out = sum(LEGACY_HEADS)
    state["actor.weight"] = torch.arange(old_out*24,dtype=torch.float32).view(old_out,24)
    state["actor.bias"] = torch.arange(old_out,dtype=torch.float32)
    trainer = Trainer(small_config())
    data = {
        "schema":2,"config":{**vars(trainer.cfg),"action_steps":5},"map_hash":trainer.map_hash,
        "update":7,"env_steps":28,"games":3,"draws":1,"wins":[1,1],
        "models":[state,{k:v.clone() for k,v in state.items()}],"optimizers":[{},{}],
        "hidden":[h.clone() for h in trainer.h],
        "env":{k:getattr(trainer.env,k).clone() for k in
               ("pos","yaw","pitch","hp","cooldown","last_action","steps","rounds","score")},
        "env_rng":trainer.env.generator.get_state(),"cpu_rng":torch.get_rng_state(),"cuda_rng":[],
    }
    torch.save(data,path)
    return data


def test_schema2_conversion_transfers_core_and_interpolates_actor(tmp_path):
    source,destination = tmp_path/"legacy.pt",tmp_path/"converted.pt"
    old = legacy_checkpoint(source)
    metadata = convert_schema2_checkpoint(source,destination,action_steps=9)
    new = torch.load(destination,map_location="cpu",weights_only=True)
    assert new["schema"] == 3 and new["config"]["action_steps"] == 9
    assert new["optimizers"] is None and metadata["optimizer"] == "reset"
    assert metadata["resume_mode"] == "weights_only" and metadata["source_progress"]["update"] == 7
    for name in ("encoder.weight","encoder.bias","recurrent.weight","critic.weight","critic.bias"):
        assert torch.equal(new["models"][0][name],old["models"][0][name])
    old_actor,new_actor = old["models"][0]["actor.bias"],new["models"][0]["actor.bias"]
    assert torch.equal(new_actor[:5],old_actor[:5])
    assert torch.equal(new_actor[5:14:2],old_actor[5:10])
    assert torch.equal(new_actor[14:23:4],old_actor[10:13])
    assert torch.equal(new_actor[-2:],old_actor[-2:])
    resumed = Trainer(small_config(resume=str(destination)))
    assert resumed.update == 0 and not resumed.optimizers[0].state
    assert not resumed.h[0].any() and (resumed.env.rounds == 1).all()


def test_resume_rejects_precision_mismatch(tmp_path):
    trainer = Trainer(small_config(action_steps=17))
    checkpoint = tmp_path/"17.pt"
    trainer.save(checkpoint)
    with pytest.raises(ValueError,match="action_steps"):
        Trainer(small_config(action_steps=9,resume=str(checkpoint)))


@pytest.mark.parametrize('steps,spacing', [(201,.01),(2001,.001)])
def test_decimal_actions_change_physical_view(steps, spacing):
    arena = Arena(1,width=8,height=6,action_steps=steps,turn_speed=1.234,pitch_speed=.567)
    arena.yaw.zero_(); arena.pitch.zero_()
    mid = (steps-1)//2
    values = arena.action_spec.decode_index(torch.tensor([0,mid-1,mid,mid+1,steps-1]))
    torch.testing.assert_close(values,torch.tensor([-1.,-spacing,0.,spacing,1.]))
    arena.step(torch.tensor([[[0,mid+1,mid-1,0],[0,mid,mid,0]]]),auto_reset=False)
    assert arena.yaw[0,0].item() == pytest.approx(spacing*1.234*.1,abs=3e-7)
    assert arena.pitch[0,0].item() == pytest.approx(-spacing*.567*.1,abs=1e-8)


def test_precision_training_spectator_and_resume(tmp_path):
    from flyfight.spectator import Spectator
    cfg = small_config(action_steps=2001,turn_speed=1.234,pitch_speed=.567)
    trainer = Trainer(cfg)
    data = trainer.collect()
    trainer.optimize(data)
    watch = Spectator(cfg.map_path)
    watch.offer(trainer.weight_blob())
    assert watch.env.action_spec.precision_steps == 2001
    assert watch.env.turn_speed == 1.234 and watch.env.pitch_speed == .567
    checkpoint = tmp_path/'precise.pt'
    trainer.save(checkpoint)
    Trainer(small_config(action_steps=2001,turn_speed=1.234,pitch_speed=.567,resume=str(checkpoint)))
    with pytest.raises(ValueError,match='turn_speed'):
        Trainer(small_config(action_steps=2001,resume=str(checkpoint)))


@pytest.mark.parametrize('speed',[0,-1,float('nan'),float('inf'),21])
def test_invalid_view_speed(speed):
    with pytest.raises(ValueError,match='turn_speed'):
        Arena(1,turn_speed=speed)
