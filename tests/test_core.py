"""Run: python -m pytest -q tests/test_core.py"""
from pathlib import Path
import copy
import math
import inspect
import torch
import pytest
from flyfight.environment import Arena,load_map
from flyfight.model import FlyPolicy,action_stats
from flyfight.training import Config,Trainer,generalized_advantage
from flyfight.spectator import Spectator

torch.set_num_threads(2)

def neutral(n=1):
    a=torch.zeros(n,2,4,dtype=torch.long);a[...,1]=4;a[...,2]=4
    return a


def fixture_arena():
    e=Arena(1,width=12,height=8)
    # An open segment at z=3; no map cover crosses x=-2..2 here.
    e.pos[0]=torch.tensor([[-2.,3.],[2.,3.]])
    e.yaw[0]=torch.tensor([math.pi/2,-math.pi/2]);e.pitch.zero_()
    return e


def test_map_dimensions_and_symmetry():
    s=load_map();assert (s['width'],s['depth'],len(s['obstacles']))==(64,48,16)
    for b in s['obstacles']:
        assert any(v['x']==-b['x'] and v['z']==-b['z'] and all(v[k]==b[k] for k in ('w','d','h')) for v in s['obstacles'])


def test_rgb_observation_shape_dtype():
    e=Arena(8,width=24,height=16)
    obs=e.observe();assert obs.shape==(8,2,16,24,3) and obs.dtype==torch.uint8
    assert obs.unique().numel()>5


def test_spawns_valid_separated():
    e=Arena(256,width=8,height=6)
    assert not e.blocked(e.pos).any()
    assert (torch.linalg.vector_norm(e.pos[:,0]-e.pos[:,1],dim=-1)>=6).all()


def test_policy_has_no_coordinate_argument():
    assert list(inspect.signature(FlyPolicy.forward).parameters)==['self','rgb','previous_action','hidden']


def test_model_shapes_probabilities_finite():
    m=FlyPolicy(12,8,48);e=Arena(4,width=12,height=8)
    logits,v,h=m(e.observe()[:,0],e.last_action[:,0],torch.zeros(4,48))
    a,lp,ent=action_stats(logits)
    assert a.shape==(4,4) and v.shape==(4,) and h.shape==(4,48)
    assert torch.isfinite(lp).all() and (ent>0).all() and h.abs().max()<=1


def test_turn_and_pitch_decode_to_quarter_steps():
    actions=torch.tensor([[[0,0,1,0],[0,7,6,1]]])
    decoded=Arena.decode(actions)
    assert decoded[0,0,2:].tolist()==[-1.,-.75,0.]
    assert decoded[0,1,2:].tolist()==[.75,.5,1.]


def test_masked_reset_does_not_change_other_arena():
    e=Arena(2,width=8,height=6)
    original=e.pos[1].clone();e.reset(torch.tensor([True,False]))
    assert torch.equal(e.pos[1],original)


def test_passive_actions_zero_reward():
    e=fixture_arena();r,d,_=e.step(neutral(),auto_reset=False)
    assert r.sum()==0 and not r.any() and not d.any()


def test_body_hit_deals_damage_without_shaped_reward():
    e=fixture_arena();a=neutral();a[0,0,3]=1
    r,d,i=e.step(a,auto_reset=False)
    assert not d[0] and not r.any() and e.hp[0,1]==66
    assert i['hits'][0,0] and not i['headshots'][0,0] and i['damage'][0,0]==34


def test_headshot_is_terminal_zero_sum():
    e=fixture_arena();e.pitch[0,0]=math.atan2(e.head_y-e.eye_y,4);a=neutral();a[0,0,3]=1
    r,d,i=e.step(a,auto_reset=False)
    assert d[0] and r.tolist()==[[1.,-1.]] and e.hp[0,1]==0
    assert i['headshots'][0,0] and i['damage'][0,0]==100


def test_headshot_bonus_preserves_outcome_and_score():
    e=fixture_arena();e.headshot_bonus=.375
    e.pitch[0,0]=math.atan2(e.head_y-e.eye_y,4);a=neutral();a[0,0,3]=1
    r,d,i=e.step(a,auto_reset=False)
    assert d[0] and r.tolist()==[[1.375,-1.375]]
    assert i['outcome'].tolist()==[[1.,-1.]]
    assert i['headshot_reward'].tolist()==[[.375,-.375]]
    assert e.score.tolist()==[[1,0]]


def test_simultaneous_hits_are_draw():
    e=fixture_arena();e.pitch[0]=math.atan2(e.head_y-e.eye_y,4);a=neutral();a[:,:,3]=1
    r,d,_=e.step(a,auto_reset=False)
    assert d[0] and not r.any() and not e.hp.any()


def test_wall_blocks_shot():
    e=fixture_arena();e.pos[0]=torch.tensor([[-12.,-13.],[-12.,-5.]])
    e.yaw[0]=torch.tensor([0.,math.pi]);a=neutral();a[:,:,3]=1
    r,d,info=e.step(a,auto_reset=False)
    assert not info['hits'].any() and not r.any()


def test_timeout_zero_reward_and_reset():
    e=Arena(4,width=8,height=6,episode_seconds=.1)
    r,d,_=e.step(neutral(4))
    assert d.all() and not r.any() and (e.steps==0).all() and (e.rounds==2).all()


def test_wall_collision():
    e=fixture_arena();e.pos[0,0]=torch.tensor([31.28,3.]);e.yaw[0,0]=math.pi/2
    a=neutral();a[0,0,0]=1;before=e.pos[0,0].clone()
    e.step(a,auto_reset=False)
    assert torch.equal(e.pos[0,0],before)


def test_parallel_slab_ray_inside_outside():
    e=fixture_arena()
    o=torch.tensor([[-12.,1.95,-13.],[0.,1.95,0.]])
    dr=torch.tensor([[[0.,0.,1.]],[[0.,1.,0.]]])
    distance,_=e.cast_boxes(o,dr)
    assert abs(float(distance[0,0])-3)<1e-6 and float(distance[1,0])==e.max_range


def test_gae_does_not_bootstrap_terminal():
    r=torch.tensor([[1.],[0.]]);d=torch.tensor([[True],[True]]);v=torch.zeros_like(r)
    adv,ret=generalized_advantage(r,d,v,torch.tensor([999.]),.999,.97)
    assert torch.equal(ret,r)


def small_config(**kwargs):
    base=dict(envs=4,hidden=48,width=12,height=8,horizon=4,epochs=1,minibatch_envs=4,threads=2,episode_seconds=.4)
    base.update(kwargs);return Config(**base)


def test_identical_start_independent_parameters():
    t=Trainer(small_config())
    for a,b in zip(t.models[0].parameters(),t.models[1].parameters()):
        assert torch.equal(a,b) and a.data_ptr()!=b.data_ptr()


def test_full_rnn_encoder_actor_critic_update():
    t=Trainer(small_config())
    before={k:v.clone() for k,v in t.models[0].state_dict().items()}
    stats=t.iteration()
    after=t.models[0].state_dict()
    for name in ('encoder.weight','recurrent.weight','actor.weight','critic.weight'):
        assert not torch.equal(before[name],after[name]), name
    assert math.isfinite(stats['loss']) and stats['env_steps']==16


def test_checkpoint_exact_resume_next_update(tmp_path):
    t=Trainer(small_config());t.iteration();p=tmp_path/'check.pt';t.save(p)
    t.iteration();expected={k:v.clone() for k,v in t.models[0].state_dict().items()}
    other=Trainer(small_config(resume=str(p)));other.iteration()
    assert all(torch.equal(v,other.models[0].state_dict()[k]) for k,v in expected.items())


def test_resume_dimensions_fail_closed(tmp_path):
    t=Trainer(small_config());p=tmp_path/'x.pt';t.save(p)
    with pytest.raises(ValueError,match='envs'):
        Trainer(small_config(envs=2,resume=str(p)))


def test_spectator_activity_matches_actor_hidden():
    t=Trainer(small_config());s=Spectator(t.cfg.map_path);s.offer(t.weight_blob());frame=s.tick()
    assert s.hello()['map']['width']==64
    assert frame['agents'][0]['activity']==s.h[0][0,s.display_indices].abs().tolist()
    assert frame['input_t']<frame['t'] and len(frame['agents'][0]['observation']['rgb_base64'])>0


def test_spectator_does_not_mutate_train_weights():
    t=Trainer(small_config());before={k:v.clone() for k,v in t.models[0].state_dict().items()}
    s=Spectator(t.cfg.map_path);s.offer(t.weight_blob());s.tick()
    assert all(torch.equal(v,t.models[0].state_dict()[k]) for k,v in before.items())


def test_new_policy_only_on_round_boundary():
    t=Trainer(small_config());s=Spectator(t.cfg.map_path);s.offer(t.weight_blob());s.tick()
    t.iteration();s.offer(t.weight_blob());assert s.version==0 and s.pending is not None
    for _ in range(100):
        s.tick()
        if s.version==1: break
    assert s.version==1
