import math

import pytest
import torch

from flyfight.environment import Arena


def neutral(n=1):
    actions = torch.zeros(n,2,4,dtype=torch.long)
    actions[...,1:3] = 4
    return actions


def open_arena(**kwargs):
    arena = Arena(1,width=12,height=8,**kwargs)
    arena.pos[0] = torch.tensor([[-2.,3.],[2.,3.]])
    arena.yaw[0] = torch.tensor([math.pi/2,-math.pi/2])
    arena.pitch.zero_()
    return arena


def test_damage_reward_is_effective_zero_sum_and_independent_of_headshot_bonus():
    arena = open_arena(damage_reward=1.,headshot_bonus=.25)
    actions = neutral(); actions[0,0,3] = 1
    rewards,done,info = arena.step(actions,auto_reset=False)
    assert not done[0]
    torch.testing.assert_close(rewards,torch.tensor([[.34,-.34]]))
    torch.testing.assert_close(info["damage_reward"],torch.tensor([[.34,-.34]]))
    assert not info["headshot_reward"].any()

    arena = open_arena(damage_reward=.5,headshot_bonus=.25)
    arena.pitch[0,0] = math.atan2(arena.head_y-arena.eye_y,4)
    rewards,done,info = arena.step(actions,auto_reset=False)
    assert done[0] and info["outcome"].tolist() == [[1.,-1.]]
    torch.testing.assert_close(info["damage_reward"],torch.tensor([[.5,-.5]]))
    torch.testing.assert_close(info["headshot_reward"],torch.tensor([[.25,-.25]]))
    torch.testing.assert_close(rewards,torch.tensor([[1.75,-1.75]]))


def test_damage_reward_uses_effective_remaining_hp():
    arena = open_arena(damage_reward=1.)
    arena.hp[0,1] = 10
    actions = neutral(); actions[0,0,3] = 1
    _,_,info = arena.step(actions,auto_reset=False)
    torch.testing.assert_close(info["damage_reward"],torch.tensor([[.1,-.1]]))
    assert info["effective_damage"].tolist() == [[10.,0.]]


def test_wall_blocks_aim_credit():
    arena = Arena(1,width=12,height=8,aim_reward=1.)
    arena.pos[0] = torch.tensor([[-12.,-13.],[-12.,-5.]])
    arena.yaw[0] = torch.tensor([0.,math.pi])
    _,_,info = arena.step(neutral(),auto_reset=False)
    assert not info["aim_reward"].any()


def test_stationary_alignment_cannot_farm_positive_aim_reward():
    arena = open_arena(aim_reward=1.,shaping_gamma=.999)
    for _ in range(5):
        rewards,done,info = arena.step(neutral(),auto_reset=False)
        assert not done[0]
        assert (info["aim_reward"] <= 0).all()
        torch.testing.assert_close(rewards,info["aim_reward"])


def test_terminal_aim_potential_is_zero():
    arena = open_arena(aim_reward=1.)
    before = arena._aim_potential().clone()
    arena.pitch[0,0] = math.atan2(arena.head_y-arena.eye_y,4)
    before = arena._aim_potential().clone()
    actions = neutral(); actions[0,0,3] = 1
    _,done,info = arena.step(actions,auto_reset=False)
    assert done[0]
    torch.testing.assert_close(info["aim_reward"],-before)
    assert info["outcome"].tolist() == [[1.,-1.]]


def test_curriculum_spawns_are_near_visible_separated_and_face_each_other():
    arena = Arena(128,width=8,height=6,curriculum_fraction=1.,seed=93)
    distance = torch.linalg.vector_norm(arena.pos[:,0]-arena.pos[:,1],dim=-1)
    assert ((distance >= 6) & (distance <= 12)).all()
    assert not arena.blocked(arena.pos).any()
    assert (distance >= 2*arena.radius).all()
    assert (arena._aim_potential() > .98).all()


def test_curriculum_setter_does_not_reset_active_episode_and_zero_uses_original_rng_path():
    arena = Arena(4,width=8,height=6,seed=71)
    expected = Arena(4,width=8,height=6,seed=71,curriculum_fraction=0.)
    assert torch.equal(arena.pos,expected.pos) and torch.equal(arena.yaw,expected.yaw)
    before = arena.pos.clone()
    arena.set_curriculum_fraction(1.)
    assert torch.equal(arena.pos,before)
    arena.set_curriculum_fraction(0.)
    arena.reset(torch.ones(4,dtype=torch.bool))
    expected.reset(torch.ones(4,dtype=torch.bool))
    assert torch.equal(arena.pos,expected.pos) and torch.equal(arena.yaw,expected.yaw)


def test_legacy_map_does_not_require_curriculum_spawn_pairs(monkeypatch):
    def unavailable(_):
        raise ValueError("No nearby pairs")
    monkeypatch.setattr(Arena, "_curriculum_spawn_bank", unavailable)
    arena = Arena(1, width=8, height=6)
    assert arena.curriculum_spawn_pairs is None
    with pytest.raises(ValueError, match="nearby"):
        arena.set_curriculum_fraction(1.)


@pytest.mark.parametrize(("name","value"),[
    ("damage_reward",-1.),("damage_reward",float("nan")),
    ("aim_reward",11.),("aim_reward",float("inf")),
    ("shaping_gamma",-0.1),("shaping_gamma",1.1),
    ("curriculum_fraction",-0.1),("curriculum_fraction",1.1),
])
def test_invalid_combat_learning_values(name,value):
    with pytest.raises(ValueError,match=name):
        Arena(1,**{name:value})
