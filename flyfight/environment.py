"""Batched, on-device arena physics and RGB ray casting.

Only ``observe()`` RGB and the previous action enter a policy. Positions, hit
checks and map geometry belong to the environment, not to the actor/critic.
The renderer uses opaque boxes and a sphere for the opposing fly's hit volume;
it does NOT pretend to reproduce the detailed WebGL fly mesh pixel-for-pixel.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from .actions import ActionSpec, DEFAULT_ACTION_STEPS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "maps" / "arena_large.json"


def load_map(path: str | Path = DEFAULT_MAP) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("width", "depth", "wall_height"):
        if not isinstance(spec.get(key), (int, float)) or not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"Invalid map {key}")
    if not isinstance(spec.get("obstacles"), list) or len(spec["obstacles"]) > 64:
        raise ValueError("The map must have at most 64 box obstacles")
    for b in spec["obstacles"]:
        if not all(isinstance(b.get(k), (int, float)) and math.isfinite(b[k]) for k in ("x", "z", "w", "d", "h")):
            raise ValueError("Invalid obstacle")
        if min(b["w"], b["d"], b["h"]) <= 0:
            raise ValueError("Obstacle dimensions must be positive")
        if abs(b["x"]) + b["w"]/2 >= spec["width"]/2 or abs(b["z"]) + b["d"]/2 >= spec["depth"]/2:
            raise ValueError("Obstacle outside arena")
    return spec


class Arena:
    """Independent two-agent arenas; all hot-path tensors stay on one device."""
    radius = .70
    eye_y = 1.95
    body_y = 1.45
    body_radius = .72
    head_y = 2.30
    head_radius = .34
    body_damage = 34.
    speed = 4.5
    turn_speed = 2.0
    pitch_speed = 1.0
    cooldown_seconds = .4
    action_sizes = ActionSpec().heads

    def __init__(self, count: int = 128, device: str | torch.device = "cpu", *,
                 width: int = 24, height: int = 16, dt: float = .1,
                 episode_seconds: float = 60, seed: int = 17, map_path: str | Path = DEFAULT_MAP,
                 action_steps: int = DEFAULT_ACTION_STEPS,
                 turn_speed: float = 2.0, pitch_speed: float = 1.0,
                 headshot_bonus: float = 0.0,
                 damage_reward: float = 0.0, aim_reward: float = 0.0,
                 shaping_gamma: float = .999, curriculum_fraction: float = 0.0):
        if not 1 <= count <= 4096 or not 8 <= width <= 128 or not 6 <= height <= 96:
            raise ValueError("Invalid count / observation dimensions")
        if not 0 < dt <= .1 or episode_seconds < dt:
            raise ValueError("dt must be (0, 0.1]; episode_seconds must be >= dt")
        self.count, self.device = count, torch.device(device)
        if not math.isfinite(headshot_bonus) or not 0 <= headshot_bonus <= 10:
            raise ValueError("headshot_bonus must be finite and in [0, 10]")
        self.headshot_bonus = float(headshot_bonus)
        for name, value in (("damage_reward", damage_reward), ("aim_reward", aim_reward)):
            if not math.isfinite(value) or not 0 <= value <= 10:
                raise ValueError(f"{name} must be finite and in [0, 10]")
            setattr(self, name, float(value))
        if not math.isfinite(shaping_gamma) or not 0 <= shaping_gamma <= 1:
            raise ValueError("shaping_gamma must be finite and in [0, 1]")
        self.shaping_gamma = float(shaping_gamma)
        self.curriculum_fraction = self._validated_curriculum_fraction(curriculum_fraction)
        for name, value in (("turn_speed", turn_speed), ("pitch_speed", pitch_speed)):
            if not math.isfinite(value) or not 0 < value <= 20:
                raise ValueError(f"{name} must be finite and in (0, 20] radians/second")
            setattr(self, name, float(value))
        self.width, self.height, self.dt = width, height, dt
        self.action_spec = ActionSpec(action_steps)
        self.action_sizes = self.action_spec.heads
        self.spec = load_map(map_path)
        self.max_steps = math.ceil(episode_seconds / dt)
        self.episode_seconds = self.max_steps * dt
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.max_range = math.hypot(self.spec["width"], self.spec["depth"]) + 5
        self.bounds = torch.tensor([self.spec["width"]/2, self.spec["depth"]/2], device=self.device)
        boxes = list(self.spec["obstacles"])
        w, d, h = self.spec["width"], self.spec["depth"], self.spec["wall_height"]
        boxes += [dict(x=0, z=sgn*(d/2+.5), w=w+2, d=1, h=h) for sgn in (-1, 1)]
        boxes += [dict(x=sgn*(w/2+.5), z=0, w=1, d=d+2, h=h) for sgn in (-1, 1)]
        self.box_min = torch.tensor([[b["x"]-b["w"]/2, 0, b["z"]-b["d"]/2] for b in boxes], device=self.device)
        self.box_max = torch.tensor([[b["x"]+b["w"]/2, b["h"], b["z"]+b["d"]/2] for b in boxes], device=self.device)
        # Visible colors, not labels supplied to the model. Depth is NOT an input channel.
        self.box_colors = torch.tensor([[.27+.025*(i%3), .39+.015*(i%4), .45+.02*(i%2)] for i in range(len(boxes))], device=self.device)
        self.target_colors = torch.tensor([[.30,.83,.90],[.92,.59,.28]], device=self.device)
        yy, xx = torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij")
        vfov = 1.15
        sx = ((xx.flatten()+.5)/width*2-1)*math.tan(vfov/2)*(width/height)
        sy = (1-(yy.flatten()+.5)/height*2)*math.tan(vfov/2)
        self.local_rays = torch.nn.functional.normalize(torch.stack([sx, sy, torch.ones_like(sx)], -1), dim=-1)
        self.spawn_pairs = self._spawn_bank(seed)
        self.curriculum_spawn_pairs = self._curriculum_spawn_bank() if self.curriculum_fraction else None
        shape = (count, 2)
        self.pos = torch.zeros(*shape, 2, device=self.device)
        self.yaw = torch.zeros(shape, device=self.device)
        self.pitch = torch.zeros_like(self.yaw)
        self.hp = torch.full_like(self.yaw, 100)
        self.cooldown = torch.zeros_like(self.yaw)
        self.last_action = torch.zeros(*shape, 5, device=self.device)
        self.steps = torch.zeros(count, dtype=torch.long, device=self.device)
        self.rounds = torch.ones_like(self.steps)
        self.score = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.reset(torch.ones(count, dtype=torch.bool, device=self.device), first=True)

    @staticmethod
    def _validated_curriculum_fraction(value: float) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("curriculum_fraction must be finite and in [0, 1]")
        return float(value)

    def set_curriculum_fraction(self, value: float) -> None:
        """Change the mix used by future resets without touching active episodes."""
        fraction = self._validated_curriculum_fraction(value)
        if fraction and self.curriculum_spawn_pairs is None:
            self.curriculum_spawn_pairs = self._curriculum_spawn_bank()
        self.curriculum_fraction = fraction

    def _spawn_bank(self, seed: int) -> torch.Tensor:
        r = np.random.default_rng(seed)
        xy = r.uniform([-self.spec["width"]/2+1.5, -self.spec["depth"]/2+1.5],
                       [self.spec["width"]/2-1.5, self.spec["depth"]/2-1.5], size=(24000,2))
        good = np.ones(len(xy), dtype=bool)
        for b in self.spec["obstacles"]:
            good &= ~((np.abs(xy[:,0]-b["x"]) < b["w"]/2+1.4) & (np.abs(xy[:,1]-b["z"]) < b["d"]/2+1.4))
        xy = xy[good]
        if len(xy) < 20:
            raise ValueError("Not enough free spawn space")
        pairs = xy[r.integers(0,len(xy),(10000,2))]
        pairs = pairs[np.linalg.norm(pairs[:,0]-pairs[:,1],axis=1) >= 6][:4096]
        if len(pairs) < 128:
            raise ValueError("Not enough separated spawn pairs")
        return torch.tensor(pairs, dtype=torch.float32, device=self.device)

    def _curriculum_spawn_bank(self) -> torch.Tensor:
        distance = torch.linalg.vector_norm(self.spawn_pairs[:,0]-self.spawn_pairs[:,1],dim=-1)
        nearby = self.spawn_pairs[(distance >= 6) & (distance <= 12)]
        if len(nearby) == 0:
            raise ValueError("Not enough nearby curriculum spawn pairs")
        origins = torch.cat([
            torch.stack([nearby[:,0,0],torch.full_like(nearby[:,0,0],self.eye_y),nearby[:,0,1]],-1),
            torch.stack([nearby[:,1,0],torch.full_like(nearby[:,1,0],self.eye_y),nearby[:,1,1]],-1),
        ])
        targets = torch.cat([
            torch.stack([nearby[:,1,0],torch.full_like(nearby[:,1,0],self.body_y),nearby[:,1,1]],-1),
            torch.stack([nearby[:,0,0],torch.full_like(nearby[:,0,0],self.body_y),nearby[:,0,1]],-1),
        ])
        delta = targets-origins
        target_distance = torch.linalg.vector_norm(delta,dim=-1)
        box_distance,_ = self.cast_boxes(origins,torch.nn.functional.normalize(delta,dim=-1)[:,None,:])
        visible = (box_distance[:,0]+1e-5 >= target_distance).view(2,-1).all(0)
        result = nearby[visible]
        if len(result) == 0:
            raise ValueError("Not enough unobstructed curriculum spawn pairs")
        return result

    @torch.no_grad()
    def reset(self, mask: torch.Tensor, *, first: bool = False) -> None:
        """Masked reset without .item(), dynamic indexing or a GPU/CPU sync."""
        choice = torch.randint(len(self.spawn_pairs), (self.count,), device=self.device, generator=self.generator)
        new_pos = self.spawn_pairs[choice]
        new_yaw = torch.rand(self.yaw.shape, device=self.device, generator=self.generator)*2*math.pi-math.pi
        if self.curriculum_fraction > 0:
            use_curriculum = torch.rand(self.count, device=self.device, generator=self.generator) < self.curriculum_fraction
            curriculum_choice = torch.randint(len(self.curriculum_spawn_pairs), (self.count,), device=self.device, generator=self.generator)
            curriculum_pos = self.curriculum_spawn_pairs[curriculum_choice]
            toward = curriculum_pos.flip(1)-curriculum_pos
            facing_yaw = torch.atan2(toward[...,0],toward[...,1])
            jitter = (torch.rand(self.yaw.shape, device=self.device, generator=self.generator)*2-1)*.15
            curriculum_yaw = facing_yaw+jitter
            new_pos = torch.where(use_curriculum[:,None,None],curriculum_pos,new_pos)
            new_yaw = torch.where(use_curriculum[:,None],curriculum_yaw,new_yaw)
        self.pos = torch.where(mask[:,None,None], new_pos, self.pos)
        self.yaw = torch.where(mask[:,None], new_yaw, self.yaw)
        self.pitch = torch.where(mask[:,None], 0., self.pitch)
        self.hp = torch.where(mask[:,None], 100., self.hp)
        self.cooldown = torch.where(mask[:,None], 0., self.cooldown)
        self.last_action = torch.where(mask[:,None,None], 0., self.last_action)
        self.steps = torch.where(mask, 0, self.steps)
        if not first:
            self.rounds += mask.long()

    def blocked(self, pos: torch.Tensor) -> torch.Tensor:
        outside = (pos.abs() > self.bounds-self.radius).any(-1)
        p = pos.unsqueeze(-2)
        lo = self.box_min[:,[0,2]]-self.radius
        hi = self.box_max[:,[0,2]]+self.radius
        inside = ((p > lo) & (p < hi)).all(-1).any(-1)
        return outside | inside

    def directions(self) -> tuple[torch.Tensor, torch.Tensor]:
        sy, cy, sp, cp = self.yaw.sin(), self.yaw.cos(), self.pitch.sin(), self.pitch.cos()
        forward = torch.stack([sy*cp, sp, cy*cp],-1).flatten(0,1)
        right = torch.stack([cy, torch.zeros_like(cy), -sy],-1).flatten(0,1)
        up = torch.stack([-sy*sp, cp, -cy*sp],-1).flatten(0,1)
        rays = (right[:,None]*self.local_rays[None,:,0:1] + up[:,None]*self.local_rays[None,:,1:2]
                + forward[:,None]*self.local_rays[None,:,2:3])
        p = self.pos.flatten(0,1)
        origins = torch.stack([p[:,0], torch.full_like(p[:,0],self.eye_y),p[:,1]],-1)
        return origins, rays

    def cast_boxes(self, origins: torch.Tensor, rays: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """Slab intersection, including rays parallel to slabs and rays inside boxes."""
        o, dr = origins[:,None,None,:], rays[:,:,None,:]
        parallel = dr.abs() < 1e-7
        safe = torch.where(parallel, torch.ones_like(dr), dr)
        a = (self.box_min[None,None]-o)/safe
        b = (self.box_max[None,None]-o)/safe
        near_axis, far_axis = torch.minimum(a,b), torch.maximum(a,b)
        inside = (o >= self.box_min[None,None]) & (o <= self.box_max[None,None])
        near_axis = torch.where(parallel, torch.where(inside, -torch.inf, torch.inf), near_axis)
        far_axis = torch.where(parallel, torch.where(inside, torch.inf, -torch.inf), far_axis)
        near, far = near_axis.amax(-1), far_axis.amin(-1)
        distance = torch.where((far >= near.clamp_min(0)) & (far > 0), near.clamp_min(0), self.max_range)
        return distance.min(-1)

    def cast_opponent_parts(self, origins: torch.Tensor, rays: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        enemy = self.pos.flip(1).flatten(0,1)
        distances = []
        for y,radius in ((self.body_y,self.body_radius),(self.head_y,self.head_radius)):
            center = torch.stack([enemy[:,0],torch.full_like(enemy[:,0],y),enemy[:,1]],-1)
            oc = origins-center
            b = (oc[:,None,:]*rays).sum(-1)
            c = oc.square().sum(-1)[:,None]-radius**2
            disc = b.square()-c
            t = -b-disc.clamp_min(0).sqrt()
            distances.append(torch.where((disc >= 0) & (t > 0),t,self.max_range))
        return distances[0],distances[1]

    def cast_opponent(self, origins: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        body,head = self.cast_opponent_parts(origins,rays)
        return torch.minimum(body,head)

    def _aim_potential(self) -> torch.Tensor:
        """Bounded visible-target alignment potential for each agent."""
        p = self.pos.flatten(0,1)
        origins = torch.stack([p[:,0],torch.full_like(p[:,0],self.eye_y),p[:,1]],-1)
        enemy = self.pos.flip(1).flatten(0,1)
        targets = torch.stack([enemy[:,0],torch.full_like(enemy[:,0],self.body_y),enemy[:,1]],-1)
        delta = targets-origins
        target_distance = torch.linalg.vector_norm(delta,dim=-1)
        toward = torch.nn.functional.normalize(delta,dim=-1)
        box_distance,_ = self.cast_boxes(origins,toward[:,None,:])
        visible = box_distance[:,0]+1e-5 >= target_distance
        sy,cy,sp,cp = self.yaw.sin(),self.yaw.cos(),self.pitch.sin(),self.pitch.cos()
        forward = torch.stack([sy*cp,sp,cy*cp],-1).flatten(0,1)
        alignment = (forward*toward).sum(-1).clamp(0,1)
        return (alignment*visible).reshape(self.count,2)

    @torch.no_grad()
    def observe(self) -> torch.Tensor:
        origins,rays = self.directions()
        bd, bi = self.cast_boxes(origins,rays)
        td = self.cast_opponent(origins,rays)
        floor_d = torch.where(rays[...,1] < -1e-6,
                              -origins[:,None,1]/rays[...,1].clamp_max(-1e-6), self.max_range)
        floor_hit = (floor_d < bd) & (floor_d < td)
        target_hit = (td < bd) & (td < floor_d)
        sky_hit = torch.minimum(torch.minimum(bd,td),floor_d) >= self.max_range
        xyz = origins[:,None,:]+rays*floor_d.clamp_max(self.max_range)[...,None]
        checker = ((xyz[...,0]/3).floor()+(xyz[...,2]/3).floor()).remainder(2)
        floor_color = torch.stack([.33+.07*checker,.43+.07*checker,.47+.07*checker],-1)
        rgb = self.box_colors[bi]
        rgb = torch.where(floor_hit[...,None], floor_color, rgb)
        opponent_color = self.target_colors.flip(0).repeat(self.count,1)[:,None,:]
        rgb = torch.where(target_hit[...,None], opponent_color, rgb)
        sky = torch.tensor([.08,.12,.17],device=self.device)
        rgb = torch.where(sky_hit[...,None],sky,rgb)
        dist = torch.minimum(torch.minimum(bd,td),floor_d)
        shade = (.40+.60/(1+dist*.035))[...,None]
        rgb = (rgb*shade*255).round().clamp(0,255).to(torch.uint8)
        return rgb.reshape(self.count,2,self.height,self.width,3)

    @staticmethod
    def decode(actions: torch.Tensor, action_steps: int = DEFAULT_ACTION_STEPS) -> torch.Tensor:
        spec = ActionSpec(action_steps)
        move, turn, pitch, shoot = actions.unbind(-1)
        forward = (move==1).float()-(move==2).float()
        side = (move==4).float()-(move==3).float()
        return torch.stack([forward,side,spec.decode_index(turn),
                            spec.decode_index(pitch),shoot.float()],-1)

    @torch.no_grad()
    def step(self, actions: torch.Tensor, *, auto_reset: bool = True) -> tuple[torch.Tensor,torch.Tensor,dict]:
        """Simultaneous combat with optional potential-based learning rewards."""
        if actions.shape != (self.count,2,4):
            raise ValueError(f"Expected actions {(self.count,2,4)}, got {tuple(actions.shape)}")
        previous_potential = self._aim_potential() if self.aim_reward else torch.zeros_like(self.hp)
        self.last_action = self.decode(actions,self.action_spec.precision_steps)
        a = self.last_action
        self.yaw = (self.yaw+a[...,2]*self.turn_speed*self.dt+math.pi).remainder(2*math.pi)-math.pi
        self.pitch = (self.pitch+a[...,3]*self.pitch_speed*self.dt).clamp(-.65,.65)
        sy,cy = self.yaw.sin(),self.yaw.cos()
        delta = torch.stack([sy*a[...,0]+cy*a[...,1],cy*a[...,0]-sy*a[...,1]],-1)*self.speed*self.dt
        # Slide along walls; reject mutual overlap simultaneously without favoring A.
        old = self.pos.clone()
        for axis in (0,1):
            proposal = self.pos.clone()
            proposal[...,axis] += delta[...,axis]
            self.pos = torch.where(self.blocked(proposal)[...,None],self.pos,proposal)
        overlap = torch.linalg.vector_norm(self.pos[:,0]-self.pos[:,1],dim=-1) < self.radius*2
        self.pos = torch.where(overlap[:,None,None],old,self.pos)
        p = self.pos.flatten(0,1)
        o = torch.stack([p[:,0],torch.full_like(p[:,0],self.eye_y),p[:,1]],-1)
        sy,cy,sp,cp = self.yaw.sin(),self.yaw.cos(),self.pitch.sin(),self.pitch.cos()
        dr = torch.stack([sy*cp,sp,cy*cp],-1).flatten(0,1)[:,None,:]
        bd,_ = self.cast_boxes(o,dr)
        body_d,head_d = self.cast_opponent_parts(o,dr)
        td = torch.minimum(body_d,head_d)
        floor_d = torch.where(dr[...,1] < -1e-6,-self.eye_y/dr[...,1].clamp_max(-1e-6),self.max_range)
        self.cooldown = (self.cooldown-self.dt).clamp_min(0)
        fired = (a[...,4] > .5) & (self.cooldown <= 1e-6) & (self.hp > 0)
        self.cooldown = torch.where(fired,self.cooldown_seconds,self.cooldown)
        hits = ((td < bd) & (td < floor_d)).reshape(self.count,2) & fired
        headshots = ((head_d < body_d) & (head_d < bd) & (head_d < floor_d)).reshape(self.count,2) & fired
        damage = torch.where(headshots,100.,self.body_damage).where(hits,0.)
        effective_damage = torch.minimum(damage,self.hp.flip(1))
        self.hp = (self.hp-damage.flip(1)).clamp_min(0)
        killed = self.hp <= 0
        self.steps += 1
        done = killed.any(-1) | (self.steps >= self.max_steps)
        reward_a = killed[:,1].float()-killed[:,0].float()
        rewards = torch.stack([reward_a,-reward_a],-1)
        self.score += (rewards > 0).long()
        outcome = rewards.clone()
        bonus = self.headshot_bonus * (headshots.float() - headshots.flip(1).float())
        damage_bonus = self.damage_reward * (effective_damage-effective_damage.flip(1))/100
        if self.aim_reward:
            next_potential = self._aim_potential()
            next_potential = torch.where(done[:,None],0.,next_potential)
            aim_bonus = self.aim_reward * (self.shaping_gamma*next_potential-previous_potential)
        else:
            aim_bonus = torch.zeros_like(rewards)
        rewards = rewards + bonus + damage_bonus + aim_bonus
        end = o+dr[:,0]*torch.minimum(torch.minimum(bd,td),floor_d)[:,0,None]
        reward_components = {"outcome":outcome,"headshot":bonus,"damage":damage_bonus,"aim":aim_bonus}
        info = {"outcome":outcome,"headshot_reward":bonus,"damage_reward":damage_bonus,"aim_reward":aim_bonus,
                "reward_components":reward_components,"fired":fired,"hits":hits,"headshots":headshots,"damage":damage,
                "effective_damage":effective_damage,"ray_start":o.reshape(self.count,2,3),
                "ray_end":end.reshape(self.count,2,3),"terminal_hp":self.hp.clone(),
                "timeout":(self.steps >= self.max_steps) & ~killed.any(-1)}
        if auto_reset:
            self.reset(done)
        return rewards,done,info
