"""A separate CPU exhibition match using actual learner snapshots.

This is NOT one of the accelerated training arenas. Its experience is not fed
back to the trainer. Version changes are applied at round boundaries, not in
the middle of a recurrent trajectory. All displayed activity comes from the
same policy that selects this match's actions, not decorative random flashes.
"""
from __future__ import annotations
import base64
import io
import math
import numpy as np
import torch
from .environment import Arena
from .model import FlyPolicy,action_stats


class Spectator:
    def __init__(self,map_path):
        self.map_path = map_path
        self.models = None
        self.pending = None
        self.version = -1
        self.clock = 0.
        self.hold = 0
        self.last_frame = None
        self.env = None
        self.generator = torch.Generator().manual_seed(271828)

    def offer(self,blob: bytes) -> None:
        self.pending = blob
        if self.models is None:
            self._install()

    def _install(self) -> None:
        if self.pending is None:
            return
        data = torch.load(io.BytesIO(self.pending),map_location="cpu",weights_only=True)
        c = data["config"]
        action_steps = int(c.get("action_steps",9))
        if self.models is None:
            self.env = Arena(1,"cpu",width=c["width"],height=c["height"],dt=c["dt"],
                             episode_seconds=c["episode_seconds"],seed=2718,map_path=self.map_path,
                             action_steps=action_steps,turn_speed=c.get("turn_speed",2.0),
                             pitch_speed=c.get("pitch_speed",1.0),headshot_bonus=c.get("headshot_bonus",0.0))
            self.models = [FlyPolicy(c["width"],c["height"],c["hidden"],action_steps).eval() for _ in range(2)]
            self.hidden = c["hidden"]
            r = np.random.default_rng(100)
            # The viewer contract accepts at most 4096 displayed units. Preserve every
            # actual recurrent unit whenever the policy fits inside that bound.
            self.display_indices = np.linspace(0,self.hidden-1,min(self.hidden,4096),dtype=int)
            self.neurons = []
            lobes = [((-1.67,.48,-.08),(1.08,.88,.68),.195),((1.67,.48,.08),(1.08,.88,.68),.195),
                     ((-.57,.73,.04),(.86,.73,.56),.18),((.57,.73,-.04),(.86,.73,.56),.18),
                     ((0,-1.05,.10),(.86,1.16,.52),.25)]
            for shown,index in enumerate(self.display_indices):
                ratio = shown/max(len(self.display_indices),1)
                cumulative,location_group = 0.,len(lobes)-1
                for candidate,(_,_,share) in enumerate(lobes):
                    cumulative += share
                    if ratio < cumulative:
                        location_group = candidate
                        break
                center,scale,_ = lobes[location_group]
                direction = r.normal(size=3); direction /= np.linalg.norm(direction)
                p = np.asarray(center)+direction*(r.random()**(1/3))*np.asarray(scale)
                if location_group == 4:
                    p[0] *= .76+.24*np.clip((.25-p[1])/1.5,0,1)
                if location_group < 2:
                    color_group = 0 if shown%5==0 else 1
                elif location_group < 4:
                    color_group = 1 if shown%6==0 else 0
                else:
                    color_group = 1 if shown%4==0 else 4 if shown%7==0 else 0
                self.neurons.append(dict(id=f"syn-{index:04d}",x=float(p[0]),y=float(p[1]),z=float(p[2]),group=color_group))
            n = len(self.neurons)
            # All recurrent matrix entries are real model edges; show a bounded fixed
            # subset so denser unit displays do not turn into an H^2 rendering cost.
            self.edges = r.integers(0,n,(min(8192,n*2),2)).tolist()
        for m,s in zip(self.models,data["models"]):
            m.load_state_dict(s)
        self.h = [torch.zeros(1,self.hidden) for _ in range(2)]
        self.version = int(data["update"])
        self.pending = None

    def hello(self) -> dict:
        return {"type":"hello","version":1,"dataset":"FlyFight Native / synthetic recurrent PPO",
                "activity_source":"synthetic","neurons":self.neurons,"edges":self.edges,
                "round_duration":self.env.episode_seconds,"map":self.env.spec,"native":True,
                "observation":{"width":self.env.width,"height":self.env.height,"renderer":"analytic RGB raycaster"},
                "edge_note":"fixed subset of dense recurrent edges; positions are schematic",
                "match_source":"separate CPU spectator match, latest snapshot at round boundaries"}

    @torch.inference_mode()
    def tick(self) -> dict | None:
        if self.models is None:
            return None
        if self.hold:
            self.hold -= 1
            self.clock += self.env.dt
            frame = {**self.last_frame,"t":self.clock,"events":[],"intermission":True}
            if self.hold == 0:
                self.env.reset(torch.ones(1,dtype=torch.bool))
                self._install()
                self.h = [torch.zeros(1,self.hidden) for _ in range(2)]
            return frame
        rgb = self.env.observe()
        observations = rgb.numpy()
        acts,activity,signed_activity = [],[],[]
        for i,model in enumerate(self.models):
            logits,_,self.h[i] = model(rgb[:,i],self.env.last_action[:,i],self.h[i])
            a,_,_ = action_stats(logits,generator=self.generator,heads=model.heads)
            acts.append(a)
            activity.append(self.h[i][0,self.display_indices].abs().clamp(0,1).tolist())
            signed_activity.append(self.h[i][0,self.display_indices].clamp(-1,1).tolist())
        actions = torch.stack(acts,1)
        input_t = self.clock
        reward,done,info = self.env.step(actions,auto_reset=False)
        self.clock += self.env.dt
        events = []
        for i in range(2):
            if info["fired"][0,i]:
                events.append(dict(type="shot",agent="AB"[i],start=info["ray_start"][0,i].tolist(),end=info["ray_end"][0,i].tolist(),
                                   hit=bool(info["hits"][0,i]),headshot=bool(info["headshots"][0,i]),damage=float(info["damage"][0,i])))
        frame = dict(type="frame",t=self.clock,input_t=input_t,round=int(self.env.rounds[0]),
                     round_time=float(self.env.steps[0])*self.env.dt,score=self.env.score[0].tolist(),
                     policy_version=self.version,intermission=False,events=events,agents=[])
        for i in range(2):
            frame["agents"].append(dict(id="AB"[i],position=self.env.pos[0,i].tolist(),yaw=float(self.env.yaw[0,i]),
                pitch=float(self.env.pitch[0,i]),hp=float(self.env.hp[0,i]),action=self.env.last_action[0,i].tolist(),
                activity=activity[i],signed_activity=signed_activity[i],observation={"width":self.env.width,"height":self.env.height,
                "rgb_base64":base64.b64encode(observations[0,i].tobytes()).decode("ascii")}))
        if done[0]:
            winner = "A" if info["outcome"][0,0]>0 else "B" if info["outcome"][0,1]>0 else "DRAW"
            frame["result"] = winner
            self.hold = max(1,round(1.5/self.env.dt))
        self.last_frame = frame
        return frame

    @torch.inference_mode()
    def advance(self, steps: int) -> dict | None:
        """Advance recurrent spectator state while returning only the latest render frame."""
        if steps < 1:
            raise ValueError("steps must be positive")
        frame = None
        for _ in range(steps):
            frame = self.tick()
        return frame
