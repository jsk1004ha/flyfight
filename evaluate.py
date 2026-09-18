"""Evaluate one learned identity against a frozen baseline with balanced roles.
No learning. A win is counted against the same fixed opponent, not against an
opponent that is also changing. This script produces results, not a promise of
skill improvement. For reliable claims repeat with multiple independent seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from flyfight.environment import Arena,DEFAULT_MAP
from flyfight.model import FlyPolicy,action_stats
from flyfight.training import choose_device


def wilson(wins:int,total:int)->list[float]:
    if not total:return [0.,1.]
    z=1.96;p=wins/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    margin=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0,center-margin),min(1,center+margin)]


@torch.inference_mode()
def evaluate(checkpoint:str,opponent:str,*,games:int=128,envs:int=32,device:str="auto",seed:int=424242,identity:str="A",map_path:str=str(DEFAULT_MAP)) -> dict:
    if min(games,envs)<1:raise ValueError("games and envs must be positive")
    dev=choose_device(device);torch.set_num_threads(2);torch.manual_seed(seed)
    a=torch.load(checkpoint,map_location=dev,weights_only=True)
    b=torch.load(opponent,map_location=dev,weights_only=True)
    if a['map_hash']!=b['map_hash']:raise ValueError('Different maps in checkpoints')
    if a['map_hash']!=hashlib.sha256(Path(map_path).read_bytes()).hexdigest():raise ValueError('Supplied map does not match checkpoint')
    if identity not in ('A','B'):raise ValueError('identity must be A or B')
    cfg=a['config']
    action_steps=int(cfg.get('action_steps',9))
    speeds = {k: cfg.get(k, default) for k, default in (("turn_speed",2.0),("pitch_speed",1.0))}
    for key, value in speeds.items():
        if b['config'].get(key, 2.0 if key == 'turn_speed' else 1.0) != value:
            raise ValueError(f'Different {key}')
    if int(b['config'].get('action_steps',9)) != action_steps:raise ValueError('Different action_steps')
    for k in ('width','height','hidden','dt','episode_seconds'):
        if cfg[k]!=b['config'][k]:raise ValueError(f'Different {k}')
    models=[FlyPolicy(cfg['width'],cfg['height'],cfg['hidden'],action_steps).to(dev).eval() for _ in range(2)]
    candidate=0 if identity=='A' else 1
    models[0].load_state_dict(a['models'][candidate]);models[1].load_state_dict(b['models'][1-candidate])
    n=min(envs,games);e=Arena(n,dev,width=cfg['width'],height=cfg['height'],dt=cfg['dt'],episode_seconds=cfg['episode_seconds'],seed=seed,map_path=map_path,action_steps=action_steps,**speeds)
    h=[torch.zeros(n,cfg['hidden'],device=dev) for _ in range(2)]
    batch=torch.arange(n,device=dev);counts=[0]*n;quotas=[games//n+(i<games%n) for i in range(n)]
    wins=losses=draws=0;roles=torch.arange(n,device=dev)%2
    while any(c<q for c,q in zip(counts,quotas)):
        rgb=e.observe();acts=torch.zeros(n,2,4,dtype=torch.long,device=dev)
        for i,m in enumerate(models):
            slot=roles if i==0 else 1-roles
            logits,_,h[i]=m(rgb[batch,slot],e.last_action[batch,slot],h[i])
            actions,_,_=action_stats(logits,heads=m.heads)
            acts[batch,slot]=actions
        reward,done,info=e.step(acts)
        outcome=info['outcome'][batch,roles]
        for index,result in zip(torch.where(done)[0].cpu().tolist(),outcome[done].cpu().tolist()):
            if counts[index]<quotas[index]:
                wins+=int(result>0);losses+=int(result<0);draws+=int(result==0);counts[index]+=1
        # Alternate roles per round. Mask recurrent state on all resets.
        roles=torch.where(done,1-roles,roles)
        h=[v*(~done)[:,None] for v in h]
    return dict(checkpoint=checkpoint,opponent=opponent,candidate_identity=identity,seed=seed,games=games,
                wins=wins,losses=losses,draws=draws,win_rate_all=wins/games,win_rate_wilson95=wilson(wins,games),
                decisive_win_rate=wins/(wins+losses) if wins+losses else None,
                note="Frozen-opponent evaluation with alternating roles; no learning. Repeat across seeds.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--opponent',required=True)
    p.add_argument('--games',type=int,default=128);p.add_argument('--envs',type=int,default=32)
    p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=424242)
    p.add_argument('--identity',choices=['A','B'],default='A');p.add_argument('--map-path',default=str(DEFAULT_MAP))
    p.add_argument('--output',default='evaluation.json')
    args=vars(p.parse_args());out=Path(args.pop('output'));result=evaluate(**args)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
