"""Measure local end-to-end training throughput. Not a game-skill benchmark."""
from dataclasses import asdict
import argparse,json,statistics,time
from pathlib import Path
import torch
from flyfight.training import Config,Trainer

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--device',default='auto');p.add_argument('--envs',type=int,default=128)
 p.add_argument('--hidden',type=int,default=768);p.add_argument('--width',type=int,default=24);p.add_argument('--height',type=int,default=16)
 p.add_argument('--horizon',type=int,default=32);p.add_argument('--updates',type=int,default=5);p.add_argument('--threads',type=int,default=4)
 p.add_argument('--minibatch-envs',type=int,default=64);p.add_argument('--epochs',type=int,default=2)
 p.add_argument('--no-batch-sequence-encoder',action='store_true',help='Benchmark the previous per-step PPO path')
 p.add_argument('--amp',action='store_true');p.add_argument('--compile-policy',action='store_true');p.add_argument('--output',default='benchmark.json')
 a=p.parse_args()
 if a.updates<1:raise SystemExit('updates must be positive')
 cfg=Config(device=a.device,envs=a.envs,hidden=a.hidden,width=a.width,height=a.height,horizon=a.horizon,threads=a.threads,amp=a.amp,compile_policy=a.compile_policy,minibatch_envs=a.minibatch_envs,epochs=a.epochs,batch_sequence_encoder=not a.no_batch_sequence_encoder)
 t=Trainer(cfg);t.iteration() # warm-up incl. compile
 rows=[t.iteration() for _ in range(a.updates)]
 result=dict(torch=torch.__version__,cuda_available=torch.cuda.is_available(),device=str(t.device),
   gpu=torch.cuda.get_device_name(t.device) if t.device.type=='cuda' else None,config=asdict(cfg),warmup_updates=1,
   median_env_steps_s=statistics.median(r['env_steps_s'] for r in rows),rows=rows,
   note='End-to-end RGB+physics+two-agent recurrent PPO. Browser excluded; no HTML-baseline speedup measured.')
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
