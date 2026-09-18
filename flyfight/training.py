"""Independent recurrent PPO learners, vectorized self-play, bounded telemetry.

The two agents begin with identical weights, then own separate parameters,
optimizers, hidden states and experience. The critic has the SAME local inputs
as the actor. Entropy is an explicit optimizer regularizer, not a shaped reward.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, asdict
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import time
import traceback
from contextlib import nullcontext
import torch
from .actions import ActionSpec, action_steps_from_config, migrate_actor_tensor
from .environment import Arena, DEFAULT_MAP
from .model import FlyPolicy,action_stats


@dataclass
class Config:
    envs: int = 128
    hidden: int = 768
    width: int = 24
    height: int = 16
    horizon: int = 32
    epochs: int = 2
    minibatch_envs: int = 64
    learning_rate: float = 3e-4
    gamma: float = .999
    gae_lambda: float = .97
    entropy: float = .01
    clip: float = .2
    episode_seconds: float = 60
    dt: float = .1
    seed: int = 17
    device: str = "auto"
    amp: bool = False
    compile_policy: bool = False
    batch_sequence_encoder: bool = True
    threads: int = 4
    updates: int = 0  # 0 means run until Ctrl+C/stop event
    save_every: int = 50
    publish_every: int = 10
    run_dir: str = "runs/default"
    map_path: str = str(DEFAULT_MAP)
    resume: str = ""
    action_steps: int = 9
    turn_speed: float = 2.0
    pitch_speed: float = 1.0
    headshot_bonus: float = .25


def choose_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(request)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a CUDA-enabled PyTorch build, or select --device cpu.")
    if dev.type not in ("cpu","cuda"):
        raise ValueError("This release supports cpu / cuda only")
    return dev


def bounded_put(q, value) -> None:
    """Never let a slow spectator block training."""
    if q is None:
        return
    try:
        q.put_nowait(value)
    except queue.Full:
        pass


def atomic_save(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp = destination.with_suffix(destination.suffix+".tmp")
    torch.save(data,temp)
    os.replace(temp,destination)


def generalized_advantage(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor,
                          bootstrap: torch.Tensor, gamma: float, lam: float):
    adv = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    next_v = bootstrap
    for t in reversed(range(len(rewards))):
        mask = (~dones[t]).float()
        delta = rewards[t]+gamma*mask*next_v-values[t]
        carry = delta+gamma*lam*mask*carry
        adv[t] = carry
        next_v = values[t]
    return adv,adv+values


class Trainer:
    def __init__(self,cfg: Config):
        if min(cfg.envs,cfg.horizon,cfg.epochs,cfg.minibatch_envs,cfg.save_every,cfg.publish_every,cfg.threads) <= 0:
            raise ValueError("Batch, horizon, epochs, intervals and threads must be positive")
        if not (0 < cfg.gamma <= 1 and 0 <= cfg.gae_lambda <= 1 and 0 < cfg.clip < 1 and cfg.entropy >= 0):
            raise ValueError("Invalid PPO parameters")
        self.action_spec = ActionSpec(cfg.action_steps)
        self.cfg,self.device = cfg,choose_device(cfg.device)
        torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)
            torch.set_float32_matmul_precision("high")
        if cfg.amp and self.device.type != "cuda":
            raise ValueError("--amp requires CUDA in this release")
        if cfg.amp and not torch.cuda.is_bf16_supported():
            raise ValueError("--amp requires CUDA bfloat16 support")
        self.env = Arena(cfg.envs,self.device,width=cfg.width,height=cfg.height,dt=cfg.dt,
                         episode_seconds=cfg.episode_seconds,seed=cfg.seed,map_path=cfg.map_path,
                         action_steps=cfg.action_steps,turn_speed=cfg.turn_speed,pitch_speed=cfg.pitch_speed,
                         headshot_bonus=cfg.headshot_bonus)
        first = FlyPolicy(cfg.width,cfg.height,cfg.hidden,cfg.action_steps).to(self.device)
        self.models = [first,copy.deepcopy(first)]
        self.optimizers = [torch.optim.Adam(m.parameters(),lr=cfg.learning_rate,eps=1e-5) for m in self.models]
        self.h = [torch.zeros(cfg.envs,cfg.hidden,device=self.device) for _ in range(2)]
        self.update = self.env_steps = self.games = self.draws = 0
        self.wins = [0,0]
        self.model_fns = list(self.models)
        self.map_hash = hashlib.sha256(Path(cfg.map_path).read_bytes()).hexdigest()
        if cfg.resume:
            self.load(Path(cfg.resume))
        if cfg.compile_policy:
            # Compilation is optional and fails loudly rather than silently claiming acceleration.
            self.model_fns = [torch.compile(m,dynamic=False) for m in self.models]

    def autocast(self):
        return torch.autocast("cuda",dtype=torch.bfloat16) if self.cfg.amp else nullcontext()

    @torch.no_grad()
    def collect(self) -> dict:
        cfg = self.cfg
        starts = [h.clone() for h in self.h]
        observations,previous,actions,logps,values,rewards,dones,fired,hits,headshots = ([] for _ in range(10))
        outcomes = []
        for _ in range(cfg.horizon):
            rgb = self.env.observe()
            prev = self.env.last_action
            aa,ll,vv,hh = [],[],[],[]
            with self.autocast():
                for i,m in enumerate(self.model_fns):
                    logits,value,hidden = m(rgb[:,i],prev[:,i],self.h[i])
                    action,logp,_ = action_stats(logits,heads=self.action_spec.heads)
                    aa.append(action); ll.append(logp); vv.append(value); hh.append(hidden)
            act = torch.stack(aa,1)
            reward,done,info = self.env.step(act)
            observations.append(rgb); previous.append(prev); actions.append(act)
            logps.append(torch.stack(ll,1)); values.append(torch.stack(vv,1))
            rewards.append(reward); dones.append(done)
            outcomes.append(info["outcome"])
            fired.append(info["fired"]); hits.append(info["hits"]); headshots.append(info["headshots"])
            self.h = [v*(~done)[:,None] for v in hh]
        rgb = self.env.observe()
        with self.autocast():
            bootstrap = torch.stack([m(rgb[:,i],self.env.last_action[:,i],self.h[i])[1] for i,m in enumerate(self.model_fns)],1)
        data = dict(rgb=torch.stack(observations),prev=torch.stack(previous),actions=torch.stack(actions),
                    logps=torch.stack(logps),values=torch.stack(values),rewards=torch.stack(rewards),
                    dones=torch.stack(dones),fired=torch.stack(fired),hits=torch.stack(hits),
                    headshots=torch.stack(headshots),outcomes=torch.stack(outcomes),h0=starts)
        adv,returns = generalized_advantage(data["rewards"],data["dones"][...,None].expand(-1,-1,2),
                                           data["values"],bootstrap,cfg.gamma,cfg.gae_lambda)
        data["advantages"],data["returns"] = adv,returns
        return data

    def optimize(self,data: dict) -> dict:
        cfg = self.cfg
        losses = []
        norms = []
        entropies = []
        for i,(model,optim) in enumerate(zip(self.model_fns,self.optimizers)):
            adv = data["advantages"][:,:,i]
            adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            for _ in range(cfg.epochs):
                permutation = torch.randperm(cfg.envs,device=self.device)
                for start in range(0,cfg.envs,cfg.minibatch_envs):
                    idx = permutation[start:start+cfg.minibatch_envs]
                    h = data["h0"][i][idx].detach()
                    with self.autocast():
                        if cfg.compile_policy or not cfg.batch_sequence_encoder:
                            # Compiled modules expose only forward reliably across PyTorch versions.
                            ll,vv,ee = [],[],[]
                            for t in range(cfg.horizon):
                                logits,value,h = model(data["rgb"][t,idx,i],data["prev"][t,idx,i],h)
                                _,lp,entropy = action_stats(logits,data["actions"][t,idx,i],heads=self.action_spec.heads)
                                ll.append(lp); vv.append(value); ee.append(entropy)
                                h = h*(~data["dones"][t,idx])[:,None]
                            lp,value,entropy = torch.stack(ll),torch.stack(vv),torch.stack(ee)
                        else:
                            logits,value,_ = model.forward_sequence(
                                data["rgb"][:,idx,i],data["prev"][:,idx,i],h,data["dones"][:,idx])
                            _,lp,entropy = action_stats(logits,data["actions"][:,idx,i],heads=self.action_spec.heads)
                        ratio = (lp-data["logps"][:,idx,i]).exp()
                        a = adv[:,idx]
                        pg = -torch.minimum(ratio*a,ratio.clamp(1-cfg.clip,1+cfg.clip)*a).mean()
                        value_loss = .5*(value-data["returns"][:,idx,i]).square().mean()
                        loss = pg+.5*value_loss-cfg.entropy*entropy.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite PPO loss. Stop and inspect the checkpoint/config.")
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.models[i].parameters(),.5,error_if_nonfinite=True)
                    optim.step()
                    losses.append(loss.detach()); norms.append(norm.detach()); entropies.append(entropy.mean().detach())
        return {"loss":torch.stack(losses).mean().item(),"grad_norm":torch.stack(norms).mean().item(),
                "entropy":torch.stack(entropies).mean().item()}

    def iteration(self) -> dict:
        start = self._timestamp()
        data = self.collect()
        collected = self._timestamp()
        stats = self.optimize(data)
        optimized = self._timestamp()
        self.update += 1
        steps = self.cfg.envs*self.cfg.horizon
        self.env_steps += steps
        outcomes = torch.stack([(data["outcomes"][...,0]>0).sum(),(data["outcomes"][...,1]>0).sum(),data["dones"].sum()]).cpu().tolist()
        combat = torch.stack([data["fired"].sum(),data["hits"].sum(),data["headshots"].sum()]).cpu().tolist()
        a,b,games = map(int,outcomes)
        shots,hits,headshots = map(int,combat)
        self.wins[0] += a; self.wins[1] += b; self.games += games; self.draws += games-a-b
        collection_seconds = collected-start
        optimization_seconds = optimized-collected
        elapsed = optimized-start
        stats.update(update=self.update,env_steps=self.env_steps,env_steps_s=steps/max(elapsed,1e-9),
                     agent_decisions_s=steps*2/max(elapsed,1e-9),games=self.games,wins=self.wins.copy(),
                     draws=self.draws,draw_rate=self.draws/max(self.games,1),device=str(self.device),
                     envs=self.cfg.envs,seconds=elapsed,collection_seconds=collection_seconds,
                     optimization_seconds=optimization_seconds,
                     collection_env_steps_s=steps/max(collection_seconds,1e-9),
                     update_shots=shots,update_hits=hits,update_headshots=headshots,
                     update_hit_rate=hits/max(shots,1),
                     source="synthetic RNN / recurrent PPO",
                     gpu_memory_mb=torch.cuda.max_memory_allocated(self.device)/2**20 if self.device.type=="cuda" else 0.)
        return stats

    def _timestamp(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def weight_blob(self) -> bytes:
        out = io.BytesIO()
        torch.save({"schema":3,"config":asdict(self.cfg),"update":self.update,"map_hash":self.map_hash,
                    "models":[{k:v.detach().cpu() for k,v in m.state_dict().items()} for m in self.models]},out)
        return out.getvalue()

    def save(self,path: Path) -> None:
        state = {"schema":3,"config":asdict(self.cfg),"map_hash":self.map_hash,"update":self.update,
                 "env_steps":self.env_steps,"games":self.games,"draws":self.draws,"wins":self.wins,
                 "models":[m.state_dict() for m in self.models],"optimizers":[o.state_dict() for o in self.optimizers],
                 "hidden":self.h,"env":{k:getattr(self.env,k) for k in
                     ("pos","yaw","pitch","hp","cooldown","last_action","steps","rounds","score")},
                 "env_rng":self.env.generator.get_state(),"cpu_rng":torch.get_rng_state(),
                 "cuda_rng":torch.cuda.get_rng_state_all() if self.device.type=="cuda" else []}
        atomic_save(state,path)

    def load(self,path: Path) -> None:
        data = torch.load(path,map_location=self.device,weights_only=True)
        if data.get("schema") != 3 or data.get("map_hash") != self.map_hash:
            raise ValueError("Checkpoint schema/map does not match; convert schema 2 explicitly before resuming")
        saved_steps = action_steps_from_config(data.get("config",{}))
        if saved_steps != self.cfg.action_steps:
            raise ValueError(f"Resume requires same action_steps; saved={saved_steps}")
        for key, default in (("turn_speed", 2.0), ("pitch_speed", 1.0), ("headshot_bonus", 0.0)):
            if data["config"].get(key, default) != getattr(self.cfg, key):
                raise ValueError(f"Resume requires same {key}")
        for key in ("envs","hidden","width","height","dt","episode_seconds","seed"):
            if data["config"][key] != getattr(self.cfg,key):
                raise ValueError(f"Resume requires same {key}; saved={data['config'][key]}")
        for m,s in zip(self.models,data["models"]):
            m.load_state_dict(s)
        optimizer_states = data.get("optimizers")
        if optimizer_states is not None:
            for o,s in zip(self.optimizers,optimizer_states):
                o.load_state_dict(s)
        conversion = data.get("conversion",{})
        if conversion.get("resume_mode") == "weights_only":
            # Action semantics changed. Begin a clean rollout/optimizer trajectory
            # while retaining the learned encoder, recurrent core, actor and critic.
            return
        self.h = [h.to(self.device) for h in data["hidden"]]
        for k,v in data["env"].items():
            setattr(self.env,k,v.to(self.device))
        self.env.generator.set_state(data["env_rng"].cpu())
        torch.set_rng_state(data["cpu_rng"].cpu())
        if self.device.type == "cuda" and data["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in data["cuda_rng"]])
        self.update,self.env_steps,self.games,self.draws = [data[k] for k in ("update","env_steps","games","draws")]
        self.wins = data["wins"]


def convert_schema2_checkpoint(source: Path, destination: Path, *, action_steps: int = 9) -> dict:
    """Convert a full schema-2 checkpoint, preserving state while resetting Adam."""
    target = ActionSpec(action_steps)
    data = torch.load(source,map_location="cpu",weights_only=True)
    if data.get("schema") != 2:
        raise ValueError(f"Expected schema 2 checkpoint, got schema {data.get('schema')!r}")
    if not isinstance(data.get("models"),list) or len(data["models"]) != 2:
        raise ValueError("Schema 2 checkpoint must contain exactly two model states")
    converted_models = []
    for state in data["models"]:
        required = {"encoder.weight","encoder.bias","recurrent.weight","actor.weight","actor.bias",
                    "critic.weight","critic.bias"}
        if not required.issubset(state):
            raise ValueError("Schema 2 model state is incomplete")
        migrated = {key:value.clone() for key,value in state.items()}
        migrated["actor.weight"] = migrate_actor_tensor(state["actor.weight"],target)
        migrated["actor.bias"] = migrate_actor_tensor(state["actor.bias"],target)
        converted_models.append(migrated)
    config = dict(data.get("config",{}))
    config["action_steps"] = action_steps
    source_progress = {key:data.get(key,0) for key in ("update","env_steps","games","draws","wins")}
    result = {"schema":3,"config":config,"map_hash":data.get("map_hash"),"models":converted_models,
              "optimizers":None,
              "conversion":{"source_schema":2,"target_schema":3,
                            "actor_mapping":"linear interpolation over decoded yaw/pitch values",
                            "legacy_heads":[5,5,3,2],"target_heads":list(target.heads),
                            "optimizer":"reset","resume_mode":"weights_only",
                            "rollout_rng_and_counters":"reset","source_progress":source_progress,
                            "source":str(source.resolve())}}
    atomic_save(result,destination)
    return result["conversion"]


def train_worker(cfg_dict: dict, status_queue=None, weight_queue=None,
                 stop_event=None,pause_event=None,save_event=None) -> None:
    """Spawn-safe entrypoint. No HTTP/rendering work occurs inside this process."""
    cfg = Config(**cfg_dict)
    run = Path(cfg.run_dir); run.mkdir(parents=True,exist_ok=True)
    trainer = None
    # Queued telemetry must not keep a terminated worker alive.
    for q in (status_queue,weight_queue):
        if q is not None and hasattr(q,"cancel_join_thread"):
            q.cancel_join_thread()
    try:
        trainer = Trainer(cfg)
        (run/"config.json").write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2),encoding="utf-8")
        if not cfg.resume:
            trainer.save(run/"initial.pt")
        bounded_put(weight_queue,trainer.weight_blob())
        bounded_put(status_queue,{"state":"training","device":str(trainer.device),"envs":cfg.envs,"update":trainer.update})
        with (run/"metrics.jsonl").open("a",encoding="utf-8") as log:
            while not(stop_event is not None and stop_event.is_set()):
                if cfg.updates and trainer.update >= cfg.updates:
                    break
                if pause_event is not None and pause_event.is_set():
                    if save_event is not None and save_event.is_set():
                        trainer.save(run/"latest.pt")
                        save_event.clear()
                    time.sleep(.1)
                    continue
                stats = trainer.iteration()
                stats["state"] = "training"
                log.write(json.dumps(stats,ensure_ascii=False,allow_nan=False)+"\n"); log.flush()
                bounded_put(status_queue,stats)
                if trainer.update%cfg.publish_every==0:
                    bounded_put(weight_queue,trainer.weight_blob())
                if trainer.update%cfg.save_every==0 or (save_event is not None and save_event.is_set()):
                    trainer.save(run/"latest.pt")
                    if save_event is not None:
                        save_event.clear()
                print(f"update={trainer.update} | env_steps/s={stats['env_steps_s']:.1f} | games={trainer.games} | draws={trainer.draws}",flush=True)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        bounded_put(status_queue,{"state":"error","error":str(exc)})
        (run/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        traceback.print_exc()
        raise
    finally:
        if trainer is not None:
            trainer.save(run/"latest.pt")
            bounded_put(weight_queue,trainer.weight_blob())
            bounded_put(status_queue,{"state":"stopped","update":trainer.update,"env_steps":trainer.env_steps,
                                      "device":str(trainer.device),"envs":cfg.envs})
