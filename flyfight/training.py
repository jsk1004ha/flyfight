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
import math
import os
from pathlib import Path
import queue
import time
import traceback
from contextlib import nullcontext
import torch
from .actions import ActionSpec, action_steps_from_config, migrate_actor_tensor
from .environment import Arena, DEFAULT_MAP
from .live_view import LiveTrainingView
from .model import FlyPolicy,action_head_entropies,action_stats,exploration_regularizer


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
    damage_reward: float = 0.0
    aim_reward: float = 0.0
    curriculum_fraction: float = 0.0
    curriculum_updates: int = 0
    exploration_mix: float = 0.0
    exploration_prior: float = 0.0
    target_kl: float = 0.0
    advantage_floor: float = 1e-8


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


def bounded_put_latest(q, value) -> None:
    """Replace a stale one-slot viewer sample without ever waiting on a client."""
    if q is None:
        return
    try:
        q.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
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
        if not (math.isfinite(cfg.exploration_mix) and 0 <= cfg.exploration_mix <= .5):
            raise ValueError("exploration_mix must be finite and in [0, 0.5]")
        if not (math.isfinite(cfg.exploration_prior) and cfg.exploration_prior >= 0):
            raise ValueError("exploration_prior must be finite and non-negative")
        if not (math.isfinite(cfg.target_kl) and cfg.target_kl >= 0):
            raise ValueError("target_kl must be finite and non-negative")
        if not (math.isfinite(cfg.advantage_floor) and cfg.advantage_floor > 0):
            raise ValueError("advantage_floor must be finite and positive")
        if not (math.isfinite(cfg.curriculum_fraction) and 0 <= cfg.curriculum_fraction <= 1):
            raise ValueError("curriculum_fraction must be finite and in [0, 1]")
        if isinstance(cfg.curriculum_updates, bool) or not isinstance(cfg.curriculum_updates, int) or cfg.curriculum_updates < 0:
            raise ValueError("curriculum_updates must be a non-negative integer")
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
                         headshot_bonus=cfg.headshot_bonus,damage_reward=cfg.damage_reward,
                         aim_reward=cfg.aim_reward,shaping_gamma=cfg.gamma,
                         curriculum_fraction=cfg.curriculum_fraction)
        first = FlyPolicy(cfg.width,cfg.height,cfg.hidden,cfg.action_steps).to(self.device)
        self.models = [first,copy.deepcopy(first)]
        self.optimizers = [torch.optim.Adam(m.parameters(),lr=cfg.learning_rate,eps=1e-5) for m in self.models]
        self.h = [torch.zeros(cfg.envs,cfg.hidden,device=self.device) for _ in range(2)]
        self.update = self.env_steps = self.games = self.draws = 0
        self.wins = [0,0]
        self.model_fns = list(self.models)
        self.map_hash = hashlib.sha256(Path(cfg.map_path).read_bytes()).hexdigest()
        self.live_view = LiveTrainingView(cfg)
        self.live_queue = None
        self.live_subscribers = None
        self.live_interval = 1 / 15
        self._last_live_sample = -float("inf")
        if cfg.resume:
            self.load(Path(cfg.resume))
        if cfg.compile_policy:
            # Compilation is optional and fails loudly rather than silently claiming acceleration.
            self.model_fns = [torch.compile(m,dynamic=False) for m in self.models]

    def autocast(self):
        return torch.autocast("cuda",dtype=torch.bfloat16) if self.cfg.amp else nullcontext()

    def _set_curriculum(self) -> None:
        if self.cfg.curriculum_updates:
            remaining = max(0.0, 1.0-self.update/self.cfg.curriculum_updates)
            self.env.set_curriculum_fraction(self.cfg.curriculum_fraction*remaining)
        else:
            self.env.set_curriculum_fraction(self.cfg.curriculum_fraction)

    @torch.no_grad()
    def collect(self) -> dict:
        cfg = self.cfg
        self._set_curriculum()
        starts = [h.clone() for h in self.h]
        observations,previous,actions,logps,values,rewards,dones,fired,hits,headshots = ([] for _ in range(10))
        outcomes = []
        for rollout_step in range(cfg.horizon):
            rgb = self.env.observe()
            prev = self.env.last_action
            aa,ll,vv,hh = [],[],[],[]
            with self.autocast():
                for i,m in enumerate(self.model_fns):
                    logits,value,hidden = m(rgb[:,i],prev[:,i],self.h[i])
                    action,logp,_ = action_stats(logits,heads=self.action_spec.heads,
                                                 exploration_mix=cfg.exploration_mix)
                    aa.append(action); ll.append(logp); vv.append(value); hh.append(hidden)
            act = torch.stack(aa,1)
            # Delay terminal reset until after an optional live sample so a
            # result event is never paired with the next round's spawn/HP.
            reward,done,info = self.env.step(act,auto_reset=False)
            observations.append(rgb); previous.append(prev); actions.append(act)
            logps.append(torch.stack(ll,1)); values.append(torch.stack(vv,1))
            rewards.append(reward); dones.append(done)
            outcomes.append(info["outcome"])
            fired.append(info["fired"]); hits.append(info["hits"]); headshots.append(info["headshots"])
            now = time.monotonic()
            live_active = self.live_subscribers is not None and self.live_subscribers.is_set()
            if live_active and now-self._last_live_sample >= self.live_interval:
                env_step = self.env_steps+(rollout_step+1)*cfg.envs
                frame = self.live_view.frame(
                    self.env,rgb,hh,info,update=self.update,env_step=env_step)
                bounded_put_latest(self.live_queue,frame)
                self._last_live_sample = now
            self.env.reset(done)
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
        normalized_entropies = []
        head_entropy_totals = torch.zeros(len(self.action_spec.heads),device=self.device)
        raw_entropies = []
        raw_normalized_entropies = []
        raw_head_entropy_totals = torch.zeros(len(self.action_spec.heads),device=self.device)
        entropy_batches = 0
        approx_kls = []
        clip_fractions = []
        kl_early_stops = 0
        advantage_stds = []
        advantage_scales = []
        max_entropy = sum(math.log(size) for size in self.action_spec.heads)
        for i,(model,optim) in enumerate(zip(self.model_fns,self.optimizers)):
            adv = data["advantages"][:,:,i]
            advantage_std = adv.std(unbiased=False)
            advantage_scale = advantage_std.clamp_min(cfg.advantage_floor)
            advantage_stds.append(advantage_std.detach())
            advantage_scales.append(advantage_scale.detach())
            adv = (adv-adv.mean())/advantage_scale
            stop_agent = False
            for _ in range(cfg.epochs):
                permutation = torch.randperm(cfg.envs,device=self.device)
                for start in range(0,cfg.envs,cfg.minibatch_envs):
                    idx = permutation[start:start+cfg.minibatch_envs]
                    h = data["h0"][i][idx].detach()
                    with self.autocast():
                        if cfg.compile_policy or not cfg.batch_sequence_encoder:
                            # Compiled modules expose only forward reliably across PyTorch versions.
                            ll,vv,ee,zz = [],[],[],[]
                            for t in range(cfg.horizon):
                                logits,value,h = model(data["rgb"][t,idx,i],data["prev"][t,idx,i],h)
                                _,lp,entropy = action_stats(
                                    logits,data["actions"][t,idx,i],heads=self.action_spec.heads,
                                    exploration_mix=cfg.exploration_mix)
                                ll.append(lp); vv.append(value); ee.append(entropy); zz.append(logits)
                                h = h*(~data["dones"][t,idx])[:,None]
                            logits,lp,value,entropy = (torch.stack(zz),torch.stack(ll),
                                                       torch.stack(vv),torch.stack(ee))
                        else:
                            logits,value,_ = model.forward_sequence(
                                data["rgb"][:,idx,i],data["prev"][:,idx,i],h,data["dones"][:,idx])
                            _,lp,entropy = action_stats(
                                logits,data["actions"][:,idx,i],heads=self.action_spec.heads,
                                exploration_mix=cfg.exploration_mix)
                        log_ratio = lp-data["logps"][:,idx,i]
                        ratio = log_ratio.exp()
                        approx_kl = ((ratio-1)-log_ratio).mean()
                        clip_fraction = ((ratio-1).abs() > cfg.clip).float().mean()
                        mean_entropy = entropy.mean().detach()
                        head_entropies = action_head_entropies(
                            logits.detach(),heads=self.action_spec.heads,
                            exploration_mix=cfg.exploration_mix).mean(tuple(range(logits.ndim-1)))
                        raw_head_entropies = action_head_entropies(
                            logits.detach(),heads=self.action_spec.heads).mean(tuple(range(logits.ndim-1)))
                        raw_mean_entropy = raw_head_entropies.sum()
                        approx_kls.append(approx_kl.detach())
                        clip_fractions.append(clip_fraction.detach())
                        entropies.append(mean_entropy)
                        normalized_entropies.append(mean_entropy/max_entropy)
                        head_entropy_totals += head_entropies
                        raw_entropies.append(raw_mean_entropy)
                        raw_normalized_entropies.append(raw_mean_entropy/max_entropy)
                        raw_head_entropy_totals += raw_head_entropies
                        entropy_batches += 1
                        if cfg.target_kl and approx_kl.detach().item() > cfg.target_kl*1.5:
                            kl_early_stops += 1
                            stop_agent = True
                            break
                        a = adv[:,idx]
                        pg = -torch.minimum(ratio*a,ratio.clamp(1-cfg.clip,1+cfg.clip)*a).mean()
                        value_loss = .5*(value-data["returns"][:,idx,i]).square().mean()
                        loss = pg+.5*value_loss-cfg.entropy*entropy.mean()
                        if cfg.exploration_prior:
                            loss = loss+cfg.exploration_prior*exploration_regularizer(
                                logits,heads=self.action_spec.heads).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite PPO loss. Stop and inspect the checkpoint/config.")
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.models[i].parameters(),.5,error_if_nonfinite=True)
                    optim.step()
                    losses.append(loss.detach()); norms.append(norm.detach())
                if stop_agent:
                    break
        head_entropy_means = (head_entropy_totals/max(entropy_batches,1)).cpu().tolist()
        raw_head_entropy_means = (raw_head_entropy_totals/max(entropy_batches,1)).cpu().tolist()
        return {"loss":torch.stack(losses).mean().item() if losses else 0.0,
                "grad_norm":torch.stack(norms).mean().item() if norms else 0.0,
                "entropy":torch.stack(entropies).mean().item() if entropies else 0.0,
                "entropy_normalized":torch.stack(normalized_entropies).mean().item() if normalized_entropies else 0.0,
                "entropy_heads":head_entropy_means,
                "raw_entropy":torch.stack(raw_entropies).mean().item() if raw_entropies else 0.0,
                "raw_entropy_normalized":torch.stack(raw_normalized_entropies).mean().item() if raw_normalized_entropies else 0.0,
                "raw_entropy_heads":raw_head_entropy_means,
                "exploration_mix":cfg.exploration_mix,
                "advantage_std":torch.stack(advantage_stds).mean().item(),
                "advantage_normalization_scale":torch.stack(advantage_scales).mean().item(),
                "advantage_floor":cfg.advantage_floor,
                "approx_kl":torch.stack(approx_kls).mean().item() if approx_kls else 0.0,
                "clip_fraction":torch.stack(clip_fractions).mean().item() if clip_fractions else 0.0,
                "kl_early_stops":kl_early_stops}

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
                     curriculum_fraction=self.env.curriculum_fraction,
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
        for key, default in (("turn_speed", 2.0), ("pitch_speed", 1.0), ("headshot_bonus", 0.0),
                             ("damage_reward", 0.0), ("aim_reward", 0.0),
                             ("curriculum_fraction", 0.0), ("curriculum_updates", 0),
                             ("exploration_mix", 0.0), ("exploration_prior", 0.0),
                             ("target_kl", 0.0), ("advantage_floor", 1e-8)):
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


def train_worker(cfg_dict: dict, status_queue=None, live_queue=None,
                 stop_event=None,pause_event=None,save_event=None,live_subscribers=None) -> None:
    """Spawn-safe entrypoint. No HTTP/rendering work occurs inside this process."""
    cfg = Config(**cfg_dict)
    run = Path(cfg.run_dir); run.mkdir(parents=True,exist_ok=True)
    trainer = None
    # Queued telemetry must not keep a terminated worker alive.
    for q in (status_queue,live_queue):
        if q is not None and hasattr(q,"cancel_join_thread"):
            q.cancel_join_thread()
    try:
        trainer = Trainer(cfg)
        trainer.live_queue = live_queue
        trainer.live_subscribers = live_subscribers
        (run/"config.json").write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2),encoding="utf-8")
        if not cfg.resume:
            trainer.save(run/"initial.pt")
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
            bounded_put(status_queue,{"state":"stopped","update":trainer.update,"env_steps":trainer.env_steps,
                                      "device":str(trainer.device),"envs":cfg.envs})
