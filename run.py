"""Usage: python run.py --device cuda --envs 128 --open-browser"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import multiprocessing as mp
from flyfight.training import Config,convert_schema2_checkpoint,train_worker


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FlyFight Native: synthetic RNN PPO + HTML spectator")
    for name in ("envs","hidden","width","height","horizon","epochs","minibatch_envs","seed","threads","updates","save_every","publish_every","action_steps","curriculum_updates"):
        p.add_argument("--"+name.replace("_","-"),type=int,default=getattr(Config(),name))
    for name in ("learning_rate","gamma","gae_lambda","entropy","clip","episode_seconds","dt","turn_speed","pitch_speed","headshot_bonus","damage_reward","aim_reward","curriculum_fraction","exploration_mix","target_kl","advantage_floor","exploration_prior"):
        p.add_argument("--"+name.replace("_","-"),type=float,default=getattr(Config(),name))
    p.add_argument("--action-precision",type=int,choices=(2,3),default=None,
                   help="Yaw/pitch action grid: 2 = 0.01, 3 = 0.001; overrides --action-steps")
    p.add_argument("--device",default="auto",help="auto / cpu / cuda / cuda:0")
    p.add_argument("--run-dir",default=None)
    p.add_argument("--map-path",default=str(Config().map_path))
    p.add_argument("--resume",default="")
    p.add_argument("--convert-schema2",metavar="CHECKPOINT",default="",
                   help="Convert a legacy schema-2 checkpoint and exit")
    p.add_argument("--converted-output",default="",help="Destination for --convert-schema2")
    p.add_argument("--amp",action="store_true",help="Optional CUDA bfloat16 autocast; benchmark before use")
    p.add_argument("--compile-policy",action="store_true",help="Optional torch.compile; compiler support required")
    p.add_argument("--headless",action="store_true",help="Train without HTTP server or spectator")
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--open-browser",action="store_true")
    return p


def main() -> None:
    args = vars(parser().parse_args())
    precision = args.pop("action_precision")
    if precision is not None:
        args["action_steps"] = 2 * 10 ** precision + 1
    headless,port,open_browser = [args.pop(k) for k in ("headless","port","open_browser")]
    convert_source = args.pop("convert_schema2")
    converted_output = args.pop("converted_output")
    if convert_source:
        source = Path(convert_source)
        if not source.is_file():
            raise SystemExit("Schema-2 checkpoint does not exist")
        destination = Path(converted_output) if converted_output else source.with_name(source.stem+"_schema3.pt")
        if destination.exists():
            raise SystemExit("Converted output already exists; choose a new --converted-output")
        metadata = convert_schema2_checkpoint(source,destination,action_steps=args["action_steps"])
        print(f"Converted checkpoint: {destination.resolve()}",flush=True)
        print(f"Actor mapping: {metadata['actor_mapping']}; optimizer/rollout state: reset",flush=True)
        return
    if not 1024 <= port <= 65535:
        raise SystemExit("Choose a port between 1024 and 65535")
    if args["run_dir"] is None:
        from datetime import datetime
        args["run_dir"] = str(Path(args["resume"]).resolve().parent) if args["resume"] else str(Path("runs")/datetime.now().strftime("session_%Y%m%d_%H%M%S"))
    cfg = Config(**args)
    print(f"Run directory: {Path(cfg.run_dir).resolve()}",flush=True)
    # Never silently overwrite an old experiment. Explicit --resume is required.
    run = Path(cfg.run_dir)
    if not cfg.resume and any((run/n).exists() for n in ("initial.pt","latest.pt","metrics.jsonl")):
        raise SystemExit("Run directory already contains an experiment. Use a NEW --run-dir or --resume runs/.../latest.pt with the original dimensions.")
    if cfg.resume and not Path(cfg.resume).is_file():
        raise SystemExit("Checkpoint does not exist")
    if headless:
        train_worker(asdict(cfg))
    else:
        from aiohttp import web
        from flyfight.server import create_app
        web.run_app(create_app(cfg,open_browser=open_browser,port=port),host="127.0.0.1",port=port,print=None)


if __name__ == "__main__":
    mp.freeze_support()
    main()
