"""Persistent single-instance hosted trainer with public read-only viewing."""
from dataclasses import fields
import json
import multiprocessing as mp
import os
from pathlib import Path

from aiohttp import web

from flyfight.training import Config
from flyfight.server import create_app


def cloud_config(directory: Path) -> Config:
    latest = directory / 'latest.pt'
    initial = directory / 'initial.pt'
    checkpoint = latest if latest.is_file() else initial
    saved = directory / 'config.json'
    if saved.exists():
        if not checkpoint.is_file():
            raise RuntimeError('Saved experiment has no checkpoint; refusing to reset learned progress')
        values = json.loads(saved.read_text(encoding='utf-8'))
        known = {f.name for f in fields(Config)}
        values = {k: v for k, v in values.items() if k in known}
        values.setdefault('headshot_bonus', 0.0)
        cfg = Config(**values)
        cfg.resume = str(checkpoint)
    else:
        if checkpoint.is_file() or (directory / 'metrics.jsonl').exists():
            raise RuntimeError('Existing experiment is missing config.json')
        cfg = Config(envs=8, hidden=768, horizon=16, epochs=2, minibatch_envs=8,
                     threads=2, action_steps=201, save_every=10, publish_every=5)
    cfg.run_dir = str(directory)
    cfg.device = os.environ.get('FLYFIGHT_DEVICE', 'cpu')
    cfg.map_path = str(Path(__file__).parent / 'maps' / 'arena_large.json')
    cfg.updates = 0
    return cfg


def main():
    origin = os.environ.get('FLYFIGHT_PUBLIC_ORIGIN')
    if not origin:
        raise RuntimeError('Set FLYFIGHT_PUBLIC_ORIGIN to the hosted HTTPS origin')
    directory = Path(os.environ.get('FLYFIGHT_RUN_DIR', '/data/flyfight'))
    if not directory.is_dir():
        raise RuntimeError('Mount a persistent, writable run directory before starting cloud training')
    port = int(os.environ.get('PORT', '8765'))
    cfg = cloud_config(directory)
    web.run_app(create_app(cfg,port=port,public_origin=origin),host='0.0.0.0',port=port)


if __name__ == '__main__':
    mp.freeze_support()
    main()
