"""Explicit, non-destructive transition to a new combat-learning experiment."""
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from .training import Config, Trainer


PROFILE = "combat-v1"


def combat_config(base: Config) -> Config:
    """Keep observation/action geometry; improve signal and rollout stability."""
    return replace(base, horizon=64, learning_rate=1e-4, entropy=.03,
                   exploration_mix=.10, target_kl=.02, advantage_floor=.1,
                   exploration_prior=.002, damage_reward=.5, aim_reward=.05,
                   curriculum_fraction=.8, curriculum_updates=2000,
                   resume="", updates=0)


def prepare_combat_run(directory: Path, base: Config) -> Config:
    """Create once in a separate directory, never rewrite the source run.

    Encoder/recurrent weights are retained. Actor logits are softened by 20x;
    critic, Adam, recurrent trajectory and counters restart for the new objective.
    This is transfer learning, NOT an exact continuation of the old experiment.
    """
    directory = Path(directory).resolve()
    target = directory / PROFILE
    if target.exists():
        saved = target / "config.json"
        checkpoint = target / "latest.pt"
        if not checkpoint.is_file():
            checkpoint = target / "initial.pt"
        if not saved.is_file() or not checkpoint.is_file():
            raise RuntimeError("Combat run is incomplete; refusing to overwrite it")
        values = json.loads(saved.read_text(encoding="utf-8"))
        known = {field.name for field in fields(Config)}
        cfg = Config(**{k: v for k, v in values.items() if k in known})
        return replace(cfg, run_dir=str(target), resume=str(checkpoint),
                       map_path=base.map_path, device=base.device, updates=0)

    cfg = combat_config(base)
    cfg.run_dir = str(target)
    trainer = Trainer(cfg)
    provenance = {"profile": PROFILE, "source": None, "mode": "fresh"}
    if base.resume:
        source = Path(base.resume).resolve()
        with source.open("rb") as stream:
            data = torch.load(stream, map_location=trainer.device, weights_only=True)
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if data.get("schema") != 3 or data.get("map_hash") != trainer.map_hash:
            raise ValueError("Transfer requires matching schema-3 map")
        saved_cfg = data["config"]
        for key in ("hidden", "width", "height", "action_steps", "turn_speed", "pitch_speed"):
            defaults = {"action_steps": 9, "turn_speed": 2., "pitch_speed": 1.}
            if saved_cfg.get(key, defaults.get(key)) != getattr(cfg, key):
                raise ValueError(f"Transfer requires matching {key}")
        if len(data.get("models", [])) != 2:
            raise ValueError("Transfer requires two complete model states")
        with torch.no_grad():
            for model, state in zip(trainer.models, data["models"]):
                model.load_state_dict(state, strict=True)
                model.actor.weight.mul_(.05)
                model.actor.bias.mul_(.05)
                model.critic.weight.zero_()
                model.critic.bias.zero_()
        provenance.update(source=str(source), source_sha256=digest,
                          source_update=data.get("update"), source_env_steps=data.get("env_steps"),
                          mode="transfer", actor_logit_scale=.05,
                          reset=["critic", "optimizer", "rollout", "counters"])

    # Publish a complete new experiment atomically; preserve staging on failure.
    staging = Path(tempfile.mkdtemp(prefix="combat-v1-staging-", dir=directory))
    trainer.save(staging / "initial.pt")
    (staging / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (staging / "transfer.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    staging.rename(target)
    return replace(cfg, resume=str(target / "initial.pt"))
