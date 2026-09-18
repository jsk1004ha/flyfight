"""Pipe into the built container's Python stdin; uses only temporary state."""
import json
import math
import tempfile
from pathlib import Path

import torch

from flyfight.training import Config, Trainer


def main():
    cfg = Config(envs=8, hidden=768, horizon=16, epochs=2,
                 minibatch_envs=8, threads=2, action_steps=201, device="cpu")
    trainer = Trainer(cfg)
    stats = trainer.iteration()
    assert stats["env_steps"] == 128 and math.isfinite(stats["loss"])
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "latest.pt"
        trainer.save(checkpoint)
        restored = Trainer(cfg)
        restored.load(checkpoint)
        assert restored.env_steps == 128 and restored.update == 1
    print(json.dumps({"torch": torch.__version__, "update": trainer.update,
                      "env_steps": trainer.env_steps, "seconds": stats["seconds"],
                      "checkpoint_reload": True}))


if __name__ == "__main__":
    main()
