"""Displayed neurons preserve the native model's unit identity and activity."""
import numpy as np
import torch

from flyfight.spectator import Spectator
from flyfight.training import Config, Trainer


def test_large_native_model_displays_every_unit():
    trainer = Trainer(Config(envs=1, hidden=1536, width=8, height=6, threads=2))
    spectator = Spectator(trainer.cfg.map_path)
    spectator.offer(trainer.weight_blob())
    hello = spectator.hello()
    assert len(hello['neurons']) == 1536
    assert len({n['id'] for n in hello['neurons']}) == 1536
    assert np.array_equal(spectator.display_indices, np.arange(1536))
    xyz = np.array([[n['x'], n['y'], n['z']] for n in hello['neurons']])
    assert np.isfinite(xyz).all() and np.ptp(xyz[:, 2]) > 1
    assert all(0 <= i < 1536 and 0 <= j < 1536 for i, j in hello['edges'])
    frame = spectator.tick()
    for i, agent in enumerate(frame['agents']):
        expected = spectator.h[i][0].abs().clamp(0, 1)
        torch.testing.assert_close(torch.tensor(agent['activity']), expected)
