"""Shared discrete action layout and legacy actor-head migration helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_ACTION_STEPS = 9
LEGACY_HEADS = (5, 5, 3, 2)


@dataclass(frozen=True)
class ActionSpec:
    """Movement, yaw, pitch and shoot categorical head sizes."""

    precision_steps: int = DEFAULT_ACTION_STEPS

    def __post_init__(self) -> None:
        if type(self.precision_steps) is not int or not 3 <= self.precision_steps <= 2001 or self.precision_steps % 2 == 0:
            raise ValueError("action_steps must be an odd integer between 3 and 2001")

    @property
    def heads(self) -> tuple[int, int, int, int]:
        return (5, self.precision_steps, self.precision_steps, 2)

    def decode_index(self, index: torch.Tensor) -> torch.Tensor:
        midpoint = (self.precision_steps - 1) // 2
        return (index.float() - midpoint) / midpoint


def action_steps_from_config(config: dict) -> int:
    """Schema-3 checkpoints created before configurability imply 9 steps."""
    return int(config.get("action_steps", DEFAULT_ACTION_STEPS))


def _interpolate_rows(rows: torch.Tensor, old_count: int, new_count: int) -> torch.Tensor:
    """Linearly interpolate ordered categorical rows over their decoded values."""
    if old_count == new_count:
        return rows.clone()
    positions = torch.linspace(0, old_count - 1, new_count, device=rows.device, dtype=torch.float64)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    weight = (positions - lower).to(rows.dtype)
    shape = (new_count,) + (1,) * (rows.ndim - 1)
    return rows[lower] * (1 - weight.view(shape)) + rows[upper] * weight.view(shape)


def migrate_actor_tensor(tensor: torch.Tensor, target: ActionSpec) -> torch.Tensor:
    """Map schema-2 actor rows (5/5/3/2) into a target schema-3 layout."""
    if tensor.shape[0] != sum(LEGACY_HEADS):
        raise ValueError(f"Expected legacy actor output size {sum(LEGACY_HEADS)}, got {tensor.shape[0]}")
    move, yaw, pitch, shoot = tensor.split(LEGACY_HEADS, dim=0)
    return torch.cat([
        move.clone(),
        _interpolate_rows(yaw, LEGACY_HEADS[1], target.precision_steps),
        _interpolate_rows(pitch, LEGACY_HEADS[2], target.precision_steps),
        shoot.clone(),
    ], dim=0)
