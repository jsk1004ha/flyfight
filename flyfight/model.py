"""Trainable synthetic leaky RNN. No FlyWire/BANC or anatomical claim."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .actions import ActionSpec, DEFAULT_ACTION_STEPS

HEADS = ActionSpec().heads


class FlyPolicy(nn.Module):
    def __init__(self, width: int = 24, height: int = 16, hidden: int = 768,
                 action_steps: int = DEFAULT_ACTION_STEPS):
        super().__init__()
        if not 16 <= hidden <= 4096:
            raise ValueError("hidden must be between 16 and 4096")
        self.width, self.height, self.hidden = width,height,hidden
        self.action_spec = ActionSpec(action_steps)
        self.heads = self.action_spec.heads
        self.encoder = nn.Linear(width*height*3+5,hidden)
        self.recurrent = nn.Linear(hidden,hidden,bias=False)
        self.actor = nn.Linear(hidden,sum(self.heads))
        self.critic = nn.Linear(hidden,1)
        nn.init.normal_(self.recurrent.weight, std=.35/math.sqrt(hidden))
        nn.init.normal_(self.actor.weight, std=.01/math.sqrt(hidden))
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def encode(self, rgb: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
        pixels = rgb.flatten(1).float()/127.5-1
        x = torch.cat([pixels,previous_action.float()],dim=-1)
        return self.encoder(x)

    def recurrent_step(self, encoded: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return .35*hidden+.65*torch.tanh(encoded+self.recurrent(hidden))

    def forward(self, rgb: torch.Tensor, previous_action: torch.Tensor,
                hidden: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        next_hidden = self.recurrent_step(self.encode(rgb,previous_action),hidden)
        return self.actor(next_hidden).float(), self.critic(next_hidden).squeeze(-1).float(), next_hidden

    def forward_sequence(self, rgb: torch.Tensor, previous_action: torch.Tensor,
                         hidden: torch.Tensor, dones: torch.Tensor
                         ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """Evaluate a PPO sequence while batching the time-independent encoder."""
        steps,batch = rgb.shape[:2]
        encoded = self.encode(rgb.flatten(0,1),previous_action.flatten(0,1)).view(steps,batch,self.hidden)
        states = []
        for t in range(steps):
            hidden = self.recurrent_step(encoded[t],hidden)
            states.append(hidden)
            hidden = hidden*(~dones[t])[:,None]
        states = torch.stack(states)
        return self.actor(states).float(),self.critic(states).squeeze(-1).float(),states


def _policy_distributions(logits: torch.Tensor, heads: tuple[int, ...],
                          exploration_mix: float
                          ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return (sampling probability, sampling log-probability, raw log-probability) per head."""
    if logits.shape[-1] != sum(heads):
        raise ValueError(f"Logit size {logits.shape[-1]} does not match action heads {heads}")
    if not math.isfinite(exploration_mix) or not 0 <= exploration_mix <= .5:
        raise ValueError("exploration_mix must be finite and in [0, 0.5]")
    out = []
    for head in logits.split(heads, dim=-1):
        raw_logp = F.log_softmax(head, dim=-1)
        if exploration_mix == 0:
            mixed_logp = raw_logp
        else:
            policy_term = raw_logp + math.log1p(-exploration_mix)
            uniform_term = torch.full_like(raw_logp, math.log(exploration_mix / head.shape[-1]))
            mixed_logp = torch.logaddexp(policy_term, uniform_term)
        out.append((mixed_logp.exp(), mixed_logp, raw_logp))
    return out


def action_stats(logits: torch.Tensor, actions: torch.Tensor | None = None,
                 *, generator: torch.Generator | None = None,
                 heads: tuple[int, ...] = HEADS,
                 exploration_mix: float = 0.0) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Sample/evaluate the exact behavior policy, including optional uniform mixing."""
    sampled, logps, entropies = [], [], []
    for i, (p, logp, _) in enumerate(_policy_distributions(logits, heads, exploration_mix)):
        a = torch.multinomial(p,1,generator=generator).squeeze(-1) if actions is None else actions[...,i]
        sampled.append(a)
        logps.append(logp.gather(-1,a.unsqueeze(-1)).squeeze(-1))
        entropies.append(-(p*logp).sum(-1))
    return torch.stack(sampled,-1),torch.stack(logps,-1).sum(-1),torch.stack(entropies,-1).sum(-1)


def exploration_regularizer(logits: torch.Tensor, *, heads: tuple[int, ...] = HEADS
                            ) -> torch.Tensor:
    """Uniform-to-policy cross entropy; unlike entropy, it recovers saturated logits."""
    if logits.shape[-1] != sum(heads):
        raise ValueError(f"Logit size {logits.shape[-1]} does not match action heads {heads}")
    return torch.stack([-F.log_softmax(head, dim=-1).mean(-1)
                        for head in logits.split(heads, dim=-1)], -1).sum(-1)


def action_head_entropies(logits: torch.Tensor, *, heads: tuple[int, ...] = HEADS,
                          exploration_mix: float = 0.0) -> torch.Tensor:
    """Per-head behavior-policy entropy for telemetry."""
    distributions = _policy_distributions(logits, heads, exploration_mix)
    return torch.stack([-(p*logp).sum(-1) for p, logp, _ in distributions], -1)
