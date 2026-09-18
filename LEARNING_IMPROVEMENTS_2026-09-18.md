# Combat-learning improvement and validation

## Acceptance criteria

- Keep 201/2001-step action precision and headshot rewards.
- Prevent accidental sampling/PPO probability mismatch; report raw policy entropy separately from forced exploration.
- Supply useful combat signals without rewarding shots in empty space or farming static aim.
- Preserve the source checkpoint and logs; use a separate, explicit transfer-learning run.
- Keep the actual training-environment live viewer and unattended server training.
- Separate unit/integration correctness from measured combat skill.

## Implementation

`combat-v1` uses 8 cloud environments, horizon 64 (512 transitions per update instead of 128), learning rate 0.0001, entropy coefficient 0.03, per-head uniform exploration mixture 0.10, target KL 0.02, advantage normalization denominator floor 0.1 and raw-policy uniform cross-entropy prior 0.002. All precise action categories remain available.

Effective damage shaping is `0.5 * (HP damage dealt - HP damage received) / 100`. Headshot bonus stays separate, and victory counters use only terminal outcomes. Aim shaping is `0.05 * (gamma * Phi(next) - Phi(previous))`, with terminal potential zero and line-of-sight gating. It uses privileged world state ONLY for training rewards, not policy inputs. No blanket policy-optimality guarantee is claimed for this partially observed self-play system.

Easy spawn probability starts at 0.8 and declines to 0 over 2,000 updates. Easy pairs are 6–12 units apart, unobstructed and approximately facing each other; other episodes retain the original full-map random start. Evaluation always uses the original map distribution without curriculum or extra exploration.

Cloud startup selects `FLYFIGHT_LEARNING_PROFILE=combat-v1` by default. `legacy` retains the original run. The new run lives in `/data/flyfight/combat-v1`, preserves the old source, copies learned encoder/recurrent weights, softens actor logits by 20x and resets critic/Adam/rollout/counters. `transfer.json` records source hash and prior progress. This is explicitly a new objective/transfer experiment, not exact checkpoint continuation. Existing `combat-v1` runs resume normally and are not repeatedly reset.

## Comparison design recorded before running

- Question: does the complete improved training package reduce collapse and improve combat relative to the previous configuration under an equal environment-step budget?
- Independent units: training seeds, not individual frames. Initial weights, evaluation seed, frozen opponent and role balancing are matched within each seed.
- Planned smoke comparison: seeds 17 and 29; 32,768 environment steps per profile; hidden 64, same RGB/map/action precision; 16 evaluation games per identity, evaluate both identities. Randomize profile order with a reproducible schedule.
- Report training shots/hits/headshots, raw/behavior entropy, KL, throughput and draw rate; separately report fixed-opponent full-map evaluation before and after. Easier training spawn metrics cannot establish final-map skill improvement.
- This small two-seed study is exploratory and not a production-model performance guarantee. Do not tune repeatedly to the same evaluation seed and call it held-out confirmation.

## References

- PPO and KL early stopping: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- PPO paper: https://arxiv.org/abs/1707.06347
- Potential shaping: https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf

## Validation results

- Full Python suite: 102 passed after integration.
- Production-shape CPU smoke: hidden 768, envs 8, horizon 64, 201 action bins; one successful 512-step update, raw entropy 12.9088, behavior entropy 12.9089, sampled KL 0.000400, 215 shots and 5 hits. This is execution evidence with easy starts, NOT learned-skill improvement.
- Controlled comparison is still running; no combat superiority claim is made.
- Online rollout remains dependent on Git signing; do not treat local changes as a production deployment.
