# JEPA research loop

Human-readable copy of the loop law. `scripts/cycle.py` is the authority if this file and the runner disagree.

The OpenRouter models propose, patch, and judge. They do not replace the world model. A local Python trial is the only thing that can produce a metric, and this setup does not download data or start that trial by itself.

## Loop law

- Closed loop. Cycle n+1 may read only the ledger, open questions, and pinned baselines from cycle n. No chat memory, no unscored notes, no rereading the repo as hidden state.
- Claim before action. No run and no reading pass without a one-sentence claim and a falsifier checkable in this cycle. If it is not checkable, split the claim.
- Separate invention from judgment. The pass that proposes the hypothesis does not grade it. Different model, different prompt, no shared scratchpad.
- One intervention. One factor against a frozen baseline. Multi-factor jumps cannot support a causal claim. The runner rejects a diff that touches more than one allowed factor.
- Literature is a constraint set. Keep claims, assumptions, metrics, and failure modes. A paper enters the ledger only if it changes a hypothesis, a baseline, or a stop rule. That admission is the only routine path to Kimi K3.
- Name the degenerate solution. Every objective has a trivial way to pass (collapse, leakage, metric gaming). The cycle must block it before the run.
- Negative results close cycles. A killed hypothesis stays in the ledger.
- Reproducibility is part of the claim. Seed, config, split, metric script, and commit are required before a number counts. A missing field voids the number.
- Budget is part of the hypothesis. "Works if we scale" is outside the cycle. Each claim names a wall-clock cap and a token cap. Default trial cap 20 minutes. A cycle that calls K3 or GLM 5.2 must record that cost in the ledger row.
- Forced stop. Each cycle ends in exactly one of: continue, deepen, pivot, kill, escalate. The runner refuses to start cycle n+1 without that field on cycle n. continue and kill never call a deep model. deepen may call GLM 5.2 once. escalate must call Kimi K3 once, then stop for a human.

## Models

Cheap roles never use GLM 5.2 or Kimi K3.

| Role | Model | Fallback | max_tokens |
|---|---|---|---|
| propose | `deepseek/deepseek-v4-flash` | `z-ai/glm-4.7-flash` | 800 |
| patch | `qwen/qwen3-coder-30b-a3b-instruct` | `deepseek/deepseek-v4-flash` | 1200 |
| judge | `deepseek/deepseek-v3.2` | `z-ai/glm-4.7-flash` | 600 |
| deepen | `z-ai/glm-5.2` | `qwen/qwen3.8-27b` | 1200 |
| escalate or admit | `moonshotai/kimi-k3` | `z-ai/glm-5.2` | 1000 |

The judge slug must differ from the proposer slug. The runner allows one Kimi K3 call per cycle. Every call sets provider `max_price` from the catalog price of that exact slug. `:free`, `:latest`, prime, and max aliases are refused.

## Allowed factors

One of: SIGReg weight, latent dim, predictor depth, CEM population, or CEM iterations. CEM population and CEM iterations are separate factors. The first open questions use only SIGReg weight (a) and latent dim (b).

## Launch

From the repo root, after `set -a; source ~/.config/openrouter/env; set +a`:

```bash
source scripts/env.sh
python scripts/cycle.py launch
```

`scripts/launch.sh` does the same thing. `launch` first calls the OpenRouter setup agent. A missing `pusht_expert_train.h5` is that agent's job. The runner then runs propose, run, judge, and close for one cycle. `run` still voids the metric if the file is absent after setup, or if the pinned config asks for a GPU and none is visible.

Smoke, which does not call GLM 5.2 or Kimi K3:

```bash
python scripts/smoke_openrouter.py
```
