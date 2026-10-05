# Open questions

Cycle n+1 may read this file, `ledger.jsonl`, and `baselines.json`. Nothing else.

## Active

(a) SIGReg lambda is the stability knob on Push-T.
Factor: `sigreg_weight` (`loss.sigreg.weight` in `le-wm/config/train/lewm.yaml`).
Falsifier shape: one change of that weight, against baseline-0, moves a non-collapsed latent metric by a margin named in the claim.
Degenerate block: latent collapse to a constant.
Status: tested. sigreg_weight=0.18 produced latent_std=0.1748 (vs baseline 0.201). Weight increase reduces latent variance. Open replications and other values.

(b) Latent dim below 192 loses CEM success on Push-T.
Factor: `latent_dim` (`embed_dim` in `le-wm/config/train/lewm.yaml`), value below 192.
Falsifier shape: CEM success falls relative to baseline-0. The metric has to be planner success.
Degenerate block: a metric that ignores planning.
Status: open. baseline-0 is filled (metric=0.201). A causal comparison is now possible.

## Proposer format
The JSON object must have exactly these keys: hypothesis, claim, falsifier, degenerate_block, factor, factor_value, baseline_id, budget_minutes, budget_tokens.
claim and falsifier must each be a single sentence with no period inside.

## Parked

(c) Two-Room failure is low intrinsic dimension fighting the Gaussian regularizer.
Degenerate block: declaring failure from prediction loss alone.
Status: parked. Do not propose (c).

## Human gate

If the last ledger row has `"stop": "escalate"`, the next cycle stays stopped until this file contains a line `human-ack: <cycle number>` for that cycle.
