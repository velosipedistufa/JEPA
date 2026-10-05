# Pinned baselines

Local measurements win over papers. Sources are `research_os/lab/facts.md` and the paths named there. This file does not add a measurement.

## Numbers papers may not override

| pin | value | source |
|---|---|---|
| F2 ridge angle, held-out circular MAE | 13.751980913656148° on 110 test frames; rot180 12.078862900523902° (`n = 20`); near 0, 14.584086219971399° (`n = 19`) | `data/probes/probe_metrics.json` |
| F3 constant Δz residual ratio | 0.9914314089199128 (reports round to 0.991) | `data/probes/language_port_fit.json` |
| F4 official image-goal CEM | 45/50, seed 42, goal offset 25, budget 50 | `reports/gpu_4090/logs/baseline_official.json` |
| F7 in-place oracle rotate | reach 0.00 at `+90°` and `+180°`, budget 50, `n = 20`; `+180°` budget 100 is also 0.00 | `reports/gpu_4090/logs/oracle_rotate.json` |
| F5 ×5 CEM | 0/50, invalid, not a baseline | `reports/gpu_4090/GPU_RUN.md` |
| F12 probe-cost angle hits | episodes 52 and 74 moved 114.95 px and 130.42 px; oracle gate 0/20 | `reports/gpu_4090/logs/probe_cost_official.json` |

K1–K6 stay closed. See `LEDGER.md`.

## F2 split, copied by any readout on this npz

Script: `scripts/probe_theta.py`, `episode_split`. Seed 0. Test fraction 0.25 inside `rotation`, then inside `random_only`. Ridge `alpha = 1.0` with the library default intercept. `z` is passed as the stored float32 array. A float64 cast of the same array misses the published angle MAE by 1.329e-4°. Score is the unweighted mean over test frames.

Counts in `probe_metrics.json`: 150 train episodes, 50 test episodes, 330 train frames, 110 test frames, 440 rows, `z` shape `(440, 192)`.

`x` and `y` are `info["block_pose"][0]` and `[1]`. `scripts/probe_theta.py` `record_frame`.

File identity at the start of cycle 002, before the xy fit:

| file | sha256 |
|---|---|
| `data/probes/pusht_theta_probe.npz` | `7ddf4d39f5cf2bc44dafa345abd7590ee9244e5249026343710d2fdfda7a5121` |
| `data/probes/probe_metrics.json` | `b337e5a4f3368c83ba98d1bdd9457419fb7ecd92e64e183721436102a7b8ec0f` |
| `scripts/probe_theta.py` | `79574d99e46f01f79d744962a96e02b2b50617532da42b4a265d818f26cb7287` |

There is no saved list of F2 test episode ids. Reproducing the published angle MAE is the pin that the split and the ridge match. Cycle 002 aborts before the xy fit if that MAE misses 13.751980913656148° by 0.0001° or more.

## Action z-score, when a later cycle is allowed to plan

Mean `[-0.007812564379916167, 0.006860687229453028]`. Unbiased std `[0.2084674874515821, 0.20674867061239938]`. Source: `reports/gpu_4090/logs/baseline_official.json`, fact F13. A global scale of 5 is K1.

## Cycle 002 xy ridge (seed 0)

Scored result. Held-out frame-level MAE x=12.70093186924053 px, y=9.677809205954556 px. Constant-predictor MAE x=68.70731363573843 px, y=46.259439527506636 px. Commit `6a6b91ab4b2ac63110f681aa604fdb33d5c7d090`. Artifact `research/artifacts/002/verdict.json`. Papers may not replace this pair.

## Cycle 003 xy ridge (seed 1)

Same ridge and the same frame-level MAE. The split seed is 1. Held-out MAE x=11.374770603169216 px, y=11.400641675149357 px. Constant-predictor MAE x=72.38230019438832 px, y=74.19196581739159 px. Ten of the 50 test episodes are shared with seed 0. Commit `afe115aba89517ce64084088982cfc262ab14ed7`. Artifact `research/artifacts/003/verdict.json`. Papers may not replace this pair.

## Cycle 004 episode-level xy ridge

Same fits as cycles 002 and 003. Each test episode counts once. Seed 0: x=11.710440094140232 px, y=9.989942039258745 px. Seed 1: x=10.213864689000905 px, y=10.989067528789096 px. Commit `6be0b91c1b8e5f7ce73f5fca3c5d9406424cc2b2`. Artifact `research/artifacts/004/verdict.json`.

## Cycle 005 15 px episode gate

Same fits. Hit = both episode-mean absolute errors under 15 px. Seed 0: 0.62. Seed 1: 0.66. Constant predictor: 0.00 and 0.04. Commit `23b770776d2bf32085086361b189ca6a119b6ff1`. Artifact `research/artifacts/005/verdict.json`.

## Cycles 009 and 010 velocity

Commanded `v_x` only. `v_y` is identically 0. Cycle 009 R² = -0.0030707419982412, mean paired RMSE 0.812930856939769. Cycle 010 R² = -0.012655494701210301, mean paired RMSE 0.8479230049462124. Pose gate is 5. Artifacts `research/artifacts/009/verdict.json` and `research/artifacts/010/verdict.json`.

## How to rerun the cycle-002 readout

Headless, from the repo root, after `research/artifacts/002/claim.json` is on disk:

```
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
  .venv/bin/python scripts/latent_maps.py fit --family xy \
  --claim research/artifacts/002/claim.json \
  --out research/artifacts/002/xy_ridge_obs.json
```

The fit does not encode, does not call CEM, and does not train. The audit command is separate and is the only pass that applies the 20 px and 40 px lines.

Repo HEAD before the cycle-002 commit: `3af88c8cfb3349127214220753a0305627ead72a`. The probe script was untracked at that commit. The cycle-002 commit pins `scripts/latent_maps.py` and the claim file.
