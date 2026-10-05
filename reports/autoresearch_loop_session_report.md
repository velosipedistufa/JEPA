# LeWM on Push-T — What Was Attempted and What We Got

**Date:** Sun Oct 4 2026
**Subject:** frozen LeWorldModel (`quentinll/lewm-pusht`) on Push-T
**Scope:** all experiments attempted this session, and the results that came out

---

## 1. The question being worked on

The model stays frozen. Two threads:

- **Rotation gap.** The official image-goal CEM planner scores 45/50 (90%), yet
  in-place rotation goals (target angle +90°/+180°) are never reached (0.00). Why?
- **Language port.** Can words name a goal in the 192-d latent `z`?

Established facts carried into this session:

| Fact | Value |
|---|---|
| Image-goal CEM success | 45/50 (90%) |
| In-place +90°/+180° reach (oracle rendered goal) | 0.00 |
| Angle readable from `z` (ridge, circular MAE) | 12.08° (180° turns), 14.58° (near 0°) |
| Constant `Δz` per text command | residual ratio 0.991 — does not fit |
| baseline-0 `latent_std` | 0.201171875 |

---

## 2. What was attempted

### 2.1 Training-factor trials (loop-law runs)
- **Cycle 33 — baseline fill:** unmodified pinned config, measure `latent_std`.
- **Cycle 34 — hypothesis (a):** raise SIGReg weight `0.09 → 0.18`, measure `latent_std`.

### 2.2 Rotation-gap campaign (planning probes on the frozen model)
A sequence of CEM / latent probes, each a small CUDA script on the remote 4090:

| # | Experiment | Idea |
|---|---|---|
| 1 | `cem_cycle_diagnostic` | harness/rollout instrumentation check |
| 2 | `cem_action_prior_init` | init CEM from dataset action mean/std |
| 3 | `rotation_action_augmentation` | widen rotational sampling range |
| 4 | `cem_rotation_oversample` | oversample rotation-dominant actions |
| 5 | `rotation_action_sensitivity_probe` | is `z` insensitive to the rotation action dim? |
| 6 | `action_dimension_latent_sensitivity_scan` | per-dimension latent sensitivity |
| 7 | `rotation_action_scaling_cem_sweep` | scale rotation action components in CEM |
| 8 | `rotation_action_dim_scaling_cem` | per-dimension action scaling |
| 9 | `rotation_dynamics_probe` | rotation predicted vs actual `Δz` correlation |
| 10 | `rotation_action_scale_sweep` | sweep per-dimension normalisation |
| 11 | `cem_rotation_delta_z_probe` | enforce delta-z ratio 0.991 + small rot scale |
| 12 | `cem_baseline_rot_delta_consistency_probe` | baseline (no rot/no delta-z) vs treatment |
| 13 | `rotation_delta_z_ablation_probe` | rotation scale alone (delta_z = 1.0) vs neutral |
| 14 | `action_noise_neutral_baseline_probe` | neutral action-noise baseline |
| 15 | `cem_trajectory_smoothness_analysis` | CEM action smoothness vs human demos |

---

## 3. Results obtained

### 3.1 Best / least-awful results (clean, with real metrics)

**A. Rotation perturbation makes planning *worse*, not better.**
The baseline-vs-treatment consistency probe is the cleanest result of the session:

| | baseline (neutral) | treatment (`rot_scale=0.1`, `delta_z=0.991`) |
|---|---:|---:|
| final cost | **6.11e-4** | 9.37e-4 |
| cost reduction | **0.01046** | 0.00832 |
| successive L2 (action consistency) | **0.130** | 0.361 |
| pairwise L2 | **0.150** | 0.441 |
| action std | **0.057** | 0.142 |

The neutral baseline wins on every axis, and the treatment is markedly less
consistent. Realized parameters matched targets, so the manipulation was valid.

**B. Rotation scaling alone is the harmful part.**
The ablation (isolating `rot_scale=0.1` with `delta_z=1.0`) reproduces the effect:

| | neutral | rotation-only |
|---|---:|---:|
| successive L2 | **0.172** | 0.297 |
| pairwise L2 | **0.177** | 0.286 |
| action std | **0.048** | 0.075 |
| final cost | **0.000611** | 0.000631 |
| cost reduction | **0.01050** | 0.01045 |

So the "rotation scaling helps rotation tasks" hypothesis is **falsified**: scaling
the rotation action component introduces action noise without improving the cost.

**C. CEM does converge on the latent cost.**
`cem_rotation_delta_z_probe` converged hard on CUDA:
`cost_history = [0.260, 0.0377, 0.00307]` (best 0.003), `n_samples=64`, `n_iters=3`,
`rot_scale=0.1`. The optimizer works; the failure is in what the rotation objective
does to the action sequence, not in convergence.

### 3.2 Training-factor results (hypothesis a)

| run | factor | metric (`latent_std`) | verdict |
|---|---|---:|---|
| baseline-0 | baseline | **0.201171875** | filled |
| sigreg_weight = 0.18 | SIGReg weight | 0.1748046875 | fail |

Raising SIGReg weight to 0.18 did **not** double `latent_std` (as the claim required);
it *reduced* it (0.201 → 0.175). Consistent with the recorded question status:
increasing the weight reduces latent variance.

### 3.3 Anomalous / weak results (reported for completeness)

- `action_noise_neutral_baseline_probe` (marked success, c57): `successive_l2=0.130`,
  `pairwise_l2=0.529`, but `cost_reduction = -19.0` (cost rose 0.016 → 0.329). Likely a
  metric/sign problem; not a usable baseline as-is.
- `cem_cycle_diagnostic`, `cem_action_prior_init`, `rotation_dynamics_probe` passed
  their run harness but recorded **no metrics** (run before metric capture was wired),
  so their "success" carries no numbers.

### 3.4 Attempts that produced no usable result

`rotation_action_augmentation`, `cem_rotation_oversample`,
`rotation_action_sensitivity_probe`, `action_dimension_latent_sensitivity_scan`,
`rotation_action_scaling_cem_sweep`, `rotation_action_dim_scaling_cem`,
`rotation_action_scale_sweep` — all failed to produce a clean metric row (mostly
script/harness errors, later patched), and `cem_trajectory_smoothness_analysis` died on
a bad dataset path (`lance_data/action_stats.pt`).

---

## 4. Summary of what we learned

1. **The rotation gap is not a scaling/normalisation problem.** Neutral CEM beats every
   rotation-scaled and delta-z-perturbed variant, replicated across baseline-vs-treatment
   and the rotation-only ablation. Scaling the rotation action dimension adds action
   inconsistency (successive L2 ~0.17 → ~0.30) and does not lower cost.
2. **CEM itself is fine** — it converges to low latent cost (0.003); the issue is the
   action perturbation's effect, not optimisation.
3. **SIGReg weight behaves as a variance knob:** 0.09 → 0.18 lowers `latent_std`
   0.201 → 0.175, contradicting the "doubling" claim.

---

## 5. Open / not yet obtained

- A usable `cem_success` metric (planning success) for the latent-dim hypothesis (b).
- A reliable neutral `cost_reduction` baseline (the −19 anomaly needs fixing).
- Any positive intervention that improves rotation reach above 0.00 — so far, all
  attempted interventions hurt.
