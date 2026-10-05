# Cycle 001 — Designer

Killed bets: none. Cut bets are parked at the end with a stop rule that forbids starting them. Every CEM below uses the mean and unbiased std in F13, then the inverse transform and clip in `scripts/eval_official_cem.py`. A factor of 5 is a failed run, not a baseline.

This cycle does not execute these protocols.

## P1 — Offset 50, same planner (B1, keep)

Hypothesis. Holding F16 fixed, dataset goals 50 steps ahead succeed on at most 8 of 20 episodes.

Data. Missing file: `pusht_expert_train.lance`, columns `pixels`, `action`, `state`, `proprio`, which `open_dataset` already requests. The live env cannot supply a frame that is 50 expert steps ahead of a logged start. Action mean and std are already in `reports/gpu_4090/logs/baseline_official.json` and are hardcoded, not re-estimated.

Script. Extend `scripts/eval_official_cem.py` with a `goal_offset_steps` argument. Default stays 25.

Machine and cap. Timed 4090, 2 hours. Xeon cannot open the Lance table (F8, F18).

Metric. `eval_state` successes out of 20, seed 42, budget 50, horizon 5, receding horizon 5, action block 5, CEM 300/30/top 30.

Success number. At most 8/20. That confirms the drop.

Falsifier. 14/20 or more. The fixed local planner does not drop. Stop. Do not sweep CEM to chase Caselli et al.

Stop rule. If the hardcoded mean differs from F13, discard the run. If the first 10 episodes are all failures and the mean position L2 is above 100, stop and check that the denormalizer was used. Tear the rental down at 2 hours whatever the count is.

## P3 — Three rendered +30° stages (B3, keep)

Hypothesis. A chain of nearby in-place renders is followable on at least 8 of 20 starts even though the single `+90°` oracle is not (F7).

Data. On disk: `data/probes/pusht_theta_probe.npz` and F13. No Lance. Goals are rendered with `_set_state`, as in `scripts/oracle_rotate_cem.py`.

Script. Add a staged mode to `scripts/oracle_rotate_cem.py`. Stages at `+30°`, `+60°`, `+90°` from the start angle. Each stage runs budget 25. The goal latent is re-encoded from the render at that absolute angle and the original xy.

Machine and cap. Timed 4090, 2 hours.

Metric. After stage 3, fraction of 20 held-out starts with wrapped angle error `< 20°` and xy drift `< 15` px. Same gate as the oracle script.

Success number. At least 8/20.

Falsifier. At most 1/20. The chain is in the same regime as F7.

Stop rule. After the first 5 starts, if every stage increased xy by more than 30 px, stop. The chain is not staying in place, and the rest of the 20 will not repair the claim. Tear the rental down at 2 hours.

## P4 — Log the imagined terminal of the oracle plan (B4, keep)

Hypothesis. The plan returned for in-place `+180°` still has ridge-decoded angle error above 60°.

Data. Same starts and renders as F7. F2 ridge weights can be refit from the npz on the train episodes, which `fit_angle_ridge` in `scripts/eval_official_cem.py` already does.

Script. In `scripts/oracle_rotate_cem.py`, after `cem` returns, record `\|\|z_H − z_g\|\|` and the circular angle error of the F2 readout at `z_H`. Rerun only `rotate_180` at budget 50.

Machine and cap. Timed 4090, 2 hours. Can share a rental with P3.

Metric. Median imagined angle error over 20 starts, plus the executed angle error already defined in F7.

Success number. Median imagined error `> 60°`. The optimizer is not finding the rendered goal inside the horizon.

Falsifier. Median imagined error `< 20°` while median executed error stays `> 60°`. Imagination matched the goal and the environment did not follow.

Stop rule. If executed reach is suddenly above 0.30, stop and treat F7 as unreproduced before interpreting the imagined angle. Do not train the `z_g` MLP in that case either until a written follow-up says so. The current skip in `logs/pred_zg_cem.json` stands for this protocol.

## P5, P6, P7, P10 — One CPU script on the probe table (keep)

Hypothesis. On rows already stored: block xy is linearly readable in pixels (B5); a single matrix beats a constant `Δz` on `+90°` pairs (B6); a linear map that sees `z_t` and the command beats the constant ratio (B7); the table is too sparse to be a pose index (B10).

Data. `data/probes/pusht_theta_probe.npz`, `data/probes/pusht_theta_probe.csv`, split from `scripts/probe_theta.py` and `scripts/language_port.py` (episode split, seed 0). No new file.

Script to write. `scripts/latent_maps.py`. It must not call `encode`, CEM, or `train.py`.

Steps.

1. Rebuild the F2 episode split. Fit a ridge (`α = 1`) from `z` to `(x, y)` on train rows. Report per-coordinate held-out MAE.
2. Build same-xy pairs the way `language_port.py` does (turn within 20°, xy shift at most 5 px). Fit `R` by least squares on train `+90°` pairs only, mapping `z` to `z'`. Report test RMS(`R z − z'`) / RMS(`z' − z`).
3. Fit `Δz` on `[z_t ; one-hot]` for the three rotate commands. The one-hot is required. A column of ones with no `z_t` is the K2 model and is only printed as a check against 0.991. Report the held-out ratio for the state-dependent map, and the absolute gap versus the ratio from step 2.
4. For each held-out rotation start and each of `+90°`, `180°`, `−90°`, find the nearest other row by block xy and report the angle error at that neighbor. Exclude the start row.

Machine and cap. Xeon, 2 hours. No GPU and no download.

Metric and success numbers.

- B5 succeeds if both held-out MAEs are `< 20` px. It dies if either is `≥ 40` px.
- B6 succeeds if the test ratio is `≤ 0.70`. It dies if the ratio is `≥ 0.90`.
- B7 succeeds if its test ratio is `≤ 0.70`. It dies if the ratio is `≥ 0.90`. If B7's ratio and B6's ratio differ by `< 0.05`, the command channel did not earn its keep on `+90°`.
- B10 dies, as an index, if the median neighbor xy error is `> 30` px or the median angle error is `> 20°`. It survives as a catalog only if the median neighbor is inside 15 px and 15°.

Falsifier. The death numbers above. Between a success line and a death line the result is inconclusive: write the number and stop, with no second model class.

Stop rule. If pair construction yields fewer than 30 train `+90°` pairs, stop and report the filter. Do not relax the 5 px / 20° pair gate to fill the sample. Do not load CLIP.

## P9 — Live reset goal with the official z-score (B9, keep)

Hypothesis. Chasing `info["goal"]` from a live reset, with F13, succeeds on at least 12 of 20 episodes.

Data. No Lance. The goal image is the env render of the goal pose, which the probe note and `reset_api.json` put at `(256, 256, π/4)` when goal randomization is off. F13 is hardcoded from `baseline_official.json`.

Script. A short driver next to `scripts/eval_official_cem.py` that resets `swm/PushT-v1` with seeds `0..19`, encodes `info["goal"]`, and calls the existing CEM and `eval_state` path. `scripts/gpu_run.py` is not the driver. That file multiplies by 5.

Machine and cap. Timed 4090, 2 hours.

Metric. Successes under F9 out of 20.

Success number. At least 12/20.

Falsifier. At most 4/20.

Stop rule. At 10 episodes, if successes are 0 and mean position L2 is above 100, stop and confirm the logged action mean matches F13. A run that used a global scale of 5 is discarded. Tear the rental down at 2 hours. Do not generate a goal image from text.

## P11 — Pose-matched velocity probe (B11, keep)

Hypothesis. With block pose held fixed, a ridge from one-frame `z` to `(v_x, v_y)` has test `R² < 0.20`.

Data. Not on disk. New frames have to be rendered. The protocol does not need Lance or HDF5: velocity is set with `_set_state`, which the probe script already uses for angle.

Script to write. `scripts/probe_velocity.py`. For 40 train and 20 test episodes (new seeds, disjoint from `0..79` and `1000..1119`), render the same block and agent pose at velocity `(0, 0)` and at `(50, 0)` in env units, encode each frame, and fit the ridge on the train half only.

Machine and cap. Xeon, 3 hours. CPU encode is about 0.123 s a frame (F1). No 4090. This planning cycle does not launch it.

Metric. Held-out `R²` of the two velocity coordinates, and the mean pixel L2 between the paired renders.

Success number. `R² < 0.20`, which supports the claim that one frame does not carry velocity.

Falsifier. `R² > 0.50`.

Stop rule. If the mean pixel L2 between a pair exceeds 5, the pose match failed and the encode stops. A high `R²` on unmatched pixels is not the claim. Cap the collection at 120 frames.

## Parked cuts

### P2 — Empirical-action CEM for in-place 180° (B2, cut)

Do not start. The missing file is the `action` column of `pusht_expert_train.lance`, used as a bank of expert chunks. The live env can draw actions and cannot reconstruct the training support. A future protocol would perturb those chunks, z-score with F13, and run the F7 oracle gate. The death line is in-place `+180°` reach `≤ 1/20`. The revive line is reach `≥ 0.30`. Hour cap on a future rental: 8. Stop earlier if building the bank requires a new download during a planning cycle.

### P8 — Five-step open-loop error (B8, cut)

Do not start. The missing file is the same Lance table, columns `pixels`, `action`, and `state`, because the real future latent has to come from a dataset window. `one_step_match` in `scripts/eval_official_cem.py` is the kernel. The death line of the accumulation claim is a median five-step `\|\|z_pred − z_real\|\| < 2.5` on 20 windows. The confirm line is a median at least twice the paired one-step error. Hour cap if the table is already mounted: 3. No download is authorized to unpark this.
