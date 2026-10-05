# Cycle 001 — Proposer

Eleven bets. Order is not a ranking. `free` means the claim can be stated without a command string. `port` means language is in the claim. A paper citation is a prior. It does not replace F4, F7, F3, or F2.

K1–K5 are closed: guessed action scales, a constant vector per string, blaming a missing text encoder for in-place failure, finetuning the encoder so that 180° appears in `z`, and text-to-image goals. None of the bets below reopen those mechanisms.

## B1 — Fixed planner, longer dataset offset

`free`. Planner limit.

Claim. With the action mean and std in F13, and with every other knob held at F16, raising `goal_offset_steps` from 25 to 50 drops official `eval_state` success to 0.40 or below on 20 episodes.

Why now. F4 is one offset. Caselli et al. 2026 (arXiv:2607.12547, §3.1) already report flat LeWM at 94.0%, 52.7%, and 18.0% for offsets 25, 50, and 75 under a three-seed sweep. C3 says their 94.0% is not F4. The local fixed config has not been moved off offset 25. Maes et al. 2026 (arXiv:2603.19312, §3.2) say autoregressive error grows with horizon and that MPC is the mitigation. F11 shows that 35 of 45 local successes already finish inside the first 25 environment steps, so the second plan is a minority of the 90%.

What would change our mind. Success at offset 50, budget still 50, is at least 14/20.

Rough compute. One 20-episode run on a 4090. The Lance table has to be mounted because the starts are dataset rows. About two hours including setup. Action statistics stay at F13.

## B2 — Expert-action support for an in-place turn

`free`. Planner limit. Wild.

Claim. In-place `+180°` fails under F7 because CEM draws normalized actions from `N(0, I)`, off the expert-action support. Sampling candidates as small residuals around real training action chunks, then z-scoring with F13, raises `+180°` in-place reach above 0.30 at budget 50 on the same 20 starts.

Why now. F7 gives the planner a true rendered goal and still scores 0.00. F2 says the angle is already in `z`. Caselli et al. 2026 (§3.3) show that unconstrained CEM can pick latent macro-actions that look good under a terminal latent cost and that constraining search to macro-actions encoded from training trajectories changes Push-T success at long dataset offsets (64.0% at offset 50 with staged execution; 32.7% at offset 75 online). Those gains are on dataset goals. They are not a measurement of in-place rotation. The transfer is the bet.

What would change our mind. The constrained sampler still has in-place `+180°` reach at most 1/20 under the oracle gate (angle `< 20°` and xy `< 15` px).

Rough compute. A new sampler plus the action column of `pusht_expert_train.lance`. A 4090 day if the table is already mounted. This is the large one.

## B3 — A chain of small rendered turns

`free`. Planner limit.

Claim. Three successive rendered goals at `+30°`, `+60°`, and `+90°`, each chased with the official planner for 25 environment steps, leave the block within 20° of `+90°` and within 15 px of the start xy on at least 8 of 20 held-out starts.

Why now. F7 says a single `+90°` or `+180°` image is not followed. The open node in `lab/graph.md` is whether a chain of nearby images is followable. Caselli et al. 2026 (§3.2) find that true future latents from a dataset trajectory are often executable (`73.3 ± 4.2%` at offset 50). C4 says those goals are on a demonstration. A same-xy angle chain is a different set. Maes et al. equation (4) is the cost each stage would use.

What would change our mind. Fewer than 2 of 20 starts meet the angle and xy gate at the end of the third stage.

Rough compute. Reuse `scripts/oracle_rotate_cem.py`. Sixty CEM solves. Under two hours on a 4090 once F13 is hardcoded. No Lance.

## B4 — The selected oracle plan already misses in latent space

`free`. Planner limit.

Claim. On the F7 `+180°` budget-50 runs, the plan CEM returns has imagined terminal angle, read by the F2 ridge from `z_H`, still more than 60° from the commanded angle. The optimizer never presents an imagined latent that matches the rendered goal.

Why now. F7 records executed angle and `\|\|z_g − z_t\|\|`. It does not record the cost or the decoded angle of the chosen plan. Maes et al. equation (4) defines that cost. A high executed error is compatible with two stories: imagination found the goal and the environment did not follow, or imagination never found it. F14 is one ordinary macro-action with prediction error 1.87, which does not decide the in-place case.

What would change our mind. Median imagined angle error across the 20 selected plans is below 20°, while executed error stays above 60°.

Rough compute. Add two scalars to the oracle loop and rerun the 20 × budget-50 `+180°` condition. About two hours on a 4090.

## B5 — Block position on the probe table

`free`. Representation.

Claim. On the existing 440 rows and the same episode split as F2, a ridge from `z` to block `(x, y)` has held-out mean absolute error below 20 px on each coordinate.

Why now. The local probe asked only about angle. The npz already stores `x`, `y`, and `z` (F18). Maes et al. 2026, Table 1, report a linear block-location correlation of 0.986 on their Push-T probe. That number is a correlation, not pixels, and it is not this split. F2's decision rule does not mention position.

What would change our mind. Held-out MAE is at least 40 px on either coordinate.

Rough compute. Minutes on the Xeon. No encode. `data/probes/pusht_theta_probe.npz`.

## B6 — A linear rotation operator on `z`

`free`. Representation. Wild.

Claim. There is one matrix `R` in `R^{192×192}` such that, on held-out same-xy `+90°` pairs, the residual RMS of `R z − z'` divided by the RMS of `z' − z` is at most 0.70.

Why now. F3 says one added vector per command leaves a ratio of 0.991. F2 says a linear readout of angle works at every tested slice, which does not imply that the turn itself is a linear map on `z`. Maes et al. §5.1 report temporal straightening of expert-time trajectories. Straightening is a statement about consecutive frames in a demonstration, not about a fixed-xy quarter turn. The two claims can both be true.

What would change our mind. The held-out ratio is at least 0.90, in the same band as F3.

Rough compute. A least-squares fit on the pairs already implied by the probe table. Under two hours on the Xeon. No encode.

## B7 — State-dependent command displacement

`port`. Language as a map into `z`, and not a goal image.

Claim. A linear map from `(z_t, one-hot command)` to `Δz`, fit on the train pairs, has held-out residual RMS over target RMS at most 0.70 for the three rotate commands.

Why now. K2 killed a vector that ignores `z_t`. The GPU skill MLP is a nonlinear version of this idea and was scored only under the ×5 planner (F15): held-out residual L2 11.68 against target L2 16.28, then CEM reach near zero. Myers et al. 2023 (arXiv:2307.00117) train an embedding of the change between start and goal and then train a policy. That is a different object from a constant lookup. F3 conflicts with the constant special case and does not measure a state-dependent linear map on this checkpoint. CLIP is not part of the claim. F18 says the weights were never downloaded, and four template strings would still be four discrete codes.

What would change our mind. The held-out ratio is at least 0.90. If it beats 0.90 and the language-free matrix from B6 matches it within 0.05, the command channel is not what the fit is using.

Rough compute. Same Xeon afternoon as B6. Pairs come from `pusht_theta_probe.npz` and the split in `scripts/language_port.py`. No CEM.

## B8 — Horizon-wise open-loop latent error

`free`. Representation of the predictor.

Claim. On 20 dataset windows, `\|\|z_pred − z_real\|\|` after five macro-actions (the F16 horizon) is at least twice the one-step error, using F13 for the action transform.

Why now. F14 is a single window: one-step error 1.871 against a real step of 8.984. Maes et al. §3.2 state that autoregressive rollouts accumulate error. The local file has no curve. F19 records a 3-frame predictor window and a 1-frame pixel stack; the script comment treats the 3 as an attention width, so this bet does not smuggle in a second history length.

What would change our mind. The median five-step error on those 20 windows is below 2.5, close to the single one-step value 1.871.

Rough compute. A loop around `one_step_match` in `scripts/eval_official_cem.py`. Requires the Lance table. A few hours on a 4090.

## B9 — The reset goal image, under the measured z-score

`port`. This is the goal-image shelf. The string only names the environment's own goal image.

Claim. Official CEM with F13, chasing the encoding of `info["goal"]` after a live `reset`, succeeds on at least 12 of 20 episodes by the F9 predicate.

Why now. The 0/50 live goal-image run is F5, the invalid ×5 scale. F4 chases a dataset frame 25 steps ahead, which is a different goal distribution. The probe note records that the default goal overlay in `reset_api.json` is `(256, 256, π/4)`. Nobody has chased that image with the z-score. Caselli et al. §3.1 is a prior that flat LeWM gets weaker as the dataset offset grows to 75 (18.0% in their sweep). A random reset to the fixed target can be a long push. Their 18.0% is not a forecast for this bet.

What would change our mind. Success is at most 4/20.

Rough compute. Twenty live episodes. Hardcode F13. No Lance. About two hours on a 4090. `scripts/eval_official_cem.py` already has the cost and the denormalizer.

## B10 — Does the probe table contain a neighbor of the commanded pose?

`port`. Index, before any planner.

Claim. For a held-out start and a rotate command, the nearest other row in `pusht_theta_probe.csv` matches the commanded block pose within 15 px and 15°.

Why now. A course index needs a gallery. The gallery on disk is 440 frames, not the Lance set (F18). F7 already shows that a perfect same-xy render is not executed, so this bet is about whether a cache even has a geometric neighbor. It becomes a CEM bet only after that neighbor exists. F12 says the angle-only probe cost did not produce in-place hits.

What would change our mind. On the 20 held-out rotation starts, the median nearest-neighbor xy error is above 30 px, or the median angle error is above 20°.

Rough compute. A pass over the csv. Minutes on the Xeon.

## B11 — Agent velocity is absent from one frame

`free`. Representation. Wild.

Claim. A ridge from a single-frame `z` to agent velocity has held-out `R²` below 0.20 when velocity is varied and the block pose is held fixed.

Why now. F2 and Maes et al. Table 1 cover pose. F18 says the saved table has agent position and no velocity. One RGB frame has no motion, so a high `R²` would mean the encoder is using a confound (a pose that happened to correlate with velocity in the collection) or a pixel change from `_set_state`. The collection in `scripts/probe_theta.py` zeros velocity on scripted turns and does not store it.

What would change our mind. Pose-matched pairs with two different velocities have test `R²` above 0.50.

Rough compute. A few hundred headless renders and CPU encodes at about 0.123 s each (F1), then a ridge. A half day on the Xeon. This planning cycle does not encode them.
