# Cycle 001 — Skeptic

No bet is a rewording of K1–K5. None asks for a guessed action scale, a from-scratch LeWM train on the Xeon, or a "looks better" score. None treats the in-place miss as a missing text encoder. Two bets are real and too big for this group. The other nine have a falsifier a student can run.

## B1 — Fixed planner, longer dataset offset

Keep. The local config in F16 has been measured only at offset 25 (F4), and Caselli et al. 2026 §3.1 already saw a drop under a different sweep (C3), so the only surviving claim is the fixed local planner. Rests on F4, F11, F16, C3.

## B2 — Expert-action support for an in-place turn

Cut. The support-constraint idea is already a paper (Caselli et al. 2026 §3.3) on dataset goals, and a faithful local version needs a new sampler plus the Lance action column. That is a second project. It is not K3: the goal image stays the rendered oracle from F7. Rests on F7, F2, F13.

## B3 — A chain of small rendered turns

Keep. Caselli et al. 2026 §3.2 measured dataset-future subgoals (C4). They did not measure a same-xy `+30°` chain, and F7 only covers the single large jump. The oracle script and F13 are enough. Rests on F7, F13, F16, C4.

## B4 — The selected oracle plan already misses in latent space

Keep. F7 logs the executed state and `\|\|z_g − z_t\|\|`, and the chosen plan's latent cost is absent. One extra pair of scalars decides which story is open. Rests on F7, F2, F14.

## B5 — Block position on the probe table

Keep. Maes et al. 2026 Table 1 already reports a block-location correlation of 0.986 on their probe. That is not a pixel MAE on this npz, and the fit uses rows already on disk (F18). Rests on F2, F18.

## B6 — A linear rotation operator on `z`

Keep. K2 is a constant vector. A single matrix is a different mechanism, and the 0.90 kill line matches the band F3 already published. Maes et al. §5.1 (temporal straightening along expert time) does not settle a fixed-xy quarter turn. Rests on F3, F2.

## B7 — State-dependent command displacement

Keep. The protocol has to fit coefficients on `z_t`. A fit that ignores `z_t` is K2 and stays dead. GRIF (Myers et al. 2023) trains a policy on an embedding of an image change; it is not this linear map, and F3 does not refute it. F15 is the nonlinear head under the ×5 planner, so it is not this result. Rests on F3, F15, K2.

## B8 — Horizon-wise open-loop latent error

Cut. The claim matches Maes et al. §3.2, and the local evidence is one window (F14). Estimating the curve needs the Lance table, which is not on this machine (F18) and is not a three-student afternoon. Rests on F14, F18, F19.

## B9 — The reset goal image, under the measured z-score

Keep. F5 is the ×5 live-goal run and is not evidence about this planner. F4 is a dataset frame 25 steps ahead. The falsifier is an `eval_state` count with the mean and std in F13. Rests on F4, F5, F9, F13.

## B10 — Does the probe table contain a neighbor of the commanded pose?

Keep. This asks whether a cache has a geometric neighbor. It does not launch CEM and does not lean on the angle-only 0.10, which F12 shows were translations of 115 px and 130 px. Rests on F12, F18, F7.

## B11 — Agent velocity is absent from one frame

Keep. The falsifier is a number (`R²`), the pose is matched so a high score cannot hide in the angle probe, and the encode budget is a few hundred CPU frames (F1). It is a new collection because F18's table has no velocity column. Rests on F1, F2, F18.
