# Cycle 001

This was a planning cycle on the Xeon. It read the reports, the 4090 logs, the official CEM script, and three papers. It did not SSH, did not download Lance or HDF5, did not encode frames, and did not run CEM or `train.py`. The RX 580 stayed off. Connection details in `reports/gpu_4090/GPU_RUN.md` are not copied here.

`reports/HANDOFF.md` is an index. Measurements come from the probe note, the status note, the GPU note, the oracle note, and the logs those notes name.

## What is already known

Frozen LeWM encodes a Push-T frame to a 192-dimensional vector. A ridge reads block angle from that vector to about 12–15° on held-out frames, including scripted 180° frames, so the encoder stays frozen. One constant latent step per command string explains about 1% of `Δz` (residual ratio 0.991).

The planner that z-scores actions with the training-column mean and standard deviation reaches a dataset goal image on 45 of 50 episodes. The same cost with a guessed scale of 5 reaches 0 of 50. Those two numbers are different experiments. The success bit is the environment's `eval_state`: the four position coordinates within 20 and the angle within 20°. It is not the 95% coverage attribute on the class.

Thirty-five of those 45 successes finish inside the first 25 environment steps, which is one open-loop plan. All five misses run to step 50 and still fail the position gate. Their position errors are 42.7, 66.4, 37.0, 24.7, and 23.4 in the write-up, and the log has the unrounded values.

An in-place rendered goal, same block position and a true `+90°` or `+180°` turn, is not executed: reach 0.00 at budget 50, and 0.00 again for `+180°` at budget 100. The angle-probe cost's 0.10 reach on `+90°` is two episodes that slid the block by 115 px and 130 px. Under the in-place gate used by the oracle, that rate is 0/20.

Maes et al. 2026 (arXiv:2603.19312) already plan with the same latent cost and already probe block angle, block location, and agent location. Their correlations are not our degree errors and do not replace the 45/50. Caselli et al. 2026 (arXiv:2607.12547) already add hierarchy on a frozen LeWM. Their flat planner, after a sweep, is 94.0% / 52.7% / 18.0% at goal offsets 25 / 50 / 75. Their 94.0% is not our 45/50. Their oracle subgoals are future frames from a demonstration (73.3% at offset 50). That result does not say an in-place half-turn is executable. GRIF aligns language with an image change and then trains a policy. Our constant vector is a special case that already failed. GRIF itself was not run.

## Short fact table

| id | result | status |
|---|---|---|
| F2 | Ridge angle MAE 12.08° at 180°, 14.58° near 0 | measured |
| F3 | Constant `Δz` ratio 0.991 | measured |
| F4 | Z-scored image-goal CEM 45/50 | measured |
| F5 | ×5 image-goal CEM 0/50, invalid | measured |
| F7 | In-place rendered goal, reach 0.00 | measured |
| F9 | Success means position L2 `< 20` and angle `< 20°` | measured |
| F11 | 35/45 successes finish by step 25 | measured |
| F12 | Probe-cost "hits" moved 115 px and 130 px | measured |
| C3 | Paper 94% at offset 25 is a different aggregate from F4 | conflict |
| C4 | Dataset-future oracle goals are a different set from F7 | conflict |

The full table with paths is `research_os/lab/facts.md`.

## Bets that survived

Nine bets survived. The falsifier is the number that kills the claim.

| bet | claim in one line | falsifier | machine |
|---|---|---|---|
| B1 | Offset 50 with the fixed official planner falls to ≤ 8/20 | ≥ 14/20 | 4090, needs Lance |
| B3 | Three rendered `+30°` stages reach `+90°` in place on ≥ 8/20 | ≤ 1/20 | 4090 |
| B4 | The chosen in-place plan is still > 60° off in imagination | median imagined error `< 20°` while execution stays `> 60°` | 4090 |
| B5 | Block xy is a ridge on the saved `z`, MAE `< 20` px | either coordinate MAE `≥ 40` px | Xeon |
| B6 | One matrix `R` predicts a same-xy `+90°` step, ratio `≤ 0.70` | ratio `≥ 0.90` | Xeon |
| B7 | A linear map of `(z_t, command)` predicts `Δz`, ratio `≤ 0.70` | ratio `≥ 0.90` | Xeon |
| B9 | The live reset goal, under the measured z-score, succeeds on ≥ 12/20 | ≤ 4/20 | 4090 |
| B10 | The 440-frame table contains a neighbor within 15 px and 15° | median neighbor `> 30` px or `> 20°` | Xeon |
| B11 | One frame does not reveal agent velocity, `R² < 0.20` | `R² > 0.50` on pose-matched pairs | Xeon, new encodes |

B9 is the one bet that points a command at a goal image the env already draws. B10 only asks whether a cache has a nearby pose. The other seven are planner or representation claims. B7 is a language port into `Δz`. It is not a lookup of a goal image, and it is not the constant vector that already failed.

## What was killed, and what was only parked

No proposed bet was killed. K1–K5 stay closed.

K6 is new. Do not cite the official probe-cost reach of 0.10 as evidence that an in-place quarter-turn sometimes works. The two counted successes translated the block by 114.95 px and 130.42 px (`reports/gpu_4090/logs/probe_cost_official.json`).

Two bets were cut as second projects. B2, constraining CEM to expert action chunks for an in-place half-turn, repeats the shape of Caselli et al. §3.3 and needs the Lance action column. B8, a five-step prediction-error curve, needs that same table and is the accumulation claim Maes et al. already state in prose. Neither is authorized to download data.

## The single next action

Write and run `scripts/latent_maps.py` on this Xeon, as specified in protocol P5–P7 and P10. It fits four maps on `data/probes/pusht_theta_probe.npz`: block xy, the rotation matrix, the state-dependent command map, and the nearest-pose catalog. Cap it at 2 hours. It does not need a GPU, a rental, or an encode.

Leave the 4090 protocols (B1, B3, B4, B9) and the velocity collection (B11) unstarted until that script has written its four numbers. A rental is not a next step while those CPU falsifiers are still open.

Cycle 002 does not start from this file.
