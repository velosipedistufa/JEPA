# Cycle 001 — Librarian

No new experiment was run. This file records what the reports, logs, scripts, and the Push-T environment source already say. `reports/HANDOFF.md` was read as an index. Its numbers are not used as measurements. The lowercase path `reports/handoff.md` is absent.

## Fact table

The rows below are the lab table. Paper claims are in the next section and are not facts of this repo.

| id | statement | source | status |
|---|---|---|---|
| F1 | Frozen `quentinll/lewm-pusht` encodes one frame to `z` of shape `(1, 1, 192)`. CPU 0.123 s; 4090 sanity 0.068280 s. | probe tex §Methods; `reports/gpu_4090/GPU_RUN.md`; `scripts/sanity_encode.py` | measured |
| F2 | Ridge circular MAE 12.08° on `rot180` (`n = 20`), 14.58° within 30° of angle 0 (`n = 19`), 13.75° overall (`n = 110`). Decision: do not finetune. OLS agrees (18.88° / 20.06° / 26.09°). JSON holds the unrounded ridge values. | `reports/lewm_language_port_probe.tex`; `reports/README.md`; `data/probes/probe_metrics.json` | measured |
| F3 | Constant `Δz` train residual ratio 0.9914314089199128. `put T on the target` was not fit. | `data/probes/language_port_fit.json`; probe tex | measured |
| F4 | Z-scored image-goal CEM: 45/50, mean position L2 17.47269057271119, mean state distance 121.086546360662. | `reports/gpu_4090/logs/baseline_official.json`; `GPU_RUN.md`; status tex | measured |
| F5 | ×5 image-goal CEM: 0/50, mean position L2 300.07, mean angle error 77.64°. Invalid. | `GPU_RUN.md`; GPU addendum tex; status tex | measured |
| F6 | Official-z-score probe cost, angle-only reach: 0.10 at `+90°`, 0.00 at `180°` (`n = 20`). | `logs/probe_cost_official.json`; `scripts/eval_official_cem.py` | measured |
| F7 | In-place rendered `z_g`: reach 0.00 at budget 50 for `+90°` and `+180°`, and 0.00 for `+180°` at budget 100. `z_g` MLP skipped. | `GPU_RUN.md`; `reports/lewm_oracle_rotate_cem.tex`; `logs/pred_zg_cem.json` | measured |
| F8 | This Xeon has no AVX2 or BMI2. Lance and `kornia_rs` SIGILL. RX 580 unused. Push-T is headless. | `SETUP.md` | measured |
| F9 | `eval_state` succeeds when the 4-D position L2 is `< 20` and the wrapped angle gap is `< π/9`. Coverage threshold 0.95 is a different attribute and is not the official CEM predicate. | `stable_worldmodel/envs/pusht/env.py`; `scripts/eval_official_cem.py` | measured |
| F10 | Five official misses, all `steps = 50`, all position L2 above 20: 42.743, 66.389, 36.983, 24.704, 23.432 on episodes (7572, 117), (8373, 102), (10117, 136), (15701, 34), (16400, 0). Terminal angle is not in the log. | `logs/baseline_official.json`; `GPU_RUN.md` | measured |
| F11 | 35 of 45 official successes have `steps ≤ 25`. 10 have `steps ≥ 26`. Step counts are listed in `lab/facts.md`. | `logs/baseline_official.json` | measured |
| F12 | The two `+90°` probe-cost angle hits (episodes 52 and 74) have xy error 114.95 and 130.42 px. Angle `< 20°` and xy `< 15` px holds for 0/20 and 0/20. | `logs/probe_cost_official.json`; `scripts/oracle_rotate_cem.py` | measured |
| F13 | Action mean `[-0.007812564379916167, 0.006860687229453028]`, unbiased std `[0.2084674874515821, 0.20674867061239938]`, `n = 2336736`. | `logs/baseline_official.json` | measured |
| F14 | One logged macro-action: real `\|\|Δz\|\|` 8.984, imagined 8.703, prediction error 1.871. | `logs/baseline_official.json` | measured |
| F15 | Skill MLP, ×5 planner only: held-out residual L2 11.68 vs target L2 16.28; CEM reach 0.00 / 0.00 / 0.05. | `GPU_RUN.md`; GPU addendum; `logs/skill_head_cem.json` | measured |
| F16 | Official constants match the LeWM Push-T eval yaml and the CEM yaml. Plans are inverse-transformed with F13 and clipped to `[-1, 1]`. | `scripts/eval_official_cem.py`; `le-wm/config/eval/pusht.yaml`; `le-wm/config/eval/solver/cem.yaml` | measured |
| F17 | Scripted turns shift xy by mean 0.126 px (max 3.781). Oracle renders match the commanded angle to 0.023° and 0.065°, with `\|\|z_g − z_t\|\|` 15.69 and 18.33. | `probe_metrics.json`; probe tex; `GPU_RUN.md` | measured |
| F18 | Lance, HDF5, and CLIP weights are absent on this Xeon. The probe table has no velocity column. | `GPU_RUN.md`; `language_port_fit.json`; `scripts/probe_theta.py`; `data/` listing | measured |
| F19 | The official log records `history_len = 1` and `predictor_num_frames = 3`. The script comment says the 3 is the predictor attention window. | `logs/baseline_official.json`; `scripts/eval_official_cem.py` | measured |
| C1 | 0/50 and 45/50 are different action transforms (F5 and F4). | GPU addendum; status tex; `GPU_RUN.md` | conflict |
| C2 | Probe-cost `+90°` reach 0.15 and 0.10 belong to the ×5 planner and the z-scored planner. | GPU addendum; `GPU_RUN.md`; `probe_cost_official.json` | conflict |
| C3 | Published flat-LeWM 94.0% at offset 25 (three seeds, best sweep) is a different aggregate from local 45/50. Local row remains F4. | arXiv:2607.12547 §3.1; F4 | conflict |
| C4 | Dataset-trajectory oracle subgoals in that paper are a different goal family from F7. "Any true `z_g` is executable" disagrees with F7. | arXiv:2607.12547 §3.2; F7 | conflict |

## Literature read, not promoted

These sentences were read in the papers. They do not enter `lab/facts.md` as local measurements.

Maes, Le Lidec, Scieur, LeCun, and Balestriero, 2026, arXiv:2603.19312, LeWorldModel.

- The planning cost in equation (4) is the squared Euclidean distance between the imagined terminal latent and the encoding of a goal observation. The solver is CEM. The paper says autoregressive rollouts accumulate error as the horizon grows, and MPC executes a prefix before replanning (§3.2).
- Table 1, Push-T, linear probes: block angle MSE `0.187 ± 0.359`, correlation `0.902`; block location correlation `0.986`; agent location correlation `0.974`. MLP block-angle correlation is `0.990`. The table does not report circular degrees, a `rot180` slice, or a same-xy `Δz`.
- §5.1 says latent trajectories on Push-T become straighter during training. No straightness number is copied here. That claim is about consecutive-time velocity in latent space. It is silent on a fixed-xy rotation.
- The absolute Push-T success rate is in a figure and is not transcribed. The prose gives an 18% gap over PLDM and a `48×` planning-time claim. Neither replaces F4.

Caselli, Massafra, Punzo, Lo Sardo, Pantelidis, and Bhethanabhotla, 2026, arXiv:2607.12547, Hi-LeWM. HTML of v1.

- Flat LeWM under their protocol: 94.0% at offset 25, 52.7% at 50, 18.0% at 75. Naive high-level CEM: 89.3%, 38.7%, 15.3%. At offset 25 the best hierarchical sweep uses a high-level horizon of 1.
- Oracle-subgoal acting, true future latents from the dataset, offset 50: `73.3 ± 4.2%`.
- Constraining high-level search to macro-actions encoded from training trajectories, with staged execution: 64.0% at offset 50, which they describe as `+11.3` points over flat LeWM. Online constrained replanning: 32.7% at offset 75, `+14.7` points over their flat 18.0%.
- Their oracle goals lie on demonstration futures. F7's goals are rendered in-place rotations. C4 records that distinction.

Myers, He, Fang, Walke, Hansen-Estruch, Cheng, Jalobeanu, Kolobov, Dragan, and Levine, 2023, arXiv:2307.00117, GRIF.

- The abstract says the learned embedding aligns a language instruction with the change between the start and goal images, and a policy is trained on that embedding. GRIF is not a single frozen vector per template string. F3 and K2 falsify that narrower map. They do not by themselves falsify GRIF.

Chen et al., 2025, VL-JEPA, arXiv:2512.10942, is in `reports/refs.bib` and in the probe note's related work. It was not trained here. LeCun, 2022, is cited by the probe note as the latent-planning programme. No sentence from that PDF was re-read in this cycle, so no new quotation is added.

`reports/HANDOFF.md` says the open task is to rank course bets, and it forbids another constant-`Δz` head, CLIP for its own sake, and a finetune for spin. Those three already match K2, an unmeasured CLIP download (F18), and K4. The ranking itself is not a measurement.

## Files expected and not found

- `reports/handoff.md`. The index that exists is `reports/HANDOFF.md`.
- `data/datasets/` and `pusht_expert_train.lance` on this Xeon. `SETUP.md` describes the Lance and HDF5 archives; they are not in the tree.
- CLIP weights. `data/probes/language_port_fit.json` records that the classes import and the weights were not fetched.
- A `z_g` MLP checkpoint. `reports/gpu_4090/logs/pred_zg_cem.json` has `status: skipped`.
- Terminal angle, coverage, or latent cost inside `baseline_official.json`. The episode keys are success, state distance, terminated, steps, position L2, episode, and start.
- Agent velocity inside `data/probes/pusht_theta_probe.npz`.
- A second official CEM seed.

## What the two image-goal rates are

C1 and C2 are naming conflicts. `GPU_RUN.md` already separates the sections. The GPU addendum's conclusion that image-goal CEM scored 0/50 describes the ×5 run. The status note's 45/50 describes the z-scored run. F12 is a reading of logged columns: the published 0.10 reach is the angle-only bit the script wrote, and those two episodes are large translations.
