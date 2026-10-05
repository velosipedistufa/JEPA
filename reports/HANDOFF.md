# Handoff

Index for a new session. Numbers are copied from the files named next to them. This file does not add a research plan.

## Project goal

Frozen LeWM (`quentinll/lewm-pusht`) is the world model.
Language is a port into the latent `z`, not a second dynamics model.
No LLM sits inside the JEPA.
No text-to-image goal generator.

## Measured

Ridge, held-out circular MAE, from `reports/lewm_language_port_probe.tex` (same figures in `reports/README.md` and `reports/lewm_mmagai_status.tex`): rot180 **12.08°** (`n=20`); frames with `|θ|<30°` **14.58°** (`n=19`). Overall test MAE in that note is 13.75° (`n=110`). The note's decision sentence is: do not finetune LeWM.

Constant `Δz` per command, from `reports/lewm_language_port_probe.tex`: train RMSE of `Δz` divided by the RMS of `Δz` is **0.991**. `reports/README.md` states the same ratio. `put T on the target` was not fit.

Official image-goal CEM, from `reports/gpu_4090/GPU_RUN.md` (section "Official protocol"): **45/50 (90%)**. Mean position L2 17.47. Goal is 25 steps ahead, budget 50, seed 42. Log cited there: `reports/gpu_4090/logs/baseline_official.json`. `reports/lewm_mmagai_status.tex` states 45/50 as well.

Guessed ×5 CEM, from `reports/gpu_4090/GPU_RUN.md` (section "Baseline image-goal CEM"): **0/50**, marked there as not a working baseline. `reports/lewm_mmagai_status.tex` calls this run invalid. Do not use it.

Probe-cost rotate reach, prefer the official-protocol table in `reports/gpu_4090/GPU_RUN.md`: **0.10** for "rotate T 90" and **0.00** for "rotate T 180" (`n=20` each). `reports/lewm_mmagai_status.tex` matches those two rates. An earlier table in the same `GPU_RUN.md` (section "Probe-cost CEM", before the official z-score) reports **0.15 / 0.00**. That earlier table is the pre-official planner. Use 0.10 / 0.00.

Oracle rendered `z_g` for in-place `+90°` / `+180°`, from `reports/gpu_4090/GPU_RUN.md` (section "Oracle in-place rotation"): reach **0.00** at budget 50 for both turns, and **0.00** for `+180°` at budget 100 (`n=20`). `reports/lewm_oracle_rotate_cem.tex` and `reports/lewm_mmagai_status.tex` agree. Mean `||z_g-z_t||` in `GPU_RUN.md` is 15.69 (`+90°`) and 18.33 (`+180°`), so the goal frame was not a copy of the start.

Action mean/std, prefer `reports/gpu_4090/GPU_RUN.md`: mean `[-0.007813, 0.006861]`, unbiased std `[0.208467, 0.206749]`, 2,336,736 rows. `reports/lewm_mmagai_status.tex` rounds the same pair to `[-0.00781, 0.00686]` and `[0.20847, 0.20675]`.

CEM settings for the official and oracle runs, from `reports/gpu_4090/GPU_RUN.md` and the constants in `scripts/eval_official_cem.py`: 300 samples, 30 steps, top 30, horizon 5, receding horizon 5, action block 5, pixel history 1. Actions are z-scored with the mean and std above, then inverse-transformed and clipped to `[-1, 1]` before `env.step`.

## Locked decisions

Do not finetune the encoder. Source: `reports/lewm_language_port_probe.tex` (rot180 MAE is not worse than the near-0 slice) and `reports/lewm_mmagai_status.tex`.

Do not treat in-place 180° as a language failure. Source: `reports/gpu_4090/GPU_RUN.md` oracle section. Reach stays 0.00 when `z_g` is the encoding of a real rendered frame, so the miss is in the planner under this horizon and budget. `reports/lewm_oracle_rotate_cem.tex` says the same.

The official planner is `scripts/eval_official_cem.py`. `scripts/gpu_run.py` is the invalid ×5 run. `scripts/oracle_rotate_cem.py` calls the official planner and does not define a second action scale.

## File map

Scripts: `scripts/sanity_encode.py` (one CPU frame), `scripts/probe_theta.py` (ridge probe), `scripts/language_port.py` (constant `Δz`), `scripts/make_lance_optional.py` (Xeon only; AVX2/BMI2 wheels), `scripts/eval_official_cem.py` (image-goal CEM), `scripts/oracle_rotate_cem.py` (rendered in-place goal), `scripts/gpu_run.py` (do not use as the baseline).

Probe table: `data/probes/pusht_theta_probe.npz` and `data/probes/pusht_theta_probe.csv`. Metrics: `data/probes/probe_metrics.json`. Language-port fit: `data/probes/language_port_fit.json`.

Reports: `reports/lewm_language_port_probe.tex` (CPU ridge and constant `Δz`), `reports/lewm_mmagai_status.tex` (status for the course), `reports/lewm_language_port_probe_gpu.tex` (first 4090 addendum, includes the invalid ×5 baseline), `reports/lewm_oracle_rotate_cem.tex` (oracle rotation), `reports/README.md` (how to rerun the CPU probe), `reports/refs.bib`.

4090 index: `reports/gpu_4090/GPU_RUN.md`. Raw logs next to it under `reports/gpu_4090/logs/`, including `baseline_official.json`, `probe_cost_official.json`, `oracle_rotate.json`, and `pred_zg_cem.json` (MLP skipped).

Checkpoint loader: `load_pretrained("quentinll/lewm-pusht")` with `STABLEWM_HOME` pointed at `data/`. Pin `transformers>=4.50,<5` (`SETUP.md`, root `README.md`).

## Open question

Rank course bets that use these results.

Forbidden as the main bet: another constant-`Δz` head, CLIP for its own sake, and finetuning LeWM to add spin.
