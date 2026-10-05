# RTX 4090 run (Clore, port 2394)

Encoder stayed frozen. No `train.py`. No goal images from text. `eval.py` was not used: it reads `pusht_expert_train` from the HDF5/Lance sets, which were not downloaded.

## CUDA sanity

From `logs/sanity_gpu.txt` (rewritten at the start of the calibrated run):

```
torch 2.5.1+cu121
cuda True
gpu NVIDIA GeForce RTX 4090
torch.version.cuda 12.1
encode_s 0.068280
z_shape (1, 1, 192)
z_mean -0.039798151701688766
z_std 0.97542405128479
z_min -2.342271566390991
z_max 2.6890833377838135
```

`nvidia-smi` on the box: driver 535.309.01, CUDA 12.2, 24564 MiB. The image Python was 3.8 with torch 2.0.1+cu117. The job used a new venv, CPython 3.11.16, and a cu121 wheel. cu124/cu128 was not installed: driver 535 does not support those runtimes.

## What failed

- First SSH attempts to this port reset before the banner. A later login succeeded.
- `eval.py` cannot start a live episode without the expert dataset. The baseline is a live CEM in `scripts/gpu_run.py`.
- An uncalibrated pass (model actions equal to env actions in `[-1, 1]`) scored 0/10. Predicted latents barely moved. Eight live frameskip checks preferred dividing env actions by 0.2 before the action encoder (one-step prediction error 3.64 versus a real latent change of 9.03). The numbers below use that factor of 5.
- Dataset action z-score statistics were not available. The factor 5 is a global scale, not the training normalizer.
- The energy network in step 4 was not trained.

## Baseline image-goal CEM

50 live `swm/PushT-v1` episodes, `render_mode=rgb_array`, goal image `info["goal"]` from `reset`. Cost is `||z_H - z_goal||^2`. CEM: 300 samples, 30 steps, top-30 elites, horizon 5, action block 5, budget 50 env steps.

| episodes | successes | mean position L2 | mean angle error | wall time |
|---:|---:|---:|---:|---:|
| 10 | 0 | (see `baseline_cem_n10.json`) | | 8.16 s |
| 20 | 0 | | | 16.17 s |
| 50 | 0 | 300.07 | 77.64° | 40.55 s |

Best of the 50: position L2 28.95 (threshold 20) and a separate episode with angle error 3.98°. Success rate 0. This does not match a working Push-T CEM. The missing per-dimension action normalizer is the likely cause.

## Probe-cost CEM

Ridge `z → (cos θ, sin θ)` and `z → (x, y)` refit on the CPU probe's train episodes (seed 0, 25% episode holdout). Cost on imagined `z_H`: wrapped angle error to `θ_now + Δθ`, or xy error to `(256, 256)` for "put T on the target". Same 20 held-out scripted starts. Reach means angle error under 20° (and, for the put command, xy under 20 px and angle under 20° of `π/4`).

| command | n | mean \|angle error\| | mean xy error | reach rate |
|---|---:|---:|---:|---:|
| rotate T 90 | 20 | 78.38° | 120.43 | 0.15 |
| rotate T 180 | 20 | 119.70° | 108.37 | 0.00 |
| rotate T -90 | 20 | 91.83° | 95.24 | 0.00 |
| put T on the target | 20 | 73.23° | 86.44 | 0.00 |

Wall time 64.68 s. A constant `Δz` per command was not refit; the CPU ratio 0.991 still stands.

## Skill head

MLP `f(z_t, one-hot) → Δz`, hidden size 256, 400 Adam steps. Training pairs: 180 from the CPU train split plus 600 new same-xy turns collected on this GPU (200 episodes × 3). Final train batch MSE 0.283. Held-out pair MSE 0.762. Held-out mean L2 of the residual is 11.68 against a target mean L2 of 16.28, so the head beats a constant vector and still leaves most of the displacement unexplained.

CEM then minimizes `||(z_H - z_t) - f(z_t, e)||^2` on the same 20 starts. "put T on the target" was not measured.

| command | n | mean \|angle error\| | median | mean xy error | reach rate |
|---|---:|---:|---:|---:|---:|
| rotate T 90 | 20 | 90.02° | 90.00° | 12.28 | 0.00 |
| rotate T 180 | 20 | 169.97° | 180.00° | 27.29 | 0.00 |
| rotate T -90 | 20 | 85.44° | 90.00° | 16.18 | 0.05 |

Wall time 48.51 s. Median errors sit on the commanded offset, which means the block usually did not turn. The probe cost turns the block more often (0.15 reach on +90°) and disturbs xy more.

Weights: `heads/skill_mlp.pt` and `data/heads/skill_mlp.pt`.

## Commands

Password is not stored in this tree. The host was `root@n1.msk.cloreai.ru` port 2394.

```bash
rsync -a -e 'ssh -p 2394' le-wm scripts SETUP.md reports root@n1.msk.cloreai.ru:/root/MMAGAIProject/
rsync -a -e 'ssh -p 2394' data/hf_pusht data/probes root@n1.msk.cloreai.ru:/root/MMAGAIProject/data/
# remote
ln -sfn ../hf_pusht /root/MMAGAIProject/data/checkpoints/models--quentinll--lewm-pusht
# musl uv 0.12.18, CPython 3.11.16
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install 'stable-worldmodel[train]==0.1.1' 'transformers>=4.50,<5' \
  pygame pymunk shapely opencv-python-headless gymnasium imageio scikit-learn matplotlib
# make_lance_optional.py was not run
cd /root/MMAGAIProject && .venv/bin/python -u scripts/gpu_run.py all
```

`scripts/gpu_run.py` is the live CEM. `WANDB_MODE=disabled`. `STABLEWM_HOME=/root/MMAGAIProject/data`.

## Official protocol (this session)

The factor-of-5 planner is not the baseline. `scripts/eval_official_cem.py` z-scores actions with the Lance column stats that `get_column_normalizer` computes, then inverse-transforms and clips to `[-1, 1]` before `env.step`.

Dataset, on the 4090 only: `galilai-group/lewm-pusht` → `pusht_expert_train.lance` (about 14 GB). Not copied to the Xeon. Not the 43 GB HDF5.

Action column, 2,336,736 rows: mean `[-0.007813, 0.006861]`, unbiased std `[0.208467, 0.206749]`. Population std differs only in the 8th digit.

One dataset macro-action (5 raw steps), normalized and rolled once: real `||Δz||` 8.98, imagined `||Δz||` 8.70, `||z_pred - z_real||` 1.87. The earlier 9.03 vs 3.64 gap is gone.

Init is `_set_state` / `_set_goal_state` from dataset rows. Goal image is the frame `goal_offset_steps=25` ahead. Budget 50. CEM 300/30/top 30, horizon 5, receding horizon 5, action block 5. Pixel history length is 1, which is `PlanConfig`'s default in `le-wm/eval.py`. The checkpoint predictor `num_frames` is 3 and is the attention window inside `rollout`, not a stack of 3 rendered frames. Seed 42.

| episodes | successes | rate | mean position L2 | mean full-state distance |
|---:|---:|---:|---:|---:|
| 10 | 10 | 100% | | |
| 50 | 45 | 90% | 17.47 | 121.09 |

The five misses end at position L2 42.7, 66.4, 37.0, 24.7, and 23.4. File: `logs/baseline_official.json`.

Probe cost was run only after that 90% result, on the same CEM and the same z-score, 20 held-out scripted starts, rotate 90 and 180 only. The skill MLP was not retrained.

| command | n | mean \|angle error\| | reach rate | mean xy error |
|---|---:|---:|---:|---:|
| rotate T 90 | 20 | 75.44° | 0.10 | 58.97 |
| rotate T 180 | 20 | 159.15° | 0.00 | 35.89 |

File: `logs/probe_cost_official.json`. Image-goal CEM is the working controller. The angle-probe cost is not, on this budget.

## Oracle in-place rotation

`scripts/oracle_rotate_cem.py` uses the same planner as `scripts/eval_official_cem.py`: dataset z-score, inverse-transform, clip to `[-1, 1]`, CEM 300/30/top 30, horizon 5, action block 5, pixel history 1. No factor of 5.

Twenty held-out starts from `pusht_theta_probe.npz`, split seed 0. For each start the block is rendered at the same xy with angle `+90°` or `+180°`. That frame is `z_g`. CEM minimizes `||z_H - z_g||^2`. Reach means wrapped angle error below 20 degrees and block xy drift below 15 pixels. The rendered goal itself is a true rotation: mean goal-render angle error is 0.023 degrees at +90° and 0.065 degrees at +180°, with xy drift 0.19 and 0.12 pixels. Mean `||z_g - z_t||` is 15.69 at +90° and 18.33 at +180°, so the target latent is not the start latent.

| command | budget | n | mean \|dθ\| | mean dxy | reach | wall s |
|---|---:|---:|---:|---:|---:|---:|
| rotate 90 | 50 | 20 | 91.94° | 27.76 | 0.00 | 15.56 |
| rotate 180 | 50 | 20 | 153.05° | 50.46 | 0.00 | 15.48 |
| rotate 180 | 100 | 20 | 151.60° | 53.39 | 0.00 | 30.97 |

No coverage field is returned by Push-T info. `coverage_arr` length is logged per row when the attribute exists.

Decision: CEM cannot execute in-place 180° even with a true goal image under this horizon/budget.

The `z_g` MLP was not trained. `logs/pred_zg_cem.json` records that skip. Official image-goal success on the 25-step dataset protocol remains 45/50.
