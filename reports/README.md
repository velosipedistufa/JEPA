# Push-T angle probe and language-port stub

CPU-only follow-up to `SETUP.md`.
LeWM is frozen.
CEM / `eval.py` was not run.
The Lance and HDF5 training sets were not downloaded.

## Run

From `~/GitRepos/MMAGAIProject`, with the existing venv:

```bash
source .venv/bin/activate
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export MUJOCO_GL=egl
export STABLEWM_HOME="$HOME/GitRepos/MMAGAIProject/data"
export HF_HOME="$HOME/GitRepos/MMAGAIProject/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/GitRepos/MMAGAIProject/.cache"
export WANDB_MODE=disabled

python -u scripts/probe_theta.py
python -u scripts/language_port.py
```

`probe_theta.py` loads `quentinll/lewm-pusht` through `stable_worldmodel.wm.utils.load_pretrained`, renders `swm/PushT-v1` with `render_mode="rgb_array"`, and encodes with the same ImageNet preprocessing as `scripts/sanity_encode.py`.
Keep `transformers` at `>=4.50,<5`.
If those packages are reinstalled, run `scripts/make_lance_optional.py` before importing them on this Xeon.

## Outputs

| Path | Contents |
|---|---|
| `data/probes/pusht_theta_probe.npz` | `z` `(440, 192)` plus pose columns |
| `data/probes/pusht_theta_probe.csv` | same rows without `z` |
| `data/probes/reset_api.json` | one seed-0 dump of obs / info |
| `data/probes/probe_metrics.json` | split sizes, MAE, decision |
| `data/probes/language_port_fit.json` | least-squares residual of `f(text)` |
| `data/probes/language_port_W.npz` | one-hot map `W` |
| `reports/figures/*.pdf` | held-out probe plots |
| `reports/lewm_language_port_probe.tex` | note |

Collection: 80 episodes × (random + 90° + 180° + −90°) and 120 random-only episodes.
Split is by episode, 25% test, seed 0 (150 / 50 episodes, 330 / 110 frames).
One encode took 0.123 s.

## Held-out circular MAE (degrees)

Ridge, `alpha=1`, target `(cos θ, sin θ)`:

| slice | n | MAE |
|---|---:|---:|
| all test | 110 | 13.75 |
| random | 50 | 12.36 |
| rot90 | 20 | 16.17 |
| rot180 | 20 | 12.08 |
| rot−90 | 20 | 16.50 |
| \|θ\| < 30° from angle 0 | 19 | 14.58 |

OLS test MAE is 18.88° overall and 20.06° on rot180.
Both probes fall on the same side of the rule.

## Decision

Do not finetune LeWM; only train the language port.

Rot180 ridge MAE is 12.08°, which is below the near-zero MAE (14.58°) and below 30°.
The constant displacement `f(text)` does not explain `Δz` (train RMSE / target RMS = 0.991).
`put T on the target` was not measured.
CLIP classes import; CLIP weights were not downloaded.
