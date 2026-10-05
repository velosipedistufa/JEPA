# MMAGAIProject

Course workspace for a frozen [LeWorldModel](https://github.com/lucas-maes/le-wm) on Push-T, plus notes on whether language can name a goal in that latent.

The rented GPU machine is gone. Do not SSH to it. Do not copy `data/datasets/` or any Lance table onto this Xeon.

## Layout

| Path | What it is |
|---|---|
| `le-wm/` | Official LeWM source. |
| `.venv/` | Local CPU environment. Do not copy this directory to a GPU machine. |
| `data/hf_pusht/` | Checkpoint `weights.pt` and `config.json`. |
| `data/checkpoints/models--quentinll--lewm-pusht` | Symlink the loader expects. |
| `data/probes/` | CPU angle-probe table. |
| `scripts/sanity_encode.py` | One-frame CPU encode. |
| `scripts/probe_theta.py` | CPU ridge probe of block angle. |
| `scripts/eval_official_cem.py` | Image-goal CEM that scored 45/50. GPU only. |
| `scripts/oracle_rotate_cem.py` | In-place rendered-goal CEM. Reach rate was 0. |
| `SETUP.md` | How this Xeon was installed. |
| `reports/` | Notes, figures, and `reports/gpu_4090/` logs. |

## Load the checkpoint

Use Python 3.10 or 3.11. Keep `transformers>=4.50,<5`. Transformers 5 renames the ViT weights and `weights.pt` will not load.

```bash
export STABLEWM_HOME="$HOME/GitRepos/MMAGAIProject/data"
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export MUJOCO_GL=egl
```

```python
from stable_worldmodel.wm.utils import load_pretrained
model = load_pretrained("quentinll/lewm-pusht")
model.eval()
```

On this Xeon, `torch.cuda.is_available()` is false. That is expected. Build a new virtualenv on any GPU box and install a CUDA wheel that matches `nvidia-smi`. Do not copy `.venv/`.

Pixels are uint8 HWC, scaled to `[0, 1]`, ImageNet mean and standard deviation, resized to 224, then a time axis so the tensor is `(B, 1, 3, 224, 224)`. `model.encode({"pixels": pixels})["emb"]` has shape `(B, T, 192)`.

Actions in the working CEM are z-scored with the Push-T column mean `[-0.00781, 0.00686]` and unbiased standard deviation `[0.20847, 0.20675]`, then inverse-transformed and clipped to `[-1, 1]` before `env.step`. Do not multiply actions by 5.

## What already ran

The encoder stays frozen. A CPU ridge reads angle at 180 degrees about as well as near 0 degrees (circular MAE 12.08 degrees versus 14.58 degrees). A constant latent step per text command does not fit (residual ratio 0.991). Official image-goal CEM is `scripts/eval_official_cem.py` and scored 45/50. An in-place rendered goal at +90 degrees or +180 degrees was not reached (reach rate 0.00). Details are in `reports/lewm_mmagai_status.tex`.
