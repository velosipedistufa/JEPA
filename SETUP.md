# LeWorldModel CPU setup (this machine)

CPU-only install of [le-wm](https://github.com/lucas-maes/le-wm) and `stable-worldmodel` 0.1.1. No training, no `eval.py`, no RX 580.

## Machine

| | |
|---|---|
| OS | Arch Linux, kernel 7.2.6-arch2-1, Wayland (dwl). No X11 assumed. |
| CPU | Intel Xeon E3-1270 v2 (Ivy Bridge). AVX yes. AVX2 and BMI2 no. |
| RAM | 16 GB |
| GPU | RX 580 ignored. `torch.cuda.is_available()` is `False`. |
| Python | 3.10.21 (uv standalone CPython). System Python is 3.14 and is not used. |
| torch | `2.14.0+cpu` / torchvision `0.29.0+cpu`. `torch.version.cuda` is `None`. |
| stable-worldmodel | 0.1.1 |
| stable-pretraining | 0.1.8 |
| transformers | **4.57.6** (must stay `<5`; see below) |

`libEGL` is installed. `libOSMesa` is not. MuJoCo was not installed. `MUJOCO_GL=egl` is set so a later import does not try to open a window; Push-T does not use it.

## Layout

```
~/GitRepos/MMAGAIProject/.tools/uv          # uv 0.12.18
~/GitRepos/MMAGAIProject/.tools/python      # CPython 3.10.21
~/GitRepos/MMAGAIProject/.venv
~/GitRepos/MMAGAIProject/le-wm              # https://github.com/lucas-maes/le-wm
~/GitRepos/MMAGAIProject/data               # STABLEWM_HOME
~/GitRepos/MMAGAIProject/data/hf_pusht      # weights.pt + config.json
~/GitRepos/MMAGAIProject/scripts/sanity_encode.py
```

`pacman -S uv` failed on this mirror (`uv-0.12.17-1` signature 404). The binary above is the official uv installer, kept inside the project. Arch has no `python3.10` package.

## Env vars

```bash
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export MUJOCO_GL=egl
export STABLEWM_HOME="$HOME/GitRepos/MMAGAIProject/data"
export HF_HOME="$HOME/GitRepos/MMAGAIProject/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$HOME/GitRepos/MMAGAIProject/.cache"
export UV_PYTHON_INSTALL_DIR="$HOME/GitRepos/MMAGAIProject/.tools/python"
export UV_CACHE_DIR="$HOME/GitRepos/MMAGAIProject/.tools/uv-cache"
export UV_TORCH_BACKEND=cpu
```

`scripts/sanity_encode.py` sets the graphics vars and `STABLEWM_HOME` itself. Do not set `SWM_ENABLE_LANCE=1` on this CPU.

## Install commands

From `~/GitRepos/MMAGAIProject`:

```bash
mkdir -p .tools
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh

export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_CACHE_DIR="$PWD/.tools/uv-cache"
export UV_TORCH_BACKEND=cpu

./.tools/uv python install 3.10
./.tools/uv venv --python 3.10 .venv

./.tools/uv pip install --python .venv/bin/python --torch-backend cpu \
  torch torchvision --index-url https://download.pytorch.org/whl/cpu

./.tools/uv pip install --python .venv/bin/python --torch-backend cpu \
  'stable-worldmodel[train]==0.1.1' \
  pygame pymunk shapely opencv-python-headless gymnasium imageio

# Checkpoint was saved with transformers 4 ViT names
# (encoder.encoder.layer.*.attention.attention.query).
# transformers 5.17 renames those to encoder.layers.*.attention.q_proj
# and load_state_dict fails. Pin 4.x.
./.tools/uv pip install --python .venv/bin/python --torch-backend cpu \
  'transformers>=4.50,<5'

./.venv/bin/python scripts/make_lance_optional.py
```

`[env]` was not installed. It pulls `gymnasium[all]`, Box2D, MuJoCo, OGBench, Craftax, and ALE. Push-T only needs pygame, pymunk, shapely, gymnasium, and `cv2`. `opencv-python-headless` is the wheel the env extra lists and does not open a Qt window. `imageio` is imported by `stable_worldmodel.wrapper` at package import.

After every reinstall of those packages, run `scripts/make_lance_optional.py` again. It edits the installed copies of:

- `stable_worldmodel/data/__init__.py` and `data/formats/__init__.py` — skip `import lance`
- `stable_pretraining/data/__init__.py` — skip Lance video/image modules
- `kornia/io/io.py` — defer `import kornia_rs`

`pylance` 12.0.0 runs AVX2 `vpermq` while loading and dies with SIGILL. `kornia_rs` 0.1.14 runs BMI2 `rorx` and dies the same way. SIGILL is not an `ImportError`, so a try/except cannot catch it. Push-T and the LeWM encoder do not need either library. `import stable_worldmodel`, `import stable_pretraining`, and `import numpy` succeed after the patch. `torch.cuda.is_available()` is `False`.

## Checkpoint

```bash
export HF_HOME="$PWD/.cache/huggingface"
export STABLEWM_HOME="$PWD/data"
hf download quentinll/lewm-pusht --local-dir "$STABLEWM_HOME/hf_pusht"
mkdir -p "$STABLEWM_HOME/checkpoints"
ln -sfn ../hf_pusht "$STABLEWM_HOME/checkpoints/models--quentinll--lewm-pusht"
```

Downloaded files: `config.json` (1313 bytes) and `weights.pt` (72290721 bytes, about 69 MB). No Hugging Face token.

Load API that works (same one `le-wm/eval.py` calls):

```python
from stable_worldmodel.wm.utils import load_pretrained
model = load_pretrained("quentinll/lewm-pusht")  # reads $STABLEWM_HOME/checkpoints/
model.eval()
info = model.encode({"pixels": pixels})  # pixels: (B, T, 3, 224, 224) float
z = info["emb"]                           # (B, T, 192)
```

`pixels` follow `le-wm/eval.py`: uint8 HWC, `ToDtype(scale=True)`, ImageNet mean/std, resize 224, then a time axis of 1.

The README conversion to `pusht/lewm_object.ckpt` does not match this `config.json`. `ARPredictor(**cfg["predictor"])` raises:

```
TypeError: ARPredictor.__init__() got an unexpected keyword argument '_target_'
```

`config.json` is Hydra instantiate format for `stable_worldmodel.wm.lewm.LeWM`, which `load_pretrained` already builds. No `_object.ckpt` was written.

`swm.policy.AutoCostModel("pusht/lewm")` looks for `$STABLEWM_HOME/checkpoints/pusht/lewm_object.ckpt` and raises:

```
AssertionError: Checkpoint path does not exist: .../checkpoints/pusht/lewm_object.ckpt
```

## Push-T env

```python
env = gym.make("swm/PushT-v1", render_mode="rgb_array")
env.reset(seed=0)
frame = env.render()  # (224, 224, 3) uint8
env.close()
```

This ran under `SDL_VIDEODRIVER=dummy`. `render_mode="human"` was not used. No window.

## Dataset (not downloaded)

| Name the code uses | Where it is | Size |
|---|---|---|
| `pusht_expert_train.lance` | `le-wm/config/train/data/pusht.yaml`, and the Lance repo `galilai-group/lewm-pusht` | about 13.3 GB |
| `pusht_expert_train.h5` | README text. HF dataset `quentinll/lewm-pusht` ships `pusht_expert_train.h5.zst` only | 12.2 GB compressed; the stable-worldmodel README lists the unpacked HDF5 at 43.12 GB |

One encode does not read the dataset. The only train file on the HDF5 repo is the full archive, so nothing was unpacked.

## Sanity output

```bash
.venv/bin/python -u scripts/sanity_encode.py
```

```
torch 2.14.0+cpu
cuda_available False
torch.version.cuda None
STABLEWM_HOME /home/alex/GitRepos/MMAGAIProject/data
frame_source pusht_rgb_array
frame_shape (224, 224, 3) uint8
encoder_input (1, 1, 3, 224, 224) torch.float32
z_shape (1, 1, 192)
z_mean -0.039807215332984924
z_std 0.9753866195678711
z_min -2.3422999382019043
z_max 2.689063310623169
z_first8 [0.402373, 0.877979, 0.820278, -0.322866, -1.091935, 0.009626, -1.125726, 0.99749]
```

Disk after setup: about 2.2 GB venv, 266 MB `.tools`, 70 MB checkpoint. About 52 GB free on `/`.

## Later, on a 24 GB NVIDIA box

- Skip `scripts/make_lance_optional.py`. That patch is for this Ivy Bridge CPU. A current cloud CPU has AVX2 and BMI2, and Lance is what `pusht_expert_train.lance` needs.
- Install the CUDA wheel that matches the driver (`UV_TORCH_BACKEND=cu128` or whatever the box ships). Confirm `torch.cuda.is_available()` is `True` before training. This tree still has the CPU wheel.
- Keep `transformers>=4.50,<5` until the checkpoint is re-exported. transformers 5 will not load `weights.pt`.
- `le-wm/eval.py` calls `model.to("cuda")` and then full planning. It was not run.
- Download `galilai-group/lewm-pusht` (`pusht_expert_train.lance`, ~13 GB) rather than unpacking the 43 GB HDF5. Set `STABLEWM_HOME` to a disk that can hold it.
- `config/train/lewm.yaml` still has a placeholder WandB entity. No WandB key is on this machine. `WANDB_MODE=disabled` is enough until a key exists.
- Host RAM on this PC is 16 GB. One forward of the 192-d encoder fits. Training does not belong here.
