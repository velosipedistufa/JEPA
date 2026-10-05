#!/usr/bin/env python3
"""Load the official Push-T LeWM checkpoint and encode one 224x224 frame on CPU.

Headless on purpose: matplotlib Agg, SDL dummy, no render_mode='human'.

Requires the venv from SETUP.md: transformers<5 (checkpoint key names) and
scripts/make_lance_optional.py (this CPU has no AVX2/BMI2).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ["MPLBACKEND"] = "Agg"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
# libOSMesa is not installed. libEGL is. MuJoCo itself is not installed.
os.environ["MUJOCO_GL"] = "egl"
os.environ["STABLEWM_HOME"] = str(ROOT / "data")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "huggingface" / "hub")
)
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
import torch
from torchvision.transforms import v2 as T

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained


def pusht_frame() -> np.ndarray:
    import gymnasium as gym

    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
    finally:
        env.close()
    if frame is None:
        raise RuntimeError("Push-T render() returned None")
    return np.asarray(frame)


def dummy_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


def to_encoder_pixels(frame: np.ndarray) -> torch.Tensor:
    """Match le-wm/eval.py: uint8 HWC -> ImageNet-normalized (1, 1, 3, 224, 224)."""
    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(size=224),
        ]
    )
    chw = transform(frame)
    return chw.unsqueeze(0).unsqueeze(0)


def main() -> None:
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("torch.version.cuda", torch.version.cuda)
    print("STABLEWM_HOME", os.environ["STABLEWM_HOME"])
    print("stable_worldmodel", swm.__file__)

    source = "pusht_rgb_array"
    try:
        frame = pusht_frame()
    except Exception as exc:
        print(f"Push-T env failed: {type(exc).__name__}: {exc}")
        frame = dummy_frame()
        source = "dummy_uint8"

    print("frame_source", source)
    print("frame_shape", tuple(frame.shape), frame.dtype)

    pixels = to_encoder_pixels(frame)
    print("encoder_input", tuple(pixels.shape), pixels.dtype)

    # HF layout: weights.pt + config.json. stable-worldmodel 0.1.1 loads this
    # with hydra instantiate. Resolved under $STABLEWM_HOME/checkpoints/.
    model = load_pretrained("quentinll/lewm-pusht")
    model.eval()
    model.to("cpu")

    with torch.no_grad():
        info = model.encode({"pixels": pixels})
    z = info["emb"]
    flat = z.detach().float().cpu()
    print("z_shape", tuple(z.shape))
    print("z_mean", float(flat.mean()))
    print("z_std", float(flat.std(unbiased=False)))
    print("z_min", float(flat.min()))
    print("z_max", float(flat.max()))
    print("z_first8", [round(float(v), 6) for v in flat.reshape(-1)[:8]])


if __name__ == "__main__":
    main()
