#!/usr/bin/env python3
"""env_common.py — shared runtime layer for MMAGAI experiments (runs on the 4090).

Designed by GLM 5.3 Prime. Provides model loading, dataset access, device management,
and the experiment_main decorator that all experiments use.
"""
from __future__ import annotations

import glob
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.environ.get("MMAGAI_ROOT", "/root/MMAGAIProjectOpen")
DATASET_DIR = os.environ.get(
    "MMAGAI_DATASET",
    os.path.join(PROJECT_ROOT, "data", "datasets", "galilai-group--lewm-pusht"),
)
STATE_DIR = os.path.join(PROJECT_ROOT, "automation", "state")
JOBS_DIR = os.path.join(STATE_DIR, "jobs")
CAPABILITIES_PATH = os.path.join(STATE_DIR, "capabilities.json")
ACTION_STATS_PATH = os.path.join(STATE_DIR, "action_stats.json")

os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

import torch  # noqa: E402


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _device()


def check_transformers_pin() -> None:
    import transformers
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        raise RuntimeError(
            f"transformers {transformers.__version__} >= 5 cannot load the LeWM checkpoint. "
            f"Pin transformers>=4.50,<5"
        )


# ── Dataset ─────────────────────────────────────────────────────────────────


def find_lance_dataset():
    """Find and open the Lance dataset. Returns (ds, path, format) or (None, None, None)."""
    candidates = [DATASET_DIR]
    for sub in sorted(glob.glob(os.path.join(DATASET_DIR, "*"))):
        candidates.append(sub)
    for sub in sorted(glob.glob(os.path.join(DATASET_DIR, "*", "*")))[:50]:
        candidates.append(sub)
    try:
        import lance
    except ImportError:
        lance = None
    for c in candidates:
        if lance is not None:
            try:
                ds = lance.dataset(c)
                if ds.count_rows() > 0:
                    return ds, c, "lance"
            except Exception:
                pass
    # Parquet fallback
    try:
        import pyarrow.dataset as pads
        pqs = glob.glob(os.path.join(DATASET_DIR, "**", "*.parquet"), recursive=True)
        if pqs:
            pqs.sort()
            base = os.path.dirname(pqs[0])
            ds = pads.dataset(pqs, format="parquet")
            if ds.count_rows() > 0:
                return ds, base, "parquet"
    except ImportError:
        pass
    return None, None, None


# ── Model Loading ────────────────────────────────────────────────────────────


MODEL_INSTANCE = None


def attach_world_model(quiet=False):
    """Load the frozen LeWM world model. Returns model instance (cached)."""
    global MODEL_INSTANCE
    if MODEL_INSTANCE is not None:
        return MODEL_INSTANCE
    from stable_worldmodel.wm.utils import load_pretrained
    if not quiet:
        print("Loading quentinll/lewm-pusht...", flush=True)
    model = load_pretrained("quentinll/lewm-pusht").to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    MODEL_INSTANCE = model
    return model


# ── Experiment Wrapper ───────────────────────────────────────────────────────


def experiment_main(run_fn: Callable):
    """Decorator for experiment entry points.
    
    Usage:
        def my_experiment(out_dir: str, args: dict) -> dict:
            ... return metrics dict
        
        if __name__ == "__main__":
            experiment_main(my_experiment)
    
    Reads env vars LEWM_JOB_DIR and LEWM_JOB_ARGS for the runner.
    Writes metrics.json and exits with 0 on success, 3 on error.
    """
    out_dir = os.environ.get("LEWM_JOB_DIR", ".")
    args = json.loads(os.environ.get("LEWM_JOB_ARGS", "{}"))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    metrics = {"job_args": args, "ok": True}
    try:
        check_transformers_pin()
        m = run_fn(out_dir, args)
        metrics.update(m or {})
    except Exception as e:
        metrics.update({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=40),
        })
    metrics["wall_time_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print("METRICS_WRITTEN", flush=True)
    sys.exit(0 if metrics.get("ok") else 3)


# ── Environment utilities ────────────────────────────────────────────────────


def make_env():
    """Create the Push-T environment (no render)."""
    import gymnasium as gym
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    return env


def encode_frame(model, frame: np.ndarray) -> torch.Tensor:
    """Encode a single RGB frame to latent z. frame: (H, W, 3) uint8."""
    from torchvision.transforms import v2 as T
    from stable_pretraining.data.dataset_stats import ImageNet
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(**ImageNet),
        T.Resize(size=224),
    ])
    t = transform(frame).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 3, 224, 224)
    with torch.inference_mode():
        z = model.encode({"pixels": t})["emb"]  # (1, 1, 192)
    return z[0, 0]  # (192,)


def action_stats_from_dataset():
    """Compute or load action mean/std from the Lance dataset."""
    if os.path.exists(ACTION_STATS_PATH):
        return json.loads(open(ACTION_STATS_PATH).read())
    ds, path, fmt = find_lance_dataset()
    if ds is None:
        return None
    if fmt == "lance":
        tbl = ds.to_table(columns=["action"])
    else:
        import pyarrow.dataset as pads
        tbl = ds.to_table(columns=["action"])
    actions = tbl.column("action").to_numpy()
    actions = np.stack(actions)  # (N, 2)
    mean = actions.mean(axis=0).tolist()
    std = actions.std(axis=0, ddof=1).tolist()
    stats = {"mean": mean, "std_unbiased": std, "n": len(actions)}
    with open(ACTION_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ── CEM Planner ──────────────────────────────────────────────────────────────


def cem_plan(
    model,
    env,
    z_goal: torch.Tensor,
    n_samples=300,
    n_steps=30,
    topk=30,
    horizon=5,
    action_block=5,
    action_stats: dict | None = None,
    device: str = "cuda",
) -> tuple:
    """Run CEM planning. Returns (best_action_seq, info_dict)."""
    ACT_DIM = 2
    MODEL_ACT = action_block * ACT_DIM
    
    if action_stats:
        mean = torch.tensor(action_stats["mean"], device=device)
        std = torch.tensor(action_stats["std_unbiased"], device=device)
    
    def _denorm(a):
        if action_stats:
            a = a * std + mean
        return torch.clamp(a, -1, 1)
    
    # Sample initial actions
    mu = torch.zeros(MODEL_ACT, device=device)
    sigma = torch.ones(MODEL_ACT, device=device)
    
    for step in range(n_steps):
        samples = torch.randn(n_samples, MODEL_ACT, device=device) * sigma + mu
        costs = torch.zeros(n_samples, device=device)
        
        for i in range(n_samples):
            z = torch.zeros(192, device=device)
            actions_seq = samples[i].view(-1, ACT_DIM)
            for a_idx in range(action_block):
                a = _denorm(actions_seq[a_idx])
                # Model rollout for one step
                inp = {"emb": z.unsqueeze(0).unsqueeze(0),
                       "action": a.unsqueeze(0).unsqueeze(0)}
                with torch.inference_mode():
                    out = model.predict(inp)
                z = out["emb"][0, 0]
            costs[i] = torch.nn.functional.mse_loss(z, z_goal)
        
        elite_idx = torch.topk(costs, topk, largest=False).indices
        elites = samples[elite_idx]
        mu = elites.mean(dim=0)
        sigma = elites.std(dim=0)
    
    best = mu.view(-1, ACT_DIM)
    return _denorm(best).cpu().numpy(), {"final_cost": costs.min().item()}


# ── Capability Discovery ─────────────────────────────────────────────────────


def discover_capabilities() -> dict:
    """Run capability checks and write capabilities.json. Returns the dict."""
    caps = {
        "device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": "",
        "lance_dataset": False,
        "dataset_rows": 0,
        "action_stats": False,
        "model_loaded": False,
    }
    if torch.cuda.is_available():
        caps["gpu_name"] = torch.cuda.get_device_name(0)
        caps["gpu_count"] = torch.cuda.device_count()
    # Dataset
    ds, path, fmt = find_lance_dataset()
    if ds is not None:
        caps["lance_dataset"] = True
        try:
            caps["dataset_rows"] = ds.count_rows()
        except Exception:
            caps["dataset_rows"] = -1
        caps["dataset_format"] = fmt
        caps["dataset_path"] = path
    # Action stats
    stats = action_stats_from_dataset()
    if stats:
        caps["action_stats"] = True
        caps["action_mean"] = stats["mean"]
        caps["action_std"] = stats["std_unbiased"]
    # Model
    try:
        model = attach_world_model(quiet=True)
        caps["model_loaded"] = True
        caps["model_device"] = str(next(model.parameters()).device)
    except Exception as e:
        caps["model_error"] = str(e)
    # Transformers
    import transformers
    caps["transformers_version"] = transformers.__version__
    # Write
    with open(CAPABILITIES_PATH, "w") as f:
        json.dump(caps, f, indent=2)
    return caps