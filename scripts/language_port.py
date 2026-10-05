#!/usr/bin/env python3
"""Language port stub on frozen LeWM latents.

A displacement head and an energy head share one map f(text) -> R^192.
f is fit by least squares on collected same-xy angle turns.
CLIP weights are not downloaded here. A bag-of-commands embedding is the
CPU stand-in. Swap that embedder for a frozen CLIP text encoder on the rental.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np

PROBE_NPZ = ROOT / "data" / "probes" / "pusht_theta_probe.npz"
OUT_JSON = ROOT / "data" / "probes" / "language_port_fit.json"
OUT_W = ROOT / "data" / "probes" / "language_port_W.npz"

COMMANDS = (
    "rotate T 90",
    "rotate T 180",
    "rotate T -90",
    "put T on the target",
)
TAG_TO_COMMAND = {
    "rot90": "rotate T 90",
    "rot180": "rotate T 180",
    "rot_m90": "rotate T -90",
}
# Pairs whose measured turn is farther than this from the command are dropped.
MAX_DTHETA_ERR_RAD = np.deg2rad(20.0)
MAX_XY_SHIFT = 5.0


def clip_import_status() -> str:
    """Import CLIP classes in a child so a SIGILL cannot kill this process."""
    code = (
        "from transformers import CLIPTextModel, CLIPTokenizer\n"
        "print('clip_classes_import_ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )
    if proc.returncode == 0 and "clip_classes_import_ok" in proc.stdout:
        return (
            "transformers CLIPTextModel and CLIPTokenizer import on CPU. "
            "Weights were not downloaded. The fitted head uses the bag-of-commands "
            "embedding. Swap CLIP on the rental."
        )
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else f"exit {proc.returncode}"
    return (
        "CLIP text encoder was not used. "
        f"Import check failed ({detail}). "
        "Bag-of-commands embedding is the stand-in. Swap CLIP on the rental."
    )


class BagOfCommands:
    """One-hot over the fixed command list. Deterministic, no native libs."""

    def __init__(self, commands=COMMANDS):
        self.commands = tuple(commands)
        self.index = {text: i for i, text in enumerate(self.commands)}
        self.dim = len(self.commands)
        self.backend = "bag_of_commands"

    def embed(self, texts) -> np.ndarray:
        rows = []
        for text in texts:
            if text not in self.index:
                raise KeyError(text)
            row = np.zeros(self.dim, dtype=np.float64)
            row[self.index[text]] = 1.0
            rows.append(row)
        return np.stack(rows, axis=0)


def wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def build_pairs(data, train_episodes: np.ndarray | None = None):
    z = data["z"].astype(np.float64)
    episode = data["episode_id"]
    tag = data["tag"]
    theta = data["theta_rad"]
    x = data["x"]
    y = data["y"]
    texts = []
    dtheta = []
    dz = []
    episodes = []
    xy_shift = []
    for ep in np.unique(episode):
        idx = np.where(episode == ep)[0]
        base = [i for i in idx if tag[i] == "random"]
        if len(base) != 1:
            continue
        b = base[0]
        for i in idx:
            command = TAG_TO_COMMAND.get(str(tag[i]))
            if command is None:
                continue
            shift = float(np.hypot(x[i] - x[b], y[i] - y[b]))
            measured = float(wrap_pi(theta[i] - theta[b]))
            commanded = {
                "rotate T 90": 0.5 * np.pi,
                "rotate T 180": np.pi,
                "rotate T -90": -0.5 * np.pi,
            }[command]
            # 180 and -180 are the same turn.
            if abs(wrap_pi(measured - commanded)) > MAX_DTHETA_ERR_RAD and abs(
                wrap_pi(measured - wrap_pi(commanded))
            ) > MAX_DTHETA_ERR_RAD:
                continue
            if shift > MAX_XY_SHIFT:
                continue
            texts.append(command)
            dtheta.append(measured)
            dz.append(z[i] - z[b])
            episodes.append(int(ep))
            xy_shift.append(shift)
    if not texts:
        raise RuntimeError("no angle pairs survived the same-xy filter")
    pack = {
        "text": np.array(texts),
        "dtheta_rad": np.array(dtheta, dtype=np.float64),
        "dz": np.stack(dz, axis=0),
        "episode_id": np.array(episodes, dtype=np.int32),
        "xy_shift": np.array(xy_shift, dtype=np.float64),
    }
    if train_episodes is None:
        return pack
    mask = np.isin(pack["episode_id"], train_episodes)
    return pack, mask


def fit_f(embedder: BagOfCommands, features: np.ndarray, dz: np.ndarray):
    """Least squares: features @ W ~= dz. W is (dim, 192)."""
    weight, residuals, rank, singular = np.linalg.lstsq(features, dz, rcond=None)
    pred = features @ weight
    err = pred - dz
    rmse = float(np.sqrt(np.mean(err ** 2)))
    dz_rms = float(np.sqrt(np.mean(dz ** 2)))
    per_command = {}
    # Recover which command each row was, from the one-hot.
    command_ids = features.argmax(axis=1)
    for index, text in enumerate(embedder.commands):
        chosen = command_ids == index
        if not np.any(chosen):
            per_command[text] = {
                "n": 0,
                "train_rmse": None,
                "mean_l2_residual": None,
                "status": "not measured",
            }
            continue
        diff = err[chosen]
        per_command[text] = {
            "n": int(chosen.sum()),
            "train_rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mean_l2_residual": float(np.mean(np.linalg.norm(diff, axis=1))),
            "mean_l2_target": float(np.mean(np.linalg.norm(dz[chosen], axis=1))),
            "status": "fit",
        }
    return {
        "W": weight,
        "rank": int(rank),
        "singular": singular.astype(float).tolist(),
        "lstsq_residual_sum_of_squares": (
            None if residuals.size == 0 else residuals.astype(float).tolist()
        ),
        "train_rmse": rmse,
        "train_target_rms": dz_rms,
        "train_rmse_over_target_rms": rmse / dz_rms if dz_rms > 0 else None,
        "per_command": per_command,
    }


def delta_z_hat(text: str, embedder: BagOfCommands, weight: np.ndarray) -> np.ndarray:
    """Interface piece: Δz_hat = f(text). Shape (192,)."""
    return (embedder.embed([text]) @ weight)[0]


def displace(z_t: np.ndarray, text: str, embedder: BagOfCommands, weight: np.ndarray) -> np.ndarray:
    """A) z_g = z_t + f(text)."""
    return np.asarray(z_t, dtype=np.float64) + delta_z_hat(text, embedder, weight)


def energy(
    z_t: np.ndarray,
    z_th: np.ndarray,
    text: str,
    embedder: BagOfCommands,
    weight: np.ndarray,
) -> float:
    """B) E = ||(z_tH - z_t) - f(text)||^2."""
    residual = np.asarray(z_th, dtype=np.float64) - np.asarray(z_t, dtype=np.float64)
    residual = residual - delta_z_hat(text, embedder, weight)
    return float(np.dot(residual, residual))


def same_split(episode_ids: np.ndarray, kinds: np.ndarray):
    """Match scripts/probe_theta.py episode split."""
    rng = np.random.default_rng(0)
    test_eps = []
    train_eps = []
    for kind in ("rotation", "random_only"):
        eps = np.unique(episode_ids[kinds == kind])
        rng.shuffle(eps)
        n_test = max(1, int(round(len(eps) * 0.25)))
        test_eps.extend(int(ep) for ep in eps[:n_test])
        train_eps.extend(int(ep) for ep in eps[n_test:])
    return np.array(train_eps, dtype=np.int32), np.array(test_eps, dtype=np.int32)


def main() -> None:
    if not PROBE_NPZ.is_file():
        raise SystemExit(f"missing {PROBE_NPZ}; run scripts/probe_theta.py first")
    data = np.load(PROBE_NPZ, allow_pickle=False)
    train_eps, test_eps = same_split(data["episode_id"], data["episode_kind"])
    pairs = build_pairs(data)
    train_mask = np.isin(pairs["episode_id"], train_eps)
    test_mask = np.isin(pairs["episode_id"], test_eps)
    embedder = BagOfCommands()
    status = clip_import_status()
    print(status)
    fit = fit_f(embedder, embedder.embed(pairs["text"][train_mask]), pairs["dz"][train_mask])
    weight = fit.pop("W")

    # Train residual already in `fit`. Score the same f on held-out episodes.
    test_feat = embedder.embed(pairs["text"][test_mask])
    test_pred = test_feat @ weight
    test_err = test_pred - pairs["dz"][test_mask]
    test_rmse = float(np.sqrt(np.mean(test_err ** 2))) if test_mask.any() else None

    energies = []
    # Rebuild z_t, z_tH for test pairs to exercise energy().
    z = data["z"].astype(np.float64)
    episode = data["episode_id"]
    tag = data["tag"]
    base_index = {}
    for i, ep in enumerate(episode.tolist()):
        if tag[i] == "random" and ep not in base_index:
            base_index[ep] = i
    for text, ep, dz_row in zip(
        pairs["text"][test_mask],
        pairs["episode_id"][test_mask],
        pairs["dz"][test_mask],
    ):
        z_t = z[base_index[int(ep)]]
        z_th = z_t + dz_row
        energies.append(energy(z_t, z_th, str(text), embedder, weight))

    # Smoke the displacement interface on the first training pair.
    example_text = str(pairs["text"][train_mask][0])
    example_ep = int(pairs["episode_id"][train_mask][0])
    z_t = z[base_index[example_ep]]
    z_g = displace(z_t, example_text, embedder, weight)

    payload = {
        "backend": embedder.backend,
        "clip": status,
        "commands": list(COMMANDS),
        "n_train_pairs": int(train_mask.sum()),
        "n_test_pairs": int(test_mask.sum()),
        "max_xy_shift_allowed": MAX_XY_SHIFT,
        "max_dtheta_err_deg": float(np.degrees(MAX_DTHETA_ERR_RAD)),
        "fit": fit,
        "test_rmse": test_rmse,
        "test_mean_energy": float(np.mean(energies)) if energies else None,
        "example": {
            "text": example_text,
            "z_t_norm": float(np.linalg.norm(z_t)),
            "z_g_norm": float(np.linalg.norm(z_g)),
            "delta_z_norm": float(np.linalg.norm(z_g - z_t)),
        },
        "unfitted_command": "put T on the target",
    }
    # Calling the unfitted command must be defined. It is the zero column.
    idle = delta_z_hat("put T on the target", embedder, weight)
    payload["put_T_on_the_target_delta_norm"] = float(np.linalg.norm(idle))
    payload["put_T_on_the_target_status"] = fit["per_command"]["put T on the target"]["status"]

    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT_W, W=weight.astype(np.float32), commands=np.array(COMMANDS))
    print("train_rmse", fit["train_rmse"])
    print("train_rmse_over_target_rms", fit["train_rmse_over_target_rms"])
    print("per_command", json.dumps(fit["per_command"]))
    print("test_rmse", test_rmse)
    print("test_mean_energy", payload["test_mean_energy"])
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
