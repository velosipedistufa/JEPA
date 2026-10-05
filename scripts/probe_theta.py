#!/usr/bin/env python3
"""Linear probe of Push-T block angle in a frozen LeWM latent.

Headless. Reuses the sanity_encode.py pixel pipeline.
Does not train LeWM and does not run CEM.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ["MPLBACKEND"] = "Agg"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["STABLEWM_HOME"] = str(ROOT / "data")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "huggingface" / "hub")
)
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from torchvision.transforms import v2 as T

import stable_pretraining as spt
from stable_worldmodel.wm.utils import load_pretrained

PROBE_DIR = ROOT / "data" / "probes"
FIG_DIR = ROOT / "reports" / "figures"

# Rotation episodes contribute one random frame plus three in-place turns.
# Extra random-only episodes thicken the official pose distribution.
N_ROT_EPISODES = 80
N_RAND_EPISODES = 120
TEST_FRACTION = 0.25
SPLIT_SEED = 0
RIDGE_ALPHA = 1.0
NEAR_DEFAULT_DEG = 30.0
ENCODE_BATCH = 8
SLOW_ENCODE_S = 30.0

ROTATIONS = (
    ("rot90", 0.5 * np.pi),
    ("rot180", np.pi),
    ("rot_m90", -0.5 * np.pi),
)


def wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def make_transform():
    """Same pipeline as scripts/sanity_encode.py."""
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(size=224),
        ]
    )


def to_encoder_pixels(transform, frame: np.ndarray) -> torch.Tensor:
    chw = transform(np.asarray(frame))
    return chw.unsqueeze(0).unsqueeze(0)


def summarize_value(value):
    if isinstance(value, np.ndarray):
        summary = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size <= 16:
            summary["value"] = np.asarray(value).tolist()
        return summary
    if isinstance(value, (float, int, str, np.floating, np.integer)):
        return value.item() if isinstance(value, np.generic) else value
    return str(type(value))


def dump_reset_api(env, path: Path) -> None:
    obs, info = env.reset(seed=0)
    unwrapped = env.unwrapped
    payload = {
        "env_id": "swm/PushT-v1",
        "render_mode": "rgb_array",
        "wrapper_type": type(env).__name__,
        "unwrapped_type": type(unwrapped).__name__,
        "obs_keys": sorted(obs.keys()),
        "obs_state": np.asarray(obs["state"]).tolist(),
        "obs_proprio": np.asarray(obs["proprio"]).tolist(),
        "info": {key: summarize_value(val) for key, val in info.items()},
        "state_layout": (
            "obs['state'] is length 7: agent_xy (2), block_xy (2), "
            "block_angle wrapped to [0, 2pi) via _get_obs, agent_vel (2). "
            "info['block_pose'] is (x, y, raw pymunk angle). "
            "info['goal_pose'] is (x, y, angle) of the goal overlay."
        ),
        "setter": (
            "PushT._set_state(state) writes agent xy, block xy, block angle, "
            "and agent velocity, then steps pymunk once. "
            "In-place turns call _set_state on the reset state with angle += dtheta."
        ),
        "default_block_angle_init_rad": 0.0,
        "methods": [
            name
            for name in ("_set_state", "_get_obs", "_get_info", "reset", "render")
            if hasattr(unwrapped, name)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print("reset_api", path)
    print(json.dumps(payload, indent=2))


def record_frame(rows, episode_id, tag, commanded_dtheta, info, kind):
    pose = np.asarray(info["block_pose"], dtype=np.float64)
    agent = np.asarray(info["pos_agent"], dtype=np.float64)
    theta = float(wrap_pi(pose[2]))
    rows.append(
        {
            "episode_id": int(episode_id),
            "episode_kind": kind,
            "tag": tag,
            "commanded_dtheta_rad": float(commanded_dtheta),
            "theta_rad": theta,
            "theta_deg": float(np.degrees(theta)),
            "x": float(pose[0]),
            "y": float(pose[1]),
            "agent_x": float(agent[0]),
            "agent_y": float(agent[1]),
            "raw_angle_rad": float(pose[2]),
        }
    )


def collect_frames(env):
    rows = []
    frames = []
    for episode_id in range(N_ROT_EPISODES):
        _obs, info = env.reset(seed=episode_id)
        unwrapped = env.unwrapped
        base_pose = np.asarray(info["block_pose"], dtype=np.float64)
        base_agent = np.asarray(info["pos_agent"], dtype=np.float64)
        base_vel = np.asarray(info["vel_agent"], dtype=np.float64)
        base_state = np.array(
            [
                base_agent[0],
                base_agent[1],
                base_pose[0],
                base_pose[1],
                base_pose[2],
                base_vel[0],
                base_vel[1],
            ],
            dtype=np.float64,
        )
        frames.append(np.asarray(env.render()))
        record_frame(rows, episode_id, "random", 0.0, info, "rotation")
        for tag, dtheta in ROTATIONS:
            state = base_state.copy()
            state[4] = base_pose[2] + dtheta
            state[5:] = 0.0
            unwrapped._set_state(state)
            turned = unwrapped._get_info()
            frames.append(np.asarray(env.render()))
            record_frame(rows, episode_id, tag, dtheta, turned, "rotation")
        if (episode_id + 1) % 20 == 0:
            print(f"collected_rotation_episodes {episode_id + 1}/{N_ROT_EPISODES}")

    for offset in range(N_RAND_EPISODES):
        episode_id = 1000 + offset
        _obs, info = env.reset(seed=10_000 + offset)
        frames.append(np.asarray(env.render()))
        record_frame(rows, episode_id, "random", 0.0, info, "random_only")
        if (offset + 1) % 40 == 0:
            print(f"collected_random_episodes {offset + 1}/{N_RAND_EPISODES}")
    return rows, frames


def encode_all(model, transform, frames: list[np.ndarray]) -> np.ndarray:
    import time

    chunks = []
    one = to_encoder_pixels(transform, frames[0])
    started = time.perf_counter()
    with torch.no_grad():
        first = model.encode({"pixels": one})["emb"]
    one_s = time.perf_counter() - started
    print("single_encode_s", round(one_s, 4), "z", tuple(first.shape))
    if one_s > SLOW_ENCODE_S:
        raise SystemExit(
            f"single encode took {one_s:.1f}s > {SLOW_ENCODE_S}s; refusing to continue"
        )
    chunks.append(first[:, 0, :].cpu().numpy().astype(np.float32))
    rest = frames[1:]
    for start in range(0, len(rest), ENCODE_BATCH):
        batch_frames = rest[start : start + ENCODE_BATCH]
        pixels = torch.cat(
            [to_encoder_pixels(transform, frame) for frame in batch_frames],
            dim=0,
        )
        with torch.no_grad():
            emb = model.encode({"pixels": pixels})["emb"]
        chunks.append(emb[:, 0, :].cpu().numpy().astype(np.float32))
        done = 1 + min(start + ENCODE_BATCH, len(rest))
        if done % 80 == 0 or done == len(frames):
            print(f"encoded {done}/{len(frames)}")
    z = np.concatenate(chunks, axis=0)
    if z.shape != (len(frames), 192):
        raise RuntimeError(f"unexpected latent shape {z.shape}")
    return z


def save_table(rows, z: np.ndarray) -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROBE_DIR / "pusht_theta_probe.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        PROBE_DIR / "pusht_theta_probe.npz",
        z=z,
        episode_id=np.array([row["episode_id"] for row in rows], dtype=np.int32),
        tag=np.array([row["tag"] for row in rows]),
        episode_kind=np.array([row["episode_kind"] for row in rows]),
        commanded_dtheta_rad=np.array(
            [row["commanded_dtheta_rad"] for row in rows], dtype=np.float64
        ),
        theta_rad=np.array([row["theta_rad"] for row in rows], dtype=np.float64),
        theta_deg=np.array([row["theta_deg"] for row in rows], dtype=np.float64),
        x=np.array([row["x"] for row in rows], dtype=np.float64),
        y=np.array([row["y"] for row in rows], dtype=np.float64),
        agent_x=np.array([row["agent_x"] for row in rows], dtype=np.float64),
        agent_y=np.array([row["agent_y"] for row in rows], dtype=np.float64),
        raw_angle_rad=np.array(
            [row["raw_angle_rad"] for row in rows], dtype=np.float64
        ),
    )
    print("wrote", csv_path, "and", PROBE_DIR / "pusht_theta_probe.npz")


def episode_split(episode_ids: np.ndarray, kinds: np.ndarray):
    rng = np.random.default_rng(SPLIT_SEED)
    test_eps = []
    train_eps = []
    for kind in ("rotation", "random_only"):
        eps = np.unique(episode_ids[kinds == kind])
        rng.shuffle(eps)
        n_test = max(1, int(round(len(eps) * TEST_FRACTION)))
        test_eps.extend(eps[:n_test].tolist())
        train_eps.extend(eps[n_test:].tolist())
    test_eps = np.array(test_eps, dtype=np.int32)
    train_eps = np.array(train_eps, dtype=np.int32)
    return train_eps, test_eps


def circular_error_deg(theta_rad: np.ndarray, pred_cos: np.ndarray, pred_sin: np.ndarray):
    pred = np.arctan2(pred_sin, pred_cos)
    err = wrap_pi(pred - wrap_pi(theta_rad))
    return np.degrees(err), np.degrees(pred)


def mae(values: np.ndarray):
    if len(values) == 0:
        return None
    return float(np.mean(np.abs(values)))


def fit_probe(theta, z, train_mask):
    target = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    models = {
        "ridge": Ridge(alpha=RIDGE_ALPHA),
        "linear": LinearRegression(),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(z[train_mask], target[train_mask])
        pred = model.predict(z)
        err_deg, pred_deg = circular_error_deg(theta, pred[:, 0], pred[:, 1])
        fitted[name] = {
            "model": model,
            "pred_cos": pred[:, 0],
            "pred_sin": pred[:, 1],
            "err_deg": err_deg,
            "pred_deg": pred_deg,
        }
    return target, fitted


def subset_mae(err, mask) -> dict:
    chosen = err[mask]
    return {
        "n": int(mask.sum()),
        "mae_deg": mae(chosen),
        "median_abs_deg": (
            float(np.median(np.abs(chosen))) if mask.any() else None
        ),
    }


def decide(mae_rot180, mae_near) -> str:
    if mae_rot180 is None or mae_near is None:
        return "not measured"
    much_worse = (mae_rot180 > 2.0 * mae_near) or (mae_rot180 > 30.0)
    if much_worse:
        return "180° is weakly represented; rental finetune of LeWM is justified"
    return "do not finetune LeWM; only train the language port"


def metrics_block(name, pack, test_mask, tags, theta, train_mask) -> dict:
    err = pack["err_deg"]
    dtheta_default = np.abs(np.degrees(wrap_pi(theta)))
    near = test_mask & (dtheta_default < NEAR_DEFAULT_DEG)
    by_tag = {
        tag: subset_mae(err, test_mask & (tags == tag))
        for tag in ("random", "rot90", "rot180", "rot_m90")
    }
    edges = [0, 15, 30, 45, 60, 90, 135, 180]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 180:
            mask = test_mask & (dtheta_default >= lo) & (dtheta_default <= hi)
        else:
            mask = test_mask & (dtheta_default >= lo) & (dtheta_default < hi)
        bins.append(
            {
                "lo_deg": lo,
                "hi_deg": hi,
                **subset_mae(err, mask),
            }
        )
    theta_edges = np.linspace(-180, 180, 9)
    theta_bins = []
    true_deg = np.degrees(wrap_pi(theta))
    for lo, hi in zip(theta_edges[:-1], theta_edges[1:]):
        if hi == 180:
            mask = test_mask & (true_deg >= lo) & (true_deg <= hi)
        else:
            mask = test_mask & (true_deg >= lo) & (true_deg < hi)
        theta_bins.append({"lo_deg": float(lo), "hi_deg": float(hi), **subset_mae(err, mask)})
    target = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    pred = np.stack([pack["pred_cos"], pack["pred_sin"]], axis=1)
    ss_res = np.sum((target[train_mask] - pred[train_mask]) ** 2)
    ss_tot = np.sum((target[train_mask] - target[train_mask].mean(axis=0)) ** 2)
    block = {
        "model": name,
        "test": subset_mae(err, test_mask),
        "train": subset_mae(err, train_mask),
        "by_tag_test": by_tag,
        "near_default_test": {
            "definition": f"|wrapped theta| < {NEAR_DEFAULT_DEG} deg from the default angle 0",
            **subset_mae(err, near),
        },
        "abs_dtheta_bins_test": bins,
        "true_theta_bins_test": theta_bins,
        "train_cos_sin_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
    block["decision"] = decide(
        by_tag["rot180"]["mae_deg"], block["near_default_test"]["mae_deg"]
    )
    return block


def save_plots(theta, tags, test_mask, err, pred_deg, metrics) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    true_deg = np.degrees(wrap_pi(theta))
    colors = {
        "random": "#1f4e79",
        "rot90": "#2a9d8f",
        "rot180": "#c44536",
        "rot_m90": "#e9c46a",
    }

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for tag, color in colors.items():
        mask = test_mask & (tags == tag)
        ax.scatter(
            true_deg[mask],
            pred_deg[mask],
            s=16,
            alpha=0.75,
            c=color,
            label=tag,
            linewidths=0,
        )
    ax.plot([-180, 180], [-180, 180], color="black", linewidth=1.0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("true block angle (deg)")
    ax.set_aspect("equal")
    ax.set_ylabel("predicted angle atan2(sin, cos) (deg)")
    ax.set_title("Held-out frames, Ridge probe")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pred_vs_true_theta.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.6), sharex=True, sharey=True)
    edges = np.linspace(-180, 180, 9)
    for axis, lo, hi in zip(axes.ravel(), edges[:-1], edges[1:]):
        if hi == 180:
            mask = test_mask & (true_deg >= lo) & (true_deg <= hi)
        else:
            mask = test_mask & (true_deg >= lo) & (true_deg < hi)
        if mask.any():
            axis.hist(err[mask], bins=12, color="#1f4e79", range=(-180, 180))
        axis.set_title(f"[{lo:.0f}, {hi:.0f})", fontsize=8)
        axis.set_xlim(-180, 180)
    for axis in axes[:, 0]:
        axis.set_ylabel("count")
    for axis in axes[1, :]:
        axis.set_xlabel("circular error (deg)")
    fig.suptitle("Held-out circular error by bin of true angle")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_hist_by_theta_bin.pdf")
    plt.close(fig)

    bins = metrics["ridge"]["abs_dtheta_bins_test"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    centers = [0.5 * (item["lo_deg"] + item["hi_deg"]) for item in bins]
    heights = [item["mae_deg"] for item in bins]
    widths = [item["hi_deg"] - item["lo_deg"] for item in bins]
    heights_plot = [0.0 if item["mae_deg"] is None else item["mae_deg"] for item in bins]
    ax.bar(
        centers,
        heights_plot,
        width=np.array(widths) * 0.9,
        color="#1f4e79",
        align="center",
    )
    for center, item, height in zip(centers, bins, heights_plot):
        ax.text(center, height, f"n={item['n']}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("|angle| from default pose 0 deg")
    ax.set_ylabel("mean absolute circular error (deg)")
    ax.set_title("Held-out Ridge probe")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_vs_abs_dtheta.pdf")
    plt.close(fig)
    print("wrote figures under", FIG_DIR)


def main() -> None:
    import gymnasium as gym

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        dump_reset_api(env, PROBE_DIR / "reset_api.json")
        rows, frames = collect_frames(env)
    finally:
        env.close()

    print("n_frames", len(frames))
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    transform = make_transform()
    model = load_pretrained("quentinll/lewm-pusht")
    model.eval()
    model.to("cpu")
    z = encode_all(model, transform, frames)
    save_table(rows, z)

    theta = np.array([row["theta_rad"] for row in rows], dtype=np.float64)
    tags = np.array([row["tag"] for row in rows])
    kinds = np.array([row["episode_kind"] for row in rows])
    episode_ids = np.array([row["episode_id"] for row in rows], dtype=np.int32)
    train_eps, test_eps = episode_split(episode_ids, kinds)
    train_mask = np.isin(episode_ids, train_eps)
    test_mask = np.isin(episode_ids, test_eps)
    _target, fitted = fit_probe(theta, z, train_mask)

    report = {
        "n_frames": int(len(rows)),
        "n_rotation_episodes": N_ROT_EPISODES,
        "n_random_only_episodes": N_RAND_EPISODES,
        "test_fraction": TEST_FRACTION,
        "split_seed": SPLIT_SEED,
        "n_train_episodes": int(len(train_eps)),
        "n_test_episodes": int(len(test_eps)),
        "n_train_frames": int(train_mask.sum()),
        "n_test_frames": int(test_mask.sum()),
        "ridge_alpha": RIDGE_ALPHA,
        "default_pose_angle_rad": 0.0,
        "xy_shift_rot_vs_random_max": None,
        "ridge": None,
        "linear": None,
    }
    # How far did scripted turns move the block in the plane.
    shifts = []
    by_ep = {}
    for row in rows:
        by_ep.setdefault(row["episode_id"], []).append(row)
    for group in by_ep.values():
        base = [item for item in group if item["tag"] == "random"]
        if not base:
            continue
        bx, by = base[0]["x"], base[0]["y"]
        for item in group:
            if item["tag"] == "random":
                continue
            shifts.append(float(np.hypot(item["x"] - bx, item["y"] - by)))
    report["xy_shift_rot_vs_random_max"] = float(max(shifts)) if shifts else None
    report["xy_shift_rot_vs_random_mean"] = float(np.mean(shifts)) if shifts else None

    for name, pack in fitted.items():
        report[name] = metrics_block(name, pack, test_mask, tags, theta, train_mask)
        # models are not JSON serializable
    save_plots(
        theta,
        tags,
        test_mask,
        fitted["ridge"]["err_deg"],
        fitted["ridge"]["pred_deg"],
        report,
    )
    out = PROBE_DIR / "probe_metrics.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("decision", report["ridge"]["decision"])
    print("ridge_by_tag", json.dumps(report["ridge"]["by_tag_test"]))
    print("near_default", json.dumps(report["ridge"]["near_default_test"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
