#!/usr/bin/env python3
"""Cycle 009: one-frame velocity probe on headless Push-T.

The fit subcommand renders, and encodes only if the pose match passes.
The audit subcommand refits the ridge from the saved latents and applies
the frozen R^2 lines. It does not render again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLBACKEND"] = "Agg"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["STABLEWM_HOME"] = str(ROOT / "data")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git_text(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def pair_rmse(left: np.ndarray, right: np.ndarray) -> float:
    diff = left.astype(np.float64) - right.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def coordinate_r2(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y - pred) ** 2, axis=0)
    center = y.mean(axis=0)
    ss_tot = np.sum((y - center) ** 2, axis=0)
    if np.any(ss_tot <= 0):
        raise SystemExit("test labels have no variance")
    return 1.0 - ss_res / ss_tot


def forbidden(seed: int, ranges: list[list[int]]) -> bool:
    return any(lo <= seed <= hi for lo, hi in ranges)


def render_pairs(claim: dict) -> dict:
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  registers swm/PushT-v1

    train = [int(seed) for seed in claim["split"]["train_seeds"]]
    test = [int(seed) for seed in claim["split"]["test_seeds"]]
    ranges = claim["forbidden_seed_ranges"]
    if any(forbidden(seed, ranges) for seed in train + test):
        raise SystemExit("a seed collides with the probe table")
    if len(train) + len(test) != 60 or len(set(train) & set(test)) != 0:
        raise SystemExit("split is not 40 train and 20 test")
    velocities = [np.array(row, dtype=np.float64) for row in claim["velocities"]]
    if len(velocities) != 2:
        raise SystemExit("expected two velocities")
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    rmses = []
    frames = []
    labels = []
    episodes = []
    try:
        for index, seed in enumerate(train + test):
            _obs, info = env.reset(seed=seed)
            unwrapped = env.unwrapped
            agent = np.asarray(info["pos_agent"], dtype=np.float64)
            pose = np.asarray(info["block_pose"], dtype=np.float64)
            pair = []
            for velocity in velocities:
                state = np.array(
                    [agent[0], agent[1], pose[0], pose[1], pose[2], velocity[0], velocity[1]],
                    dtype=np.float64,
                )
                unwrapped._set_state(state)
                frame = np.asarray(env.render())
                pair.append(frame)
                frames.append(frame)
                labels.append(velocity.copy())
                episodes.append(seed)
            rmses.append(pair_rmse(pair[0], pair[1]))
            if (index + 1) % 10 == 0:
                print(f"rendered {index + 1}/60 mean_rmse {float(np.mean(rmses)):.4f}", flush=True)
    finally:
        env.close()
    return {
        "frames": frames,
        "labels": np.stack(labels),
        "episodes": np.array(episodes, dtype=np.int32),
        "pair_rmse": np.array(rmses, dtype=np.float64),
        "train_seeds": np.array(train, dtype=np.int32),
    }


def encode_frames(frames: list[np.ndarray]) -> np.ndarray:
    import torch
    from torchvision.transforms import v2 as T

    import stable_pretraining as spt
    from stable_worldmodel.wm.utils import load_pretrained

    if torch.cuda.is_available():
        raise SystemExit("cuda is available; this probe stays on CPU")
    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(size=224),
        ]
    )
    model = load_pretrained("quentinll/lewm-pusht")
    model.eval()
    model.to("cpu")
    rows = []
    with torch.no_grad():
        for index, frame in enumerate(frames):
            pixels = transform(np.asarray(frame)).unsqueeze(0).unsqueeze(0)
            emb = model.encode({"pixels": pixels})["emb"]
            rows.append(emb[0, 0].cpu().numpy().astype(np.float32))
            if (index + 1) % 20 == 0:
                print(f"encoded {index + 1}/{len(frames)}", flush=True)
    return np.stack(rows)


def fit_r2(z: np.ndarray, labels: np.ndarray, episodes: np.ndarray, train_seeds: np.ndarray) -> dict:
    from sklearn.linear_model import Ridge

    train = np.isin(episodes, train_seeds)
    test = ~train
    target = labels[:, :1]
    if float(np.std(labels[:, 1])) != 0.0:
        raise SystemExit("v_y is not the constant protocol label")
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(z[train], target[train])
    pred = model.predict(z[test])
    scored = coordinate_r2(target[test], pred)
    return {
        "r2_x": float(scored[0]),
        "n_train_frames": int(train.sum()),
        "n_test_frames": int(test.sum()),
    }


def command_fit(args: argparse.Namespace) -> int:
    claim_path = Path(args.claim)
    claim = json.loads(claim_path.read_text())
    if claim.get("cycle") not in ("009", "010"):
        raise SystemExit("claim is not a velocity cycle")
    rendered = render_pairs(claim)
    mean_rmse = float(np.mean(rendered["pair_rmse"]))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cycle": "009",
        "mean_pair_rmse": mean_rmse,
        "pair_rmse": rendered["pair_rmse"].tolist(),
        "pose_match_failed": mean_rmse > float(claim["pixel_rmse_max"]),
        "episodes": rendered["episodes"].tolist(),
        "labels": rendered["labels"].tolist(),
        "train_seeds": rendered["train_seeds"].tolist(),
    }
    if payload["pose_match_failed"]:
        payload["encoded"] = False
        print(f"pose_match_failed mean_pair_rmse {mean_rmse:.6f}")
    else:
        if len(rendered["frames"]) > int(claim["max_frames"]):
            raise SystemExit("frame cap exceeded")
        z = encode_frames(rendered["frames"])
        np.savez_compressed(out.with_suffix(".npz"), z=z)
        scored = fit_r2(z, rendered["labels"], rendered["episodes"], rendered["train_seeds"])
        payload.update(scored)
        payload["encoded"] = True
        payload["z_sha256"] = sha256_file(out.with_suffix(".npz"))
        print(f"r2_x {scored['r2_x']:.10f} mean_pair_rmse {mean_rmse:.6f}")
    claim_rel = f"research/artifacts/{claim['cycle']}/claim.json"
    status = git_text(["status", "--porcelain", "--", "scripts/probe_velocity.py", claim_rel])
    diff = git_text(["diff", "--", "scripts/probe_velocity.py", claim_rel])
    if status or diff:
        print("script or claim is dirty; obs was not written", file=sys.stderr)
        return 3
    payload["git_commit"] = git_text(["rev-parse", "HEAD"])
    payload["script_blob"] = git_text(["hash-object", "scripts/probe_velocity.py"])
    payload["script_or_claim_dirty"] = False
    payload["claim_sha256"] = sha256_file(claim_path if claim_path.is_file() else ROOT / claim_path)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


def command_audit(args: argparse.Namespace) -> int:
    claim = json.loads(Path(args.claim).read_text())
    obs = json.loads(Path(args.obs).read_text())
    reasons = []
    valid = True
    if obs.get("script_or_claim_dirty") is not False:
        valid = False
        reasons.append("script or claim was dirty")
    if not isinstance(obs.get("git_commit"), str) or len(obs["git_commit"]) not in (40, 64):
        valid = False
        reasons.append("fit has no commit")
    blob = git_text(["hash-object", "scripts/probe_velocity.py"])
    if blob != obs.get("script_blob"):
        valid = False
        reasons.append("script blob changed after the fit")
    mean_rmse = float(obs["mean_pair_rmse"])
    if abs(mean_rmse - float(np.mean(obs["pair_rmse"]))) > 1e-8:
        valid = False
        reasons.append("mean RMSE does not match the pairs")
    if not valid:
        verdict = {"verdict": "invalid", "decision": None, "reasons": reasons}
    elif mean_rmse > float(claim["pixel_rmse_max"]) or not obs.get("encoded"):
        verdict = {
            "verdict": "inconclusive",
            "decision": claim["pose_fail_decision"],
            "next": claim["next_action"][claim["pose_fail_decision"]],
            "reasons": [f"mean paired RMSE {mean_rmse:.6f} failed the pose gate; R^2 was not scored"],
            "mean_pair_rmse": mean_rmse,
            "git_commit": obs.get("git_commit"),
        }
    else:
        z = np.load(Path(args.obs).with_suffix(".npz"))["z"]
        labels = np.array(obs["labels"], dtype=np.float64)
        episodes = np.array(obs["episodes"], dtype=np.int32)
        train_seeds = np.array(obs["train_seeds"], dtype=np.int32)
        fresh = fit_r2(z, labels, episodes, train_seeds)
        kill_at = float(claim["kill"]["v_x_r2_gt"])
        support_at = float(claim["success"]["v_x_r2_lt"])
        r2_x = fresh["r2_x"]
        if abs(r2_x - float(obs["r2_x"])) > 1e-6:
            verdict = {"verdict": "invalid", "decision": None, "reasons": ["refit disagrees on R^2"]}
        else:
            if r2_x > kill_at:
                name = "kills"
                reasons.append(f"kill line {kill_at} hit: r2_x {r2_x:.6f}")
            elif r2_x < support_at:
                name = "supports"
                reasons.append(f"v_x R^2 {r2_x:.6f} is below {support_at}")
            else:
                name = "weakens"
                reasons.append(f"v_x R^2 {r2_x:.6f} is between the lines")
            decision = claim["decision_map"][name]
            reasons.append(f"mean paired RMSE {mean_rmse:.6f}")
            reasons.append(f"decision {decision}")
            reasons.append(claim["next_action"][decision])
            verdict = {
                "verdict": name,
                "decision": decision,
                "next": claim["next_action"][decision],
                "reasons": reasons,
                "r2_x": fresh["r2_x"],
                "mean_pair_rmse": mean_rmse,
                "n_test_frames": fresh["n_test_frames"],
                "git_commit": obs.get("git_commit"),
            }
    out = Path(args.out)
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["verdict"] != "invalid" else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--claim", required=True)
    fit_p.add_argument("--out", required=True)
    audit_p = sub.add_parser("audit")
    audit_p.add_argument("--claim", required=True)
    audit_p.add_argument("--obs", required=True)
    audit_p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "fit":
        return command_fit(args)
    return command_audit(args)


if __name__ == "__main__":
    sys.exit(main())
