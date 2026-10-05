#!/usr/bin/env python3
"""Xy ridge from stored Push-T latents to block (x, y).

Cycle 002 is split seed 0. Cycle 003 is split seed 1 and no other change.
The fit subcommand writes measurements. It does not apply the 20 px or
40 px lines. The audit subcommand applies the frozen claim file.
Other map families are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("rotation", "random_only")
PROTOCOL_SEED = 0
PROTOCOL_TEST_FRACTION = 0.25
PROTOCOL_ALPHA = 1.0


class ProtocolStop(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    rooted = ROOT / path
    if rooted.is_file():
        return rooted
    return candidate


def load_claim(path: Path) -> dict:
    claim = json.loads(path.read_text())
    cycle = claim.get("cycle")
    if claim.get("family") != "xy" or cycle not in ("002", "003"):
        raise ProtocolStop(3, "claim is not an xy ridge for cycle 002 or 003")
    model = claim["model"]
    split = claim["split"]
    if float(model["alpha"]) != PROTOCOL_ALPHA:
        raise ProtocolStop(3, "alpha is not the frozen value 1")
    if model.get("standardize") is not False:
        raise ProtocolStop(3, "standardize must be false")
    if model.get("features") != "z" or model.get("targets") != ["x", "y"]:
        raise ProtocolStop(3, "features must be z and targets must be x, y")
    if model.get("fit_intercept") is not True or model.get("solver") != "auto":
        raise ProtocolStop(3, "ridge intercept or solver does not match the freeze")
    if model.get("dtype") != "float32":
        raise ProtocolStop(3, "z must be the stored float32 latent, the dtype that reproduces F2")
    seed = int(split["seed"])
    if cycle == "002" and seed != PROTOCOL_SEED:
        raise ProtocolStop(3, "split seed is not 0")
    if cycle == "003" and seed != 1:
        raise ProtocolStop(3, "cycle 003 seed is 1")
    if cycle == "003" and "seed0_regression" not in claim:
        raise ProtocolStop(3, "cycle 003 requires the seed-0 regression pin")
    if cycle == "003" and "baseline_pin" in claim:
        raise ProtocolStop(3, "cycle 003 must not pin the F2 angle MAE on the new split")
    if float(split["test_fraction"]) != PROTOCOL_TEST_FRACTION:
        raise ProtocolStop(3, "test fraction is not 0.25")
    if tuple(split["kinds"]) != KINDS:
        raise ProtocolStop(3, "kind order is part of the frozen split")
    return claim


def check_environment(claim: dict) -> None:
    import sklearn

    python_mm = f"{sys.version_info[0]}.{sys.version_info[1]}"
    expected = claim["environment"]
    if python_mm != expected["python_major_minor"]:
        raise ProtocolStop(3, f"python {python_mm} != {expected['python_major_minor']}")
    if np.__version__ != expected["numpy"]:
        raise ProtocolStop(3, f"numpy {np.__version__} != {expected['numpy']}")
    if sklearn.__version__ != expected["sklearn"]:
        raise ProtocolStop(3, f"sklearn {sklearn.__version__} != {expected['sklearn']}")


def check_input_hashes(claim: dict) -> None:
    inputs = claim["inputs"]
    for key, name_key in (
        ("npz_sha256", "npz"),
        ("metrics_sha256", "metrics"),
        ("split_source_sha256", "split_source"),
    ):
        path = resolve(inputs[name_key])
        if not path.is_file():
            raise ProtocolStop(3, f"missing {inputs[name_key]}")
        digest = sha256_file(path)
        if digest != inputs[key]:
            raise ProtocolStop(3, f"hash mismatch for {inputs[name_key]}")


def episode_split(
    episode_ids: np.ndarray, kinds: np.ndarray, seed: int = PROTOCOL_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Match scripts/probe_theta.py episode_split. Seed 0 is that function."""
    rng = np.random.default_rng(seed)
    test_eps: list[int] = []
    train_eps: list[int] = []
    for kind in KINDS:
        eps = np.unique(episode_ids[kinds == kind])
        rng.shuffle(eps)
        n_test = max(1, int(round(len(eps) * PROTOCOL_TEST_FRACTION)))
        test_eps.extend(eps[:n_test].tolist())
        train_eps.extend(eps[n_test:].tolist())
    return np.array(train_eps, dtype=np.int32), np.array(test_eps, dtype=np.int32)


def wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def make_ridge(alpha: float) -> Ridge:
    return Ridge(alpha=alpha, fit_intercept=True, solver="auto")


def angle_test_mae(
    z: np.ndarray, theta: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
) -> float:
    target = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    model = make_ridge(PROTOCOL_ALPHA)
    model.fit(z[train_mask], target[train_mask])
    pred = model.predict(z[test_mask])
    pred_angle = np.arctan2(pred[:, 1], pred[:, 0])
    err = wrap_pi(pred_angle - wrap_pi(theta[test_mask]))
    return float(np.mean(np.abs(np.degrees(err))))


def coordinate_mae(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y - pred), axis=0)


def evaluate(claim: dict) -> dict:
    """Fit the xy ridge. Caller must already have checked hashes and the environment."""
    started = time.perf_counter()
    npz_path = resolve(claim["inputs"]["npz"])
    data = np.load(npz_path, allow_pickle=False)
    required = (
        "z",
        "x",
        "y",
        "theta_rad",
        "episode_id",
        "episode_kind",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ProtocolStop(3, f"npz missing {missing}")

    z = np.asarray(data["z"])
    if z.dtype != np.float32:
        raise ProtocolStop(3, f"z dtype {z.dtype} is not stored float32")
    theta = data["theta_rad"].astype(np.float64)
    y = np.stack(
        [data["x"].astype(np.float64), data["y"].astype(np.float64)], axis=1
    )
    episode_ids = data["episode_id"].astype(np.int32)
    kinds = data["episode_kind"]
    split = claim["split"]

    if z.ndim != 2 or z.shape[1] != int(split["expected_z_dim"]):
        raise ProtocolStop(3, f"z shape {z.shape} is not the frozen latent")
    if z.shape[0] != int(split["expected_rows"]) or y.shape[0] != z.shape[0]:
        raise ProtocolStop(3, f"row count {z.shape[0]} is not {split['expected_rows']}")
    if not np.isfinite(z).all() or not np.isfinite(theta).all() or not np.isfinite(y).all():
        raise ProtocolStop(4, "non-finite values; MAE cannot be computed")

    reproduced = None
    pin_error = None
    if "seed0_regression" in claim:
        reg = claim["seed0_regression"]
        if int(reg["seed"]) != PROTOCOL_SEED:
            raise ProtocolStop(3, "regression seed is not 0")
        reg_train, reg_test = episode_split(episode_ids, kinds, PROTOCOL_SEED)
        if np.intersect1d(reg_train, reg_test).size:
            raise ProtocolStop(3, "seed 0 split overlaps")
        reg_train_mask = np.isin(episode_ids, reg_train)
        reg_test_mask = np.isin(episode_ids, reg_test)
        reg_angle = angle_test_mae(z, theta, reg_train_mask, reg_test_mask)
        if abs(reg_angle - float(reg["angle_mae_deg"])) > float(reg["angle_abs_tol_deg"]):
            raise ProtocolStop(
                2,
                "seed0 angle regression failed "
                f"mae_deg={reg_angle:.12f}; scored fit was not run",
            )
        reg_model = make_ridge(PROTOCOL_ALPHA)
        reg_model.fit(z[reg_train_mask], y[reg_train_mask])
        reg_mae = coordinate_mae(y[reg_test_mask], reg_model.predict(z[reg_test_mask]))
        mae_tol = float(reg["mae_abs_tol_px"])
        if (
            abs(float(reg_mae[0]) - float(reg["test_mae_x"])) > mae_tol
            or abs(float(reg_mae[1]) - float(reg["test_mae_y"])) > mae_tol
        ):
            raise ProtocolStop(
                2,
                "seed0 xy regression failed "
                f"x={float(reg_mae[0]):.12f} y={float(reg_mae[1]):.12f}; "
                "scored fit was not run",
            )
        reproduced = reg_angle
        pin_error = abs(reg_angle - float(reg["angle_mae_deg"]))

    train_eps, test_eps = episode_split(episode_ids, kinds, int(split["seed"]))
    train_mask = np.isin(episode_ids, train_eps)
    test_mask = np.isin(episode_ids, test_eps)
    overlap = np.intersect1d(train_eps, test_eps)
    n_train_frames = int(train_mask.sum())
    n_test_frames = int(test_mask.sum())
    if overlap.size or n_train_frames + n_test_frames != z.shape[0]:
        raise ProtocolStop(3, "train and test episodes do not partition the table")

    def count_kind(eps: np.ndarray, kind: str) -> int:
        kind_eps = np.unique(episode_ids[kinds == kind])
        return int(np.isin(eps, kind_eps).sum())

    counts = {
        "n_rows": int(z.shape[0]),
        "z_dim": int(z.shape[1]),
        "n_train_episodes": int(len(train_eps)),
        "n_test_episodes": int(len(test_eps)),
        "n_train_frames": n_train_frames,
        "n_test_frames": n_test_frames,
        "n_overlap_episodes": int(overlap.size),
        "n_test_rotation_episodes": count_kind(test_eps, "rotation"),
        "n_test_random_only_episodes": count_kind(test_eps, "random_only"),
    }
    expected_pairs = (
        ("n_train_episodes", "expected_train_episodes"),
        ("n_test_episodes", "expected_test_episodes"),
        ("n_train_frames", "expected_train_frames"),
        ("n_test_frames", "expected_test_frames"),
        ("n_test_rotation_episodes", "expected_test_rotation_episodes"),
        ("n_test_random_only_episodes", "expected_test_random_only_episodes"),
    )
    for got_key, expect_key in expected_pairs:
        if counts[got_key] != int(split[expect_key]):
            raise ProtocolStop(
                3,
                f"split count {got_key}={counts[got_key]} != {split[expect_key]}",
            )

    expected_ids = split.get("expected_test_episode_ids")
    if expected_ids is not None:
        got_ids = sorted(int(ep) for ep in test_eps)
        if got_ids != [int(ep) for ep in expected_ids]:
            raise ProtocolStop(
                3,
                "scored test episodes do not match the frozen list; xy fit was not run",
            )
    overlap_with_seed0 = None
    if "expected_test_overlap_with_seed0" in split:
        _, seed0_test = episode_split(episode_ids, kinds, PROTOCOL_SEED)
        overlap_with_seed0 = int(np.intersect1d(test_eps, seed0_test).size)
        if overlap_with_seed0 != int(split["expected_test_overlap_with_seed0"]):
            raise ProtocolStop(
                3,
                f"test overlap with seed 0 is {overlap_with_seed0}; xy fit was not run",
            )
        if overlap_with_seed0 >= 40:
            raise ProtocolStop(3, "split is not a second sample; xy fit was not run")

    if "baseline_pin" in claim:
        pin = claim["baseline_pin"]
        reproduced = angle_test_mae(z, theta, train_mask, test_mask)
        pin_error = abs(reproduced - float(pin["expected_mae_deg"]))
        if pin_error > float(pin["abs_tol_deg"]):
            raise ProtocolStop(
                2,
                "F2 angle pin failed "
                f"mae_deg={reproduced:.12f} expected={pin['expected_mae_deg']} "
                f"abs_err={pin_error:.3e}; xy fit was not run",
            )
    if reproduced is None or pin_error is None:
        raise ProtocolStop(3, "claim has no pin")

    model = make_ridge(PROTOCOL_ALPHA)
    model.fit(z[train_mask], y[train_mask])
    if int(model.n_features_in_) != int(split["expected_z_dim"]):
        raise ProtocolStop(3, "ridge saw a feature count other than z")
    pred_test = model.predict(z[test_mask])
    pred_train = model.predict(z[train_mask])
    y_test = y[test_mask]
    y_train = y[train_mask]
    train_mean = y_train.mean(axis=0)
    constant_test = np.broadcast_to(train_mean, y_test.shape)
    test_mae = coordinate_mae(y_test, pred_test)
    train_mae = coordinate_mae(y_train, pred_train)
    constant_mae = coordinate_mae(y_test, constant_test)
    if not np.isfinite(test_mae).all() or not np.isfinite(constant_mae).all():
        raise ProtocolStop(4, "MAE is non-finite")

    test_idx = np.flatnonzero(test_mask)
    elapsed = time.perf_counter() - started
    return {
        "family": "xy",
        "cycle": claim["cycle"],
        **counts,
        "test_overlap_with_seed0": overlap_with_seed0,
        "angle_pin_mae_deg": reproduced,
        "angle_pin_abs_err_deg": pin_error,
        "test_mae_x": float(test_mae[0]),
        "test_mae_y": float(test_mae[1]),
        "train_mae_x": float(train_mae[0]),
        "train_mae_y": float(train_mae[1]),
        "constant_test_mae_x": float(constant_mae[0]),
        "constant_test_mae_y": float(constant_mae[1]),
        "train_mean_x": float(train_mean[0]),
        "train_mean_y": float(train_mean[1]),
        "test_indices": [int(i) for i in test_idx],
        "test_episode_ids": sorted(int(ep) for ep in test_eps),
        "train_episode_ids": sorted(int(ep) for ep in train_eps),
        "pred_test_x": [float(v) for v in pred_test[:, 0]],
        "pred_test_y": [float(v) for v in pred_test[:, 1]],
        "y_test_x": [float(v) for v in y_test[:, 0]],
        "y_test_y": [float(v) for v in y_test[:, 1]],
        "coef": [[float(v) for v in row] for row in np.asarray(model.coef_)],
        "intercept": [float(v) for v in np.asarray(model.intercept_)],
        "alpha": PROTOCOL_ALPHA,
        "seed": int(split["seed"]),
        "test_fraction": PROTOCOL_TEST_FRACTION,
        "features": "z",
        "fit_intercept": True,
        "standardize": False,
        "fit_seconds": elapsed,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "python": sys.version.split()[0],
    }


def git_text(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_pins(script: Path, claim_path: Path) -> dict:
    rel_script = str(script.resolve().relative_to(ROOT))
    rel_claim = str(claim_path.resolve().relative_to(ROOT))
    status = git_text(["status", "--porcelain", "--", rel_script, rel_claim])
    diff = git_text(["diff", "--", rel_script, rel_claim])
    return {
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "script_blob": git_text(["hash-object", rel_script]),
        "script_or_claim_dirty": bool(status or diff),
        "status_porcelain": status,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def arrays_close(left: list[float], right: list[float]) -> bool:
    if len(left) != len(right):
        return False
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=1e-5, atol=1e-5))


def audit(claim: dict, obs: dict, fresh: dict, claim_path: Path) -> dict:
    reasons: list[str] = []
    valid = True

    def fail(reason: str) -> None:
        nonlocal valid
        valid = False
        reasons.append(reason)

    if obs.get("claim_sha256") != sha256_file(claim_path):
        fail("claim hash does not match the observation")
    if obs.get("script_or_claim_dirty") is not False:
        fail("script or claim was dirty at fit time")
    commit = obs.get("git_commit")
    if not isinstance(commit, str) or len(commit) not in (40, 64):
        fail("fit has no commit")
    script = Path(__file__).resolve()
    try:
        blob = git_text(["hash-object", str(script.relative_to(ROOT))])
    except (subprocess.CalledProcessError, OSError) as exc:
        fail(f"git hash-object failed: {exc}")
        blob = ""
    if blob and blob != obs.get("script_blob"):
        fail("script blob changed after the fit")
    budget = float(claim["budget"]["wall_clock_seconds_max"])
    fit_seconds = obs.get("fit_seconds")
    if not isinstance(fit_seconds, (int, float)) or float(fit_seconds) > budget:
        fail("fit missed the wall-clock cap or did not record it")

    compare_keys = (
        "test_mae_x",
        "test_mae_y",
        "constant_test_mae_x",
        "constant_test_mae_y",
        "train_mae_x",
        "train_mae_y",
        "angle_pin_mae_deg",
        "n_train_frames",
        "n_test_frames",
        "n_train_episodes",
        "n_test_episodes",
        "n_overlap_episodes",
    )
    for key in compare_keys:
        if not np.isclose(float(obs[key]), float(fresh[key]), rtol=1e-5, atol=1e-5):
            fail(f"refit disagrees on {key}")
    for key in ("pred_test_x", "pred_test_y", "y_test_x", "y_test_y", "test_indices"):
        if not arrays_close(obs[key], fresh[key]):
            fail(f"refit disagrees on {key}")
    if not arrays_close(obs["coef"][0], fresh["coef"][0]) or not arrays_close(
        obs["coef"][1], fresh["coef"][1]
    ):
        fail("refit disagrees on coef")

    recomputed = coordinate_mae(
        np.stack(
            [np.asarray(fresh["y_test_x"]), np.asarray(fresh["y_test_y"])], axis=1
        ),
        np.stack(
            [np.asarray(fresh["pred_test_x"]), np.asarray(fresh["pred_test_y"])],
            axis=1,
        ),
    )
    mae = (float(recomputed[0]), float(recomputed[1]))
    constant = (float(fresh["constant_test_mae_x"]), float(fresh["constant_test_mae_y"]))
    if abs(mae[0] - float(fresh["test_mae_x"])) > 1e-6 or abs(mae[1] - float(fresh["test_mae_y"])) > 1e-6:
        fail("stored MAE does not match the predictions")

    if not valid:
        return {
            "verdict": "invalid",
            "decision": None,
            "reasons": reasons,
            "test_mae_x": mae[0],
            "test_mae_y": mae[1],
            "constant_test_mae_x": constant[0],
            "constant_test_mae_y": constant[1],
        }

    if not np.isfinite(mae).all() or not np.isfinite(constant).all():
        verdict = "inconclusive"
        reasons.append("MAE is non-finite")
    else:
        kill_at = float(claim["kill"]["either_test_mae_px_ge"])
        support_at = float(claim["success"]["both_test_mae_px_lt"])
        need_constant = bool(claim["success"]["each_strictly_below_constant"])
        killed = mae[0] >= kill_at or mae[1] >= kill_at
        below = mae[0] < support_at and mae[1] < support_at
        beats = mae[0] < constant[0] and mae[1] < constant[1]
        if killed:
            verdict = "kills"
            reasons.append(
                f"kill line {kill_at} px hit: test MAE x={mae[0]:.6f} y={mae[1]:.6f}"
            )
        elif below and (beats or not need_constant):
            verdict = "supports"
            reasons.append(
                f"both test MAEs are below {support_at} px and below the constant predictor"
            )
        else:
            verdict = "weakens"
            if below and not beats:
                reasons.append(
                    "MAE is below 20 px and does not beat the constant predictor"
                )
            else:
                reasons.append(
                    f"computed MAE is outside the support line and short of the kill line: "
                    f"x={mae[0]:.6f} y={mae[1]:.6f}"
                )
        reasons.append(
            f"constant test MAE x={constant[0]:.6f} y={constant[1]:.6f}"
        )
        reasons.append(
            f"train MAE x={float(fresh['train_mae_x']):.6f} y={float(fresh['train_mae_y']):.6f} is not scored"
        )

    if verdict not in claim["decision_map"]:
        fail_reason = f"no decision for {verdict}"
        return {
            "verdict": "invalid",
            "decision": None,
            "reasons": reasons + [fail_reason],
            "test_mae_x": mae[0],
            "test_mae_y": mae[1],
            "constant_test_mae_x": constant[0],
            "constant_test_mae_y": constant[1],
        }

    decision = claim["decision_map"][verdict]
    if verdict == "supports" and claim.get("one_seed_only") is True and decision != "continue":
        return {
            "verdict": "invalid",
            "decision": None,
            "reasons": reasons + ["one seed cannot map supports anywhere but continue"],
            "test_mae_x": mae[0],
            "test_mae_y": mae[1],
        }
    reasons.append(f"decision {decision}")
    reasons.append(claim["next_action"][decision])
    return {
        "verdict": verdict,
        "decision": decision,
        "next": claim["next_action"][decision],
        "reasons": reasons,
        "test_mae_x": mae[0],
        "test_mae_y": mae[1],
        "constant_test_mae_x": constant[0],
        "constant_test_mae_y": constant[1],
        "angle_pin_mae_deg": float(fresh["angle_pin_mae_deg"]),
        "n_test_frames": int(fresh["n_test_frames"]),
        "n_train_frames": int(fresh["n_train_frames"]),
        "git_commit": obs.get("git_commit"),
        "fit_seconds": obs.get("fit_seconds"),
    }


def episode_level_mae(episode_ids: np.ndarray, episodes: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Unweighted mean across episodes of the mean absolute residual inside each episode."""
    scores = []
    for episode in episodes:
        chosen = episode_ids == episode
        scores.append(np.mean(np.abs(residual[chosen]), axis=0))
    return np.mean(np.stack(scores, axis=0), axis=0)


def evaluate_episode(claim: dict) -> dict:
    """Rescore the frozen xy ridge by episode. Frame-level numbers are a regression pin only."""
    started = time.perf_counter()
    data = np.load(resolve(claim["inputs"]["npz"]), allow_pickle=False)
    z = np.asarray(data["z"])
    if z.dtype != np.float32:
        raise ProtocolStop(3, f"z dtype {z.dtype} is not stored float32")
    theta = data["theta_rad"].astype(np.float64)
    y = np.stack([data["x"].astype(np.float64), data["y"].astype(np.float64)], axis=1)
    episode_ids = data["episode_id"].astype(np.int32)
    kinds = data["episode_kind"]
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ProtocolStop(4, "non-finite values; MAE cannot be computed")

    by_seed = {}
    for seed_key, pin in claim["frame_regression"].items():
        seed = int(seed_key)
        train_eps, test_eps = episode_split(episode_ids, kinds, seed)
        if np.intersect1d(train_eps, test_eps).size:
            raise ProtocolStop(3, f"seed {seed} split overlaps")
        train_mask = np.isin(episode_ids, train_eps)
        test_mask = np.isin(episode_ids, test_eps)
        if seed == PROTOCOL_SEED:
            angle = angle_test_mae(z, theta, train_mask, test_mask)
            if abs(angle - float(pin["angle_mae_deg"])) > float(pin["angle_abs_tol_deg"]):
                raise ProtocolStop(
                    2,
                    f"seed 0 angle pin failed mae_deg={angle:.12f}; episode score was not computed",
                )
        else:
            angle = None
        model = make_ridge(PROTOCOL_ALPHA)
        model.fit(z[train_mask], y[train_mask])
        pred = model.predict(z)
        frame_mae = coordinate_mae(y[test_mask], pred[test_mask])
        mae_tol = float(pin["mae_abs_tol_px"])
        if (
            abs(float(frame_mae[0]) - float(pin["test_mae_x"])) > mae_tol
            or abs(float(frame_mae[1]) - float(pin["test_mae_y"])) > mae_tol
        ):
            raise ProtocolStop(
                2,
                f"seed {seed} frame regression failed "
                f"x={float(frame_mae[0]):.12f} y={float(frame_mae[1]):.12f}; "
                "episode score was not computed",
            )
        test_eps_sorted = np.array(sorted(int(ep) for ep in test_eps), dtype=np.int32)
        frame_counts = [int(np.sum(episode_ids == ep)) for ep in test_eps_sorted]
        if sorted(frame_counts) != sorted(claim["expected_test_frame_counts"]):
            raise ProtocolStop(3, f"seed {seed} frame counts do not match the freeze")
        train_mean = y[train_mask].mean(axis=0)
        constant = np.broadcast_to(train_mean, y.shape)
        episode_mae = episode_level_mae(episode_ids, test_eps_sorted, pred - y)
        # episode_level_mae takes residual = pred - y, abs inside. Good.
        constant_mae = episode_level_mae(episode_ids, test_eps_sorted, constant - y)
        by_seed[str(seed)] = {
            "frame_test_mae_x": float(frame_mae[0]),
            "frame_test_mae_y": float(frame_mae[1]),
            "episode_test_mae_x": float(episode_mae[0]),
            "episode_test_mae_y": float(episode_mae[1]),
            "episode_constant_mae_x": float(constant_mae[0]),
            "episode_constant_mae_y": float(constant_mae[1]),
            "n_test_episodes": int(len(test_eps_sorted)),
            "angle_pin_mae_deg": None if angle is None else float(angle),
        }
    return {
        "family": "xy-episode",
        "cycle": claim["cycle"],
        "by_seed": by_seed,
        "fit_seconds": time.perf_counter() - started,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "python": sys.version.split()[0],
    }


def _episode_quad(block: dict, prefix: str) -> dict:
    return {
        f"{prefix}_x": float(block["episode_test_mae_x"]),
        f"{prefix}_y": float(block["episode_test_mae_y"]),
        f"{prefix}_constant_x": float(block["episode_constant_mae_x"]),
        f"{prefix}_constant_y": float(block["episode_constant_mae_y"]),
    }


def audit_episode(claim: dict, obs: dict, fresh: dict, claim_path: Path) -> dict:
    reasons: list[str] = []
    valid = True

    def fail(reason: str) -> None:
        nonlocal valid
        valid = False
        reasons.append(reason)

    if obs.get("claim_sha256") != sha256_file(claim_path):
        fail("claim hash does not match the observation")
    if obs.get("script_or_claim_dirty") is not False:
        fail("script or claim was dirty at fit time")
    commit = obs.get("git_commit")
    if not isinstance(commit, str) or len(commit) not in (40, 64):
        fail("fit has no commit")
    try:
        blob = git_text(["hash-object", "scripts/latent_maps.py"])
    except (subprocess.CalledProcessError, OSError) as exc:
        fail(f"git hash-object failed: {exc}")
        blob = ""
    if blob and blob != obs.get("script_blob"):
        fail("script blob changed after the fit")
    if float(obs.get("fit_seconds", 1e9)) > float(claim["budget"]["wall_clock_seconds_max"]):
        fail("fit missed the wall-clock cap")

    scored = {}
    for seed in ("0", "1"):
        for key in (
            "episode_test_mae_x",
            "episode_test_mae_y",
            "episode_constant_mae_x",
            "episode_constant_mae_y",
            "frame_test_mae_x",
            "frame_test_mae_y",
        ):
            if not np.isclose(float(obs["by_seed"][seed][key]), float(fresh["by_seed"][seed][key]), rtol=0, atol=1e-6):
                fail(f"refit disagrees on seed {seed} {key}")
        block = fresh["by_seed"][seed]
        scored[seed] = (
            float(block["episode_test_mae_x"]),
            float(block["episode_test_mae_y"]),
            float(block["episode_constant_mae_x"]),
            float(block["episode_constant_mae_y"]),
        )

    if not valid:
        return {"verdict": "invalid", "decision": None, "reasons": reasons, "by_seed": fresh["by_seed"]}

    kill_at = float(claim["kill"]["any_episode_mae_px_ge"])
    support_at = float(claim["success"]["all_episode_mae_px_lt"])
    values = [scored[seed][axis] for seed in ("0", "1") for axis in (0, 1)]
    constants_ok = all(scored[seed][axis] < scored[seed][axis + 2] for seed in ("0", "1") for axis in (0, 1))
    if not np.isfinite(values).all():
        verdict = "inconclusive"
        reasons.append("episode MAE is non-finite")
    elif any(value >= kill_at for value in values):
        verdict = "kills"
        reasons.append(f"kill line {kill_at} px hit")
    elif all(value < support_at for value in values) and constants_ok:
        verdict = "supports"
        reasons.append(f"both seeds and both coordinates are below {support_at} px and below the constant predictor")
    else:
        verdict = "weakens"
        reasons.append("episode MAE missed the support line and stayed under the kill line, or it failed the constant check")
    for seed in ("0", "1"):
        mae_x, mae_y, const_x, const_y = scored[seed]
        reasons.append(
            f"seed {seed} episode MAE x={mae_x:.6f} y={mae_y:.6f} "
            f"constant x={const_x:.6f} y={const_y:.6f}"
        )
    decision = claim["decision_map"][verdict]
    reasons.append(f"decision {decision}")
    reasons.append(claim["next_action"][decision])
    return {
        "verdict": verdict,
        "decision": decision,
        "next": claim["next_action"][decision],
        "reasons": reasons,
        "by_seed": {
            seed: {
                "episode_test_mae_x": scored[seed][0],
                "episode_test_mae_y": scored[seed][1],
                "episode_constant_mae_x": scored[seed][2],
                "episode_constant_mae_y": scored[seed][3],
            }
            for seed in ("0", "1")
        },
        "git_commit": obs.get("git_commit"),
        "fit_seconds": obs.get("fit_seconds"),
    }


def command_fit_episode(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        if claim.get("cycle") != "004" or claim.get("family") != "xy-episode":
            raise ProtocolStop(3, "claim is not cycle 004 xy-episode")
        if int(claim["split"]["seeds"][0]) != 0 or int(claim["split"]["seeds"][1]) != 1:
            raise ProtocolStop(3, "cycle 004 seeds are 0 and 1")
        check_environment(claim)
        check_input_hashes(claim)
        measured = evaluate_episode(claim)
        pins = git_pins(Path(__file__).resolve(), claim_path)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    if pins["script_or_claim_dirty"]:
        print("script or claim is dirty; obs was not written", file=sys.stderr)
        return 3
    measured.update(pins)
    measured["claim_sha256"] = sha256_file(claim_path)
    measured["npz_sha256"] = claim["inputs"]["npz_sha256"]
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, measured)
    for seed in ("0", "1"):
        block = measured["by_seed"][seed]
        print(
            f"seed {seed} episode_mae_x {block['episode_test_mae_x']:.10f} "
            f"episode_mae_y {block['episode_test_mae_y']:.10f} "
            f"constant_x {block['episode_constant_mae_x']:.10f} "
            f"constant_y {block['episode_constant_mae_y']:.10f}"
        )
    print(f"fit_seconds {measured['fit_seconds']:.3f}")
    print(f"wrote {out}")
    return 0


def command_audit_episode(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    obs_path = resolve(args.obs)
    try:
        claim = json.loads(claim_path.read_text())
        check_environment(claim)
        check_input_hashes(claim)
        obs = json.loads(obs_path.read_text())
        fresh = evaluate_episode(claim)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    verdict = audit_episode(claim, obs, fresh, claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, verdict)
    print(json.dumps(verdict, indent=2, allow_nan=False))
    return 0 if verdict["verdict"] != "invalid" else 3


def evaluate_gate(claim: dict) -> dict:
    """Per-episode absolute errors for the frozen ridge. The fit does not apply the hit-rate lines."""
    started = time.perf_counter()
    data = np.load(resolve(claim["inputs"]["npz"]), allow_pickle=False)
    z = np.asarray(data["z"])
    if z.dtype != np.float32:
        raise ProtocolStop(3, f"z dtype {z.dtype} is not stored float32")
    theta = data["theta_rad"].astype(np.float64)
    y = np.stack([data["x"].astype(np.float64), data["y"].astype(np.float64)], axis=1)
    episode_ids = data["episode_id"].astype(np.int32)
    kinds = data["episode_kind"]
    by_seed = {}
    for seed_key, pin in claim["frame_regression"].items():
        seed = int(seed_key)
        train_eps, test_eps = episode_split(episode_ids, kinds, seed)
        train_mask = np.isin(episode_ids, train_eps)
        test_mask = np.isin(episode_ids, test_eps)
        if seed == PROTOCOL_SEED:
            angle = angle_test_mae(z, theta, train_mask, test_mask)
            if abs(angle - float(pin["angle_mae_deg"])) > float(pin["angle_abs_tol_deg"]):
                raise ProtocolStop(2, f"seed 0 angle pin failed mae_deg={angle:.12f}; gate was not computed")
        model = make_ridge(PROTOCOL_ALPHA)
        model.fit(z[train_mask], y[train_mask])
        pred = model.predict(z)
        frame_mae = coordinate_mae(y[test_mask], pred[test_mask])
        tol = float(pin["mae_abs_tol_px"])
        if abs(float(frame_mae[0]) - float(pin["test_mae_x"])) > tol or abs(float(frame_mae[1]) - float(pin["test_mae_y"])) > tol:
            raise ProtocolStop(
                2,
                f"seed {seed} frame regression failed x={float(frame_mae[0]):.12f} y={float(frame_mae[1]):.12f}; gate was not computed",
            )
        train_mean = y[train_mask].mean(axis=0)
        rows = []
        for episode in sorted(int(ep) for ep in test_eps):
            chosen = episode_ids == episode
            ridge = np.mean(np.abs(pred[chosen] - y[chosen]), axis=0)
            constant = np.mean(np.abs(train_mean - y[chosen]), axis=0)
            rows.append(
                {
                    "episode": episode,
                    "mae_x": float(ridge[0]),
                    "mae_y": float(ridge[1]),
                    "constant_mae_x": float(constant[0]),
                    "constant_mae_y": float(constant[1]),
                }
            )
        by_seed[str(seed)] = {"n_test_episodes": len(rows), "episodes": rows}
    return {
        "family": "xy-gate",
        "cycle": claim["cycle"],
        "by_seed": by_seed,
        "fit_seconds": time.perf_counter() - started,
    }


def _hit_rate(rows: list[dict], gate: float, constant: bool) -> float:
    if constant:
        hits = sum(row["constant_mae_x"] < gate and row["constant_mae_y"] < gate for row in rows)
    else:
        hits = sum(row["mae_x"] < gate and row["mae_y"] < gate for row in rows)
    return float(hits) / float(len(rows))


def audit_gate(claim: dict, obs: dict, fresh: dict, claim_path: Path) -> dict:
    reasons: list[str] = []
    valid = True

    def fail(reason: str) -> None:
        nonlocal valid
        valid = False
        reasons.append(reason)

    if obs.get("claim_sha256") != sha256_file(claim_path):
        fail("claim hash does not match the observation")
    if obs.get("script_or_claim_dirty") is not False:
        fail("script or claim was dirty at fit time")
    if not isinstance(obs.get("git_commit"), str) or len(obs["git_commit"]) not in (40, 64):
        fail("fit has no commit")
    try:
        blob = git_text(["hash-object", "scripts/latent_maps.py"])
    except (subprocess.CalledProcessError, OSError) as exc:
        fail(f"git hash-object failed: {exc}")
        blob = ""
    if blob and blob != obs.get("script_blob"):
        fail("script blob changed after the fit")
    gate = float(claim["gate_px"])
    rates = {}
    for seed in ("0", "1"):
        fresh_rows = fresh["by_seed"][seed]["episodes"]
        obs_rows = obs["by_seed"][seed]["episodes"]
        if len(fresh_rows) != int(claim["expected_test_episodes"]) or len(obs_rows) != len(fresh_rows):
            fail(f"seed {seed} episode count is not {claim['expected_test_episodes']}")
            continue
        for left, right in zip(obs_rows, fresh_rows):
            if int(left["episode"]) != int(right["episode"]) or abs(left["mae_x"] - right["mae_x"]) > 1e-6 or abs(left["mae_y"] - right["mae_y"]) > 1e-6:
                fail(f"refit disagrees on seed {seed} episode {right['episode']}")
                break
        rates[seed] = {
            "ridge": _hit_rate(fresh_rows, gate, constant=False),
            "constant": _hit_rate(fresh_rows, gate, constant=True),
        }
    if not valid:
        return {"verdict": "invalid", "decision": None, "reasons": reasons, "rates": rates}
    support_at = float(claim["success"]["both_seeds_hit_rate_ge"])
    kill_at = float(claim["kill"]["either_seed_hit_rate_le"])
    ridge_rates = [rates["0"]["ridge"], rates["1"]["ridge"]]
    beats = rates["0"]["ridge"] > rates["0"]["constant"] and rates["1"]["ridge"] > rates["1"]["constant"]
    if any(rate <= kill_at for rate in ridge_rates):
        verdict = "kills"
        reasons.append(f"kill line {kill_at} hit")
    elif all(rate >= support_at for rate in ridge_rates) and beats:
        verdict = "supports"
        reasons.append(f"both seeds are at or above {support_at} and above the constant predictor")
    else:
        verdict = "weakens"
        reasons.append("hit rate missed the support line and stayed above the kill line, or it failed the constant check")
    for seed in ("0", "1"):
        reasons.append(
            f"seed {seed} ridge_hit_rate={rates[seed]['ridge']:.6f} constant_hit_rate={rates[seed]['constant']:.6f}"
        )
    decision = claim["decision_map"][verdict]
    reasons.append(f"decision {decision}")
    reasons.append(claim["next_action"][decision])
    return {
        "verdict": verdict,
        "decision": decision,
        "next": claim["next_action"][decision],
        "reasons": reasons,
        "gate_px": gate,
        "rates": rates,
        "git_commit": obs.get("git_commit"),
    }


def command_fit_gate(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        if claim.get("cycle") != "005" or claim.get("family") != "xy-gate":
            raise ProtocolStop(3, "claim is not cycle 005 xy-gate")
        check_environment(claim)
        check_input_hashes(claim)
        measured = evaluate_gate(claim)
        pins = git_pins(Path(__file__).resolve(), claim_path)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    if pins["script_or_claim_dirty"]:
        print("script or claim is dirty; obs was not written", file=sys.stderr)
        return 3
    measured.update(pins)
    measured["claim_sha256"] = sha256_file(claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, measured)
    gate = float(claim["gate_px"])
    for seed in ("0", "1"):
        rows = measured["by_seed"][seed]["episodes"]
        print(
            f"seed {seed} ridge_hit_rate {_hit_rate(rows, gate, False):.6f} "
            f"constant_hit_rate {_hit_rate(rows, gate, True):.6f} n {len(rows)}"
        )
    print(f"wrote {out}")
    return 0


def command_audit_gate(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        check_environment(claim)
        check_input_hashes(claim)
        obs = json.loads(resolve(args.obs).read_text())
        fresh = evaluate_gate(claim)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    verdict = audit_gate(claim, obs, fresh, claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, verdict)
    print(json.dumps(verdict, indent=2, allow_nan=False))
    return 0 if verdict["verdict"] != "invalid" else 3


def rms_ratio(residual: np.ndarray, delta: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(delta ** 2)))
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ProtocolStop(4, "RMS denominator is zero; ratio cannot be computed")
    return float(np.sqrt(np.mean(residual ** 2)) / denominator)


def plus90_pairs(data, train_eps: np.ndarray) -> dict:
    """Same-xy +90° pairs. Gates match scripts/language_port.py: 5 px and 20°."""
    z = np.asarray(data["z"], dtype=np.float64)
    episode = data["episode_id"]
    tag = data["tag"]
    theta = data["theta_rad"]
    x = data["x"]
    y = data["y"]
    commanded = 0.5 * np.pi
    max_err = np.deg2rad(20.0)
    z_t = []
    z_next = []
    episodes = []
    for ep in np.unique(episode):
        idx = np.where(episode == ep)[0]
        base = [i for i in idx if tag[i] == "random"]
        if len(base) != 1:
            continue
        origin = base[0]
        for row in idx:
            if str(tag[row]) != "rot90":
                continue
            shift = float(np.hypot(x[row] - x[origin], y[row] - y[origin]))
            measured = wrap_pi(np.array(theta[row] - theta[origin]))
            if abs(float(wrap_pi(measured - commanded))) > max_err and abs(
                float(wrap_pi(measured - wrap_pi(np.array(commanded))))
            ) > max_err:
                continue
            if shift > 5.0:
                continue
            z_t.append(z[origin])
            z_next.append(z[row])
            episodes.append(int(ep))
    if not z_t:
        return {
            "z_t": np.zeros((0, z.shape[1])),
            "z_next": np.zeros((0, z.shape[1])),
            "episode_id": np.zeros((0,), dtype=np.int32),
            "train_mask": np.zeros((0,), dtype=bool),
        }
    z_t_arr = np.stack(z_t)
    z_next_arr = np.stack(z_next)
    episode_arr = np.array(episodes, dtype=np.int32)
    return {
        "z_t": z_t_arr,
        "z_next": z_next_arr,
        "episode_id": episode_arr,
        "train_mask": np.isin(episode_arr, train_eps),
    }


def evaluate_rotation(claim: dict) -> dict:
    started = time.perf_counter()
    if float(claim["pair_gate"]["max_xy_px"]) != 5.0 or float(claim["pair_gate"]["max_angle_deg"]) != 20.0:
        raise ProtocolStop(3, "pair gate is not 5 px and 20 degrees")
    if claim.get("command") != "rotate T 90" or int(claim["split"]["seed"]) != 0:
        raise ProtocolStop(3, "cycle 006 is the seed-0 +90 map")
    data = np.load(resolve(claim["inputs"]["npz"]), allow_pickle=False)
    episode_ids = data["episode_id"].astype(np.int32)
    kinds = data["episode_kind"]
    train_eps, test_eps = episode_split(episode_ids, kinds, 0)
    pairs = plus90_pairs(data, train_eps)
    train = pairs["train_mask"]
    test = np.isin(pairs["episode_id"], test_eps)
    n_train = int(train.sum())
    n_test = int(test.sum())
    minimum = int(claim["min_train_pairs"])
    if n_train < minimum or n_test == 0:
        return {
            "family": "rotation",
            "cycle": claim["cycle"],
            "small_sample": True,
            "n_train_pairs": n_train,
            "n_test_pairs": n_test,
            "min_train_pairs": minimum,
            "fit_seconds": time.perf_counter() - started,
        }
    z_t = pairs["z_t"]
    z_next = pairs["z_next"]
    weight, _, _, _ = np.linalg.lstsq(z_t[train], z_next[train], rcond=None)
    pred = z_t[test] @ weight
    delta = z_next[test] - z_t[test]
    mean_delta = (z_next[train] - z_t[train]).mean(axis=0)
    constant_pred = z_t[test] + mean_delta
    train_pred = z_t[train] @ weight
    return {
        "family": "rotation",
        "cycle": claim["cycle"],
        "small_sample": False,
        "n_train_pairs": n_train,
        "n_test_pairs": n_test,
        "test_ratio": rms_ratio(pred - z_next[test], delta),
        "constant_test_ratio": rms_ratio(constant_pred - z_next[test], delta),
        "train_ratio": rms_ratio(train_pred - z_next[train], z_next[train] - z_t[train]),
        "fit_seconds": time.perf_counter() - started,
    }


def audit_rotation(claim: dict, obs: dict, fresh: dict, claim_path: Path) -> dict:
    reasons: list[str] = []
    valid = True

    def fail(reason: str) -> None:
        nonlocal valid
        valid = False
        reasons.append(reason)

    if obs.get("claim_sha256") != sha256_file(claim_path):
        fail("claim hash does not match the observation")
    if obs.get("script_or_claim_dirty") is not False:
        fail("script or claim was dirty at fit time")
    if not isinstance(obs.get("git_commit"), str) or len(obs["git_commit"]) not in (40, 64):
        fail("fit has no commit")
    try:
        blob = git_text(["hash-object", "scripts/latent_maps.py"])
    except (subprocess.CalledProcessError, OSError) as exc:
        fail(f"git hash-object failed: {exc}")
        blob = ""
    if blob and blob != obs.get("script_blob"):
        fail("script blob changed after the fit")
    for key in ("n_train_pairs", "n_test_pairs", "small_sample"):
        if obs.get(key) != fresh.get(key):
            fail(f"refit disagrees on {key}")
    if not fresh["small_sample"]:
        for key in ("test_ratio", "constant_test_ratio"):
            if abs(float(obs[key]) - float(fresh[key])) > 1e-8:
                fail(f"refit disagrees on {key}")
    if not valid:
        return {"verdict": "invalid", "decision": None, "reasons": reasons}
    if fresh["small_sample"]:
        reasons.append(
            f"train +90 pairs {fresh['n_train_pairs']} are below {fresh['min_train_pairs']}; ratio was not computed"
        )
        decision = claim["small_sample_decision"]
        reasons.append(f"decision {decision}")
        reasons.append(claim["next_action"][decision])
        return {
            "verdict": "inconclusive",
            "decision": decision,
            "next": claim["next_action"][decision],
            "reasons": reasons,
            "n_train_pairs": fresh["n_train_pairs"],
            "n_test_pairs": fresh["n_test_pairs"],
            "git_commit": obs.get("git_commit"),
        }
    ratio = float(fresh["test_ratio"])
    constant = float(fresh["constant_test_ratio"])
    kill_at = float(claim["kill"]["test_ratio_ge"])
    support_at = float(claim["success"]["test_ratio_le"])
    if ratio >= kill_at:
        verdict = "kills"
        reasons.append(f"kill line {kill_at} hit: test ratio {ratio:.6f}")
    elif ratio <= support_at and ratio < constant:
        verdict = "supports"
        reasons.append(f"test ratio {ratio:.6f} is at most {support_at} and below the constant map")
    else:
        verdict = "weakens"
        reasons.append(f"test ratio {ratio:.6f} missed support or failed the constant check; constant ratio {constant:.6f}")
    reasons.append(f"constant test ratio {constant:.6f}")
    reasons.append(f"train ratio {float(fresh['train_ratio']):.6f} is not scored")
    reasons.append(f"n_train {fresh['n_train_pairs']} n_test {fresh['n_test_pairs']}")
    decision = claim["decision_map"][verdict]
    reasons.append(f"decision {decision}")
    reasons.append(claim["next_action"][decision])
    return {
        "verdict": verdict,
        "decision": decision,
        "next": claim["next_action"][decision],
        "reasons": reasons,
        "test_ratio": ratio,
        "constant_test_ratio": constant,
        "train_ratio": float(fresh["train_ratio"]),
        "n_train_pairs": fresh["n_train_pairs"],
        "n_test_pairs": fresh["n_test_pairs"],
        "git_commit": obs.get("git_commit"),
    }


def command_fit_rotation(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        if claim.get("cycle") != "006" or claim.get("family") != "rotation":
            raise ProtocolStop(3, "claim is not cycle 006 rotation")
        check_environment(claim)
        check_input_hashes(claim)
        measured = evaluate_rotation(claim)
        pins = git_pins(Path(__file__).resolve(), claim_path)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    if pins["script_or_claim_dirty"]:
        print("script or claim is dirty; obs was not written", file=sys.stderr)
        return 3
    measured.update(pins)
    measured["claim_sha256"] = sha256_file(claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, measured)
    if measured["small_sample"]:
        print(f"small_sample n_train {measured['n_train_pairs']} n_test {measured['n_test_pairs']}")
    else:
        print(
            f"test_ratio {measured['test_ratio']:.10f} "
            f"constant_test_ratio {measured['constant_test_ratio']:.10f} "
            f"n_train {measured['n_train_pairs']} n_test {measured['n_test_pairs']}"
        )
    print(f"wrote {out}")
    return 0


def command_audit_rotation(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        check_environment(claim)
        check_input_hashes(claim)
        obs = json.loads(resolve(args.obs).read_text())
        fresh = evaluate_rotation(claim)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    verdict = audit_rotation(claim, obs, fresh, claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, verdict)
    print(json.dumps(verdict, indent=2, allow_nan=False))
    return 0 if verdict["verdict"] != "invalid" else 3


TURN_COMMANDS = (
    "rotate T 90",
    "rotate T 180",
    "rotate T -90",
    "put T on the target",
)
TAG_TO_INDEX = {"rot90": 0, "rot180": 1, "rot_m90": 2}
TAG_TO_ANGLE = {"rot90": 0.5 * np.pi, "rot180": np.pi, "rot_m90": -0.5 * np.pi}


def turn_pairs(data) -> dict:
    """All three same-xy turns. Gates match scripts/language_port.py."""
    z = np.asarray(data["z"], dtype=np.float64)
    episode = data["episode_id"]
    tag = data["tag"]
    theta = data["theta_rad"]
    x = data["x"]
    y = data["y"]
    max_err = np.deg2rad(20.0)
    z_t = []
    dz = []
    command = []
    episodes = []
    for ep in np.unique(episode):
        idx = np.where(episode == ep)[0]
        base = [i for i in idx if tag[i] == "random"]
        if len(base) != 1:
            continue
        origin = base[0]
        for row in idx:
            name = str(tag[row])
            if name not in TAG_TO_INDEX:
                continue
            shift = float(np.hypot(x[row] - x[origin], y[row] - y[origin]))
            measured = wrap_pi(np.array(theta[row] - theta[origin]))
            commanded = TAG_TO_ANGLE[name]
            if abs(float(wrap_pi(measured - commanded))) > max_err and abs(
                float(wrap_pi(measured - wrap_pi(np.array(commanded))))
            ) > max_err:
                continue
            if shift > 5.0:
                continue
            z_t.append(z[origin])
            dz.append(z[row] - z[origin])
            command.append(TAG_TO_INDEX[name])
            episodes.append(int(ep))
    if not z_t:
        dim = z.shape[1]
        empty = np.zeros((0, dim))
        return {
            "z_t": empty,
            "dz": empty,
            "command": np.zeros((0,), dtype=np.int32),
            "episode_id": np.zeros((0,), dtype=np.int32),
        }
    return {
        "z_t": np.stack(z_t),
        "dz": np.stack(dz),
        "command": np.array(command, dtype=np.int32),
        "episode_id": np.array(episodes, dtype=np.int32),
    }


def command_one_hot(indices: np.ndarray) -> np.ndarray:
    eye = np.eye(len(TURN_COMMANDS), dtype=np.float64)
    return eye[indices]


def evaluate_state_map(claim: dict) -> dict:
    started = time.perf_counter()
    if int(claim["split"]["seed"]) != 0:
        raise ProtocolStop(3, "cycle 007 seed is 0")
    if float(claim["pair_gate"]["max_xy_px"]) != 5.0 or float(claim["pair_gate"]["max_angle_deg"]) != 20.0:
        raise ProtocolStop(3, "pair gate is not 5 px and 20 degrees")
    data = np.load(resolve(claim["inputs"]["npz"]), allow_pickle=False)
    episode_ids = data["episode_id"].astype(np.int32)
    train_eps, test_eps = episode_split(episode_ids, data["episode_kind"], 0)
    pairs = turn_pairs(data)
    train = np.isin(pairs["episode_id"], train_eps)
    test = np.isin(pairs["episode_id"], test_eps)
    n_train = int(train.sum())
    n_test = int(test.sum())
    if n_train != int(claim["k2_pin"]["n_train_pairs"]) or n_test != int(claim["k2_pin"]["n_test_pairs"]):
        raise ProtocolStop(
            2,
            f"pair count train {n_train} test {n_test} does not match F3; state map was not fit",
        )
    one_hot = command_one_hot(pairs["command"])
    k2_weight, _, _, _ = np.linalg.lstsq(one_hot[train], pairs["dz"][train], rcond=None)
    k2_train = one_hot[train] @ k2_weight
    k2_test = one_hot[test] @ k2_weight
    train_delta = pairs["dz"][train]
    test_delta = pairs["dz"][test]
    train_ratio = rms_ratio(k2_train - train_delta, train_delta)
    train_rmse = float(np.sqrt(np.mean((k2_train - train_delta) ** 2)))
    test_rmse = float(np.sqrt(np.mean((k2_test - test_delta) ** 2)))
    pin = claim["k2_pin"]
    if abs(train_ratio - float(pin["train_ratio"])) > float(pin["ratio_abs_tol"]):
        raise ProtocolStop(
            2,
            f"K2 train ratio {train_ratio:.12f} missed {pin['train_ratio']}; state map was not fit",
        )
    if abs(train_rmse - float(pin["train_rmse"])) > float(pin["rmse_abs_tol"]) or abs(
        test_rmse - float(pin["test_rmse"])
    ) > float(pin["rmse_abs_tol"]):
        raise ProtocolStop(
            2,
            f"K2 rmse train {train_rmse:.12f} test {test_rmse:.12f} missed the pin; state map was not fit",
        )
    features = np.concatenate([pairs["z_t"], one_hot], axis=1)
    weight, _, _, _ = np.linalg.lstsq(features[train], train_delta, rcond=None)
    pred_test = features[test] @ weight
    pred_train = features[train] @ weight
    k2_test_ratio = rms_ratio(k2_test - test_delta, test_delta)
    return {
        "family": "state-map",
        "cycle": claim["cycle"],
        "n_train_pairs": n_train,
        "n_test_pairs": n_test,
        "k2_train_ratio": train_ratio,
        "k2_test_ratio": k2_test_ratio,
        "test_ratio": rms_ratio(pred_test - test_delta, test_delta),
        "train_ratio": rms_ratio(pred_train - train_delta, train_delta),
        "fit_seconds": time.perf_counter() - started,
    }


def audit_state_map(claim: dict, obs: dict, fresh: dict, claim_path: Path) -> dict:
    reasons: list[str] = []
    valid = True

    def fail(reason: str) -> None:
        nonlocal valid
        valid = False
        reasons.append(reason)

    if obs.get("claim_sha256") != sha256_file(claim_path):
        fail("claim hash does not match the observation")
    if obs.get("script_or_claim_dirty") is not False:
        fail("script or claim was dirty at fit time")
    if not isinstance(obs.get("git_commit"), str) or len(obs["git_commit"]) not in (40, 64):
        fail("fit has no commit")
    try:
        blob = git_text(["hash-object", "scripts/latent_maps.py"])
    except (subprocess.CalledProcessError, OSError) as exc:
        fail(f"git hash-object failed: {exc}")
        blob = ""
    if blob and blob != obs.get("script_blob"):
        fail("script blob changed after the fit")
    for key in ("test_ratio", "k2_test_ratio", "n_train_pairs", "n_test_pairs"):
        left = obs.get(key)
        right = fresh.get(key)
        if isinstance(right, float):
            if abs(float(left) - float(right)) > 1e-8:
                fail(f"refit disagrees on {key}")
        elif left != right:
            fail(f"refit disagrees on {key}")
    if not valid:
        return {"verdict": "invalid", "decision": None, "reasons": reasons}
    ratio = float(fresh["test_ratio"])
    k2_ratio = float(fresh["k2_test_ratio"])
    kill_at = float(claim["kill"]["test_ratio_ge"])
    support_at = float(claim["success"]["test_ratio_le"])
    if ratio >= kill_at:
        verdict = "kills"
        reasons.append(f"kill line {kill_at} hit: test ratio {ratio:.6f}")
    elif ratio <= support_at and ratio < k2_ratio:
        verdict = "supports"
        reasons.append(f"test ratio {ratio:.6f} is at most {support_at} and below the command-only map")
    else:
        verdict = "weakens"
        reasons.append(f"test ratio {ratio:.6f} missed support or failed the command-only check")
    reasons.append(f"command-only test ratio {k2_ratio:.6f}")
    reasons.append(f"train ratio {float(fresh['train_ratio']):.6f} is not scored")
    decision = claim["decision_map"][verdict]
    reasons.append(f"decision {decision}")
    reasons.append(claim["next_action"][decision])
    return {
        "verdict": verdict,
        "decision": decision,
        "next": claim["next_action"][decision],
        "reasons": reasons,
        "test_ratio": ratio,
        "k2_test_ratio": k2_ratio,
        "train_ratio": float(fresh["train_ratio"]),
        "n_train_pairs": fresh["n_train_pairs"],
        "n_test_pairs": fresh["n_test_pairs"],
        "git_commit": obs.get("git_commit"),
    }


def command_fit_state_map(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        if claim.get("cycle") != "007" or claim.get("family") != "state-map":
            raise ProtocolStop(3, "claim is not cycle 007 state-map")
        check_environment(claim)
        check_input_hashes(claim)
        measured = evaluate_state_map(claim)
        pins = git_pins(Path(__file__).resolve(), claim_path)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    if pins["script_or_claim_dirty"]:
        print("script or claim is dirty; obs was not written", file=sys.stderr)
        return 3
    measured.update(pins)
    measured["claim_sha256"] = sha256_file(claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, measured)
    print(
        f"test_ratio {measured['test_ratio']:.10f} "
        f"k2_test_ratio {measured['k2_test_ratio']:.10f} "
        f"n_train {measured['n_train_pairs']} n_test {measured['n_test_pairs']}"
    )
    print(f"wrote {out}")
    return 0


def command_audit_state_map(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        check_environment(claim)
        check_input_hashes(claim)
        obs = json.loads(resolve(args.obs).read_text())
        fresh = evaluate_state_map(claim)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    verdict = audit_state_map(claim, obs, fresh, claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, verdict)
    print(json.dumps(verdict, indent=2, allow_nan=False))
    return 0 if verdict["verdict"] != "invalid" else 3


def evaluate_catalog(claim: dict) -> dict:
    started = time.perf_counter()
    if int(claim["split"]["seed"]) != 0:
        raise ProtocolStop(3, "cycle 008 seed is 0")
    data = np.load(resolve(claim["inputs"]["npz"]), allow_pickle=False)
    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    theta = data["theta_rad"].astype(np.float64)
    episode = data["episode_id"].astype(np.int32)
    kinds = data["episode_kind"]
    tag = data["tag"]
    _, test_eps = episode_split(episode, kinds, 0)
    rotation_eps = np.unique(episode[kinds == "rotation"])
    query_eps = np.array(sorted(int(ep) for ep in test_eps if ep in set(rotation_eps.tolist())), dtype=np.int32)
    if len(query_eps) != int(claim["expected_query_episodes"]):
        raise ProtocolStop(3, f"query episodes {len(query_eps)} are not the frozen count")
    commands = (("+90", 0.5 * np.pi), ("180", np.pi), ("-90", -0.5 * np.pi))
    rows = []
    same_episode_nearest = 0
    for ep in query_eps:
        members = np.where(episode == ep)[0]
        base_list = [i for i in members if tag[i] == "random"]
        if len(base_list) != 1:
            raise ProtocolStop(3, f"episode {int(ep)} does not have one random frame")
        base = base_list[0]
        others = np.where(episode != ep)[0]
        start_only = np.where(np.arange(len(episode)) != base)[0]
        start_dist = np.hypot(x[start_only] - x[base], y[start_only] - y[base])
        start_pick = start_only[int(np.argmin(start_dist))]
        if int(episode[start_pick]) == int(ep):
            same_episode_nearest += 1
        for name, turn in commands:
            target = float(wrap_pi(np.array(theta[base] + turn)))
            dist = np.hypot(x[others] - x[base], y[others] - y[base])
            pick = others[int(np.argmin(dist))]
            angle_err = abs(float(np.degrees(wrap_pi(np.array(theta[pick] - target)))))
            rows.append(
                {
                    "episode": int(ep),
                    "command": name,
                    "xy_px": float(dist.min()),
                    "angle_deg": angle_err,
                }
            )
    return {
        "family": "catalog",
        "cycle": claim["cycle"],
        "n_queries": len(rows),
        "n_query_episodes": int(len(query_eps)),
        "same_episode_nearest_fraction": same_episode_nearest / float(len(query_eps)),
        "queries": rows,
        "median_xy_px": float(np.median([row["xy_px"] for row in rows])),
        "median_angle_deg": float(np.median([row["angle_deg"] for row in rows])),
        "fit_seconds": time.perf_counter() - started,
    }


def audit_catalog(claim: dict, obs: dict, fresh: dict, claim_path: Path) -> dict:
    reasons: list[str] = []
    valid = True

    def fail(reason: str) -> None:
        nonlocal valid
        valid = False
        reasons.append(reason)

    if obs.get("claim_sha256") != sha256_file(claim_path):
        fail("claim hash does not match the observation")
    if obs.get("script_or_claim_dirty") is not False:
        fail("script or claim was dirty at fit time")
    if not isinstance(obs.get("git_commit"), str) or len(obs["git_commit"]) not in (40, 64):
        fail("fit has no commit")
    try:
        blob = git_text(["hash-object", "scripts/latent_maps.py"])
    except (subprocess.CalledProcessError, OSError) as exc:
        fail(f"git hash-object failed: {exc}")
        blob = ""
    if blob and blob != obs.get("script_blob"):
        fail("script blob changed after the fit")
    if len(obs["queries"]) != len(fresh["queries"]):
        fail("refit query count disagrees")
    else:
        for left, right in zip(obs["queries"], fresh["queries"]):
            if left["episode"] != right["episode"] or left["command"] != right["command"]:
                fail("refit query identity disagrees")
                break
            if abs(left["xy_px"] - right["xy_px"]) > 1e-6 or abs(left["angle_deg"] - right["angle_deg"]) > 1e-6:
                fail("refit query distance disagrees")
                break
    median_xy = float(np.median([row["xy_px"] for row in fresh["queries"]]))
    median_angle = float(np.median([row["angle_deg"] for row in fresh["queries"]]))
    if abs(median_xy - float(fresh["median_xy_px"])) > 1e-6 or abs(median_angle - float(fresh["median_angle_deg"])) > 1e-6:
        fail("stored median disagrees with the queries")
    if not valid:
        return {"verdict": "invalid", "decision": None, "reasons": reasons}
    if median_xy > float(claim["kill"]["median_xy_px_gt"]) or median_angle > float(claim["kill"]["median_angle_deg_gt"]):
        verdict = "kills"
        reasons.append(f"kill line hit: median xy {median_xy:.6f} px, median angle {median_angle:.6f} deg")
    elif median_xy <= float(claim["success"]["median_xy_px_le"]) and median_angle <= float(claim["success"]["median_angle_deg_le"]):
        verdict = "supports"
        reasons.append(f"median neighbor is inside the catalog gate: xy {median_xy:.6f} px, angle {median_angle:.6f} deg")
    else:
        verdict = "weakens"
        reasons.append(f"median xy {median_xy:.6f} px and median angle {median_angle:.6f} deg are between the lines")
    reasons.append(
        f"same-episode nearest fraction {float(fresh['same_episode_nearest_fraction']):.6f} is not scored"
    )
    decision = claim["decision_map"][verdict]
    reasons.append(f"decision {decision}")
    reasons.append(claim["next_action"][decision])
    return {
        "verdict": verdict,
        "decision": decision,
        "next": claim["next_action"][decision],
        "reasons": reasons,
        "median_xy_px": median_xy,
        "median_angle_deg": median_angle,
        "n_queries": fresh["n_queries"],
        "same_episode_nearest_fraction": fresh["same_episode_nearest_fraction"],
        "git_commit": obs.get("git_commit"),
    }


def command_fit_catalog(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        if claim.get("cycle") != "008" or claim.get("family") != "catalog":
            raise ProtocolStop(3, "claim is not cycle 008 catalog")
        check_environment(claim)
        check_input_hashes(claim)
        measured = evaluate_catalog(claim)
        pins = git_pins(Path(__file__).resolve(), claim_path)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    if pins["script_or_claim_dirty"]:
        print("script or claim is dirty; obs was not written", file=sys.stderr)
        return 3
    measured.update(pins)
    measured["claim_sha256"] = sha256_file(claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, measured)
    print(
        f"median_xy_px {measured['median_xy_px']:.10f} "
        f"median_angle_deg {measured['median_angle_deg']:.10f} "
        f"n_queries {measured['n_queries']}"
    )
    print(f"wrote {out}")
    return 0


def command_audit_catalog(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    try:
        claim = json.loads(claim_path.read_text())
        check_environment(claim)
        check_input_hashes(claim)
        obs = json.loads(resolve(args.obs).read_text())
        fresh = evaluate_catalog(claim)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    verdict = audit_catalog(claim, obs, fresh, claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, verdict)
    print(json.dumps({key: value for key, value in verdict.items() if key != "queries"}, indent=2, allow_nan=False))
    return 0 if verdict["verdict"] != "invalid" else 3


def command_fit(args: argparse.Namespace) -> int:
    if args.family == "xy-episode":
        return command_fit_episode(args)
    if args.family == "catalog":
        return command_fit_catalog(args)
    if args.family == "xy-gate":
        return command_fit_gate(args)
    if args.family == "rotation":
        return command_fit_rotation(args)
    if args.family == "state-map":
        return command_fit_state_map(args)
    if args.family != "xy":
        print("only the xy family is in this cycle", file=sys.stderr)
        return 2
    claim_path = resolve(args.claim)
    try:
        claim = load_claim(claim_path)
        check_environment(claim)
        check_input_hashes(claim)
        measured = evaluate(claim)
        pins = git_pins(Path(__file__).resolve(), claim_path)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    if pins["script_or_claim_dirty"]:
        print("script or claim is dirty; obs was not written", file=sys.stderr)
        return 3
    measured.update(pins)
    measured["claim_sha256"] = sha256_file(claim_path)
    measured["npz_sha256"] = claim["inputs"]["npz_sha256"]
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, measured)
    print(
        f"test_mae_x {measured['test_mae_x']:.10f} "
        f"test_mae_y {measured['test_mae_y']:.10f}"
    )
    print(
        f"constant_test_mae_x {measured['constant_test_mae_x']:.10f} "
        f"constant_test_mae_y {measured['constant_test_mae_y']:.10f}"
    )
    print(
        f"n_train_frames {measured['n_train_frames']} "
        f"n_test_frames {measured['n_test_frames']} "
        f"overlap {measured['n_overlap_episodes']} "
        f"angle_pin_mae_deg {measured['angle_pin_mae_deg']:.12f} "
        f"fit_seconds {measured['fit_seconds']:.3f}"
    )
    print(f"wrote {out}")
    return 0


def command_audit(args: argparse.Namespace) -> int:
    claim_path = resolve(args.claim)
    family = json.loads(claim_path.read_text()).get("family")
    if family == "xy-episode":
        return command_audit_episode(args)
    if family == "xy-gate":
        return command_audit_gate(args)
    if family == "rotation":
        return command_audit_rotation(args)
    if family == "state-map":
        return command_audit_state_map(args)
    if family == "catalog":
        return command_audit_catalog(args)
    obs_path = resolve(args.obs)
    try:
        claim = load_claim(claim_path)
        check_environment(claim)
        check_input_hashes(claim)
        obs = json.loads(obs_path.read_text())
        fresh = evaluate(claim)
    except ProtocolStop as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    except (OSError, json.JSONDecodeError) as exc:
        print(f"obs unreadable: {exc}", file=sys.stderr)
        return 3
    verdict = audit(claim, obs, fresh, claim_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, verdict)
    print(json.dumps(verdict, indent=2, allow_nan=False))
    return 0 if verdict["verdict"] != "invalid" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Cycle 002 xy ridge on the probe table")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--family", required=True)
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
