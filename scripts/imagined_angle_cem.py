#!/usr/bin/env python3
"""Cycle 013: imagined angle of the first in-place +180 plan.

The F2 angle ridge is refit on stored float32 latents and pinned before CEM.
No Lance. No factor of 5. No z_g MLP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.linear_model import Ridge

import eval_official_cem as official
import oracle_rotate_cem as oracle

F13_MEAN = np.array([-0.007812564379916167, 0.006860687229453028], dtype=np.float64)
F13_STD = np.array([0.2084674874515821, 0.20674867061239938], dtype=np.float64)
F2_MAE = 13.751980913656148
BUDGET = 50
TORCH_SEED = 42


def wrap_pi(angle):
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def fit_ridge(data):
    train_eps, test_eps = oracle.episode_split(data["episode_id"], data["episode_kind"])
    train = np.isin(data["episode_id"], train_eps)
    test = np.isin(data["episode_id"], test_eps)
    z = np.asarray(data["z"])
    if z.dtype != np.float32:
        raise SystemExit(f"z dtype {z.dtype}")
    theta = data["theta_rad"].astype(np.float64)
    model = Ridge(alpha=1.0, fit_intercept=True).fit(
        z[train], np.stack([np.cos(theta[train]), np.sin(theta[train])], 1)
    )
    pred = model.predict(z[test])
    err = wrap_pi(np.arctan2(pred[:, 1], pred[:, 0]) - wrap_pi(theta[test]))
    mae = float(np.mean(np.abs(np.degrees(err))))
    if abs(mae - F2_MAE) > 1e-4:
        raise SystemExit(f"F2 pin failed mae={mae:.12f}; CEM was not run")
    weight = torch.tensor(model.coef_, dtype=torch.float32, device="cuda")
    bias = torch.tensor(model.intercept_, dtype=torch.float32, device="cuda")
    return weight, bias, mae


def decode_rad(z_h, weight, bias) -> float:
    pred = z_h @ weight.T + bias
    return float(torch.atan2(pred[1], pred[0]))


def main() -> None:
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    print("torch_check", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    data = np.load(oracle.PROBE)
    weight, bias, mae = fit_ridge(data)
    print("f2_pin_mae", mae, flush=True)
    _train, starts = oracle.test_starts(data)
    model = official.load_model()
    transform = official.make_transform()
    env = official.make_env()
    rows = []
    try:
        for index, row in enumerate(starts):
            env.reset(seed=0)
            oracle.apply_state(env, row, row["raw_angle_rad"])
            theta0 = float(env.unwrapped._get_obs()[4])
            xy0 = env.unwrapped._get_obs()[2:4].copy()
            pixels, _z_t = oracle.encode_render(model, transform, np.asarray(env.render()))
            oracle.apply_state(env, row, row["raw_angle_rad"] + np.pi)
            _pix_g, z_g = oracle.encode_render(model, transform, np.asarray(env.render()))
            oracle.apply_state(env, row, row["raw_angle_rad"])

            def cost(cand, pixels=pixels):
                z_h = official.rollout_last(model, pixels, cand)
                return ((z_h - z_g) ** 2).sum(-1)

            plan = official.cem(cost)
            z_h = official.rollout_last(model, pixels, plan.unsqueeze(0))[0]
            imagined_rad = decode_rad(z_h, weight, bias)
            imagined_err = abs(float(np.degrees(wrap_pi(np.array(imagined_rad - (theta0 + np.pi))))))
            zg_gap = float((z_h - z_g).norm())
            actions = official.denormalize_plan(plan.detach().cpu().numpy(), F13_MEAN, F13_STD)
            executed_steps = 0
            for action in actions:
                if executed_steps >= BUDGET:
                    break
                env.step(action)
                executed_steps += 1
            if executed_steps < BUDGET:
                oracle.run_cem_budget(
                    env, model, transform, F13_MEAN, F13_STD, z_g, BUDGET - executed_steps
                )
            executed = oracle.measure(env, theta0, xy0, np.pi)
            record = {
                "episode_id": row["episode_id"],
                "imagined_err_deg": imagined_err,
                "executed_err_deg": executed["angle_err_deg"],
                "executed_xy": executed["xy_err"],
                "reached": executed["reached"],
                "zh_zg_l2": zg_gap,
            }
            rows.append(record)
            print(
                f"ep {index} id={row['episode_id']} imag={imagined_err:.1f} "
                f"exec={executed['angle_err_deg']:.1f} xy={executed['xy_err']:.1f} ok={executed['reached']}",
                flush=True,
            )
            if len(rows) >= 5:
                reach = float(np.mean([item["reached"] for item in rows]))
                if reach > 0.30:
                    print("stop_rule executed reach above 0.30; imagination not scored", flush=True)
                    break
    finally:
        env.close()

    reached = [bool(item["reached"]) for item in rows]
    payload = {
        "n": len(rows),
        "stopped_early": len(rows) < len(starts),
        "reach_rate": float(np.mean(reached)) if rows else None,
        "median_imagined_err_deg": float(np.median([item["imagined_err_deg"] for item in rows])),
        "median_executed_err_deg": float(np.median([item["executed_err_deg"] for item in rows])),
        "f2_pin_mae_deg": mae,
        "f13_mean": F13_MEAN.tolist(),
        "action_scale": 1,
        "torch_seed": TORCH_SEED,
        "episodes": rows,
    }
    out = ROOT / "research" / "artifacts" / "013" / "imagined.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "WROTE",
        out,
        "median_imagined",
        payload["median_imagined_err_deg"],
        "median_executed",
        payload["median_executed_err_deg"],
        "reach",
        payload["reach_rate"],
        flush=True,
    )


if __name__ == "__main__":
    main()
