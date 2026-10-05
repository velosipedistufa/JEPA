#!/usr/bin/env python3
"""Cycle 012: three rendered in-place stages, +30, +60, +90 degrees.

Same planner and F13 z-score as eval_official_cem. No Lance. No factor of 5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import eval_official_cem as official
import oracle_rotate_cem as oracle

F13_MEAN = np.array([-0.007812564379916167, 0.006860687229453028], dtype=np.float64)
F13_STD = np.array([0.2084674874515821, 0.20674867061239938], dtype=np.float64)
STAGE_DELTAS = (np.pi / 6.0, np.pi / 3.0, np.pi / 2.0)
STAGE_BUDGET = 25
TORCH_SEED = 42


def snapshot(env) -> np.ndarray:
    return np.asarray(env.unwrapped._get_obs(), dtype=np.float64).copy()


def restore(env, state: np.ndarray) -> None:
    env.unwrapped._set_state(np.asarray(state, dtype=np.float64))


def main() -> None:
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    print(
        "torch_check",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0),
        flush=True,
    )
    data = np.load(oracle.PROBE)
    _train, starts = oracle.test_starts(data)
    print("n_starts", len(starts), flush=True)
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
            stage_xy = []
            prev_xy = 0.0
            increases = []
            for stage, delta in enumerate(STAGE_DELTAS, start=1):
                current = snapshot(env)
                oracle.apply_state(env, row, row["raw_angle_rad"] + delta)
                goal_obs = env.unwrapped._get_obs()
                _pix, z_g = oracle.encode_render(model, transform, np.asarray(env.render()))
                goal_dxy = float(np.linalg.norm(goal_obs[2:4] - xy0))
                restore(env, current)
                oracle.run_cem_budget(env, model, transform, F13_MEAN, F13_STD, z_g, STAGE_BUDGET)
                metrics = oracle.measure(env, theta0, xy0, delta)
                increase = metrics["xy_err"] - prev_xy
                increases.append(increase)
                prev_xy = metrics["xy_err"]
                stage_xy.append(
                    {
                        "stage": stage,
                        "delta_deg": float(np.degrees(delta)),
                        "angle_err_deg": metrics["angle_err_deg"],
                        "xy_err": metrics["xy_err"],
                        "xy_increase": increase,
                        "goal_render_dxy": goal_dxy,
                        "reached": metrics["reached"],
                    }
                )
            final = stage_xy[-1]
            reached = bool(
                final["angle_err_deg"] < 20.0 and final["xy_err"] < oracle.XY_LIMIT
            )
            record = {
                "episode_id": row["episode_id"],
                "reached": reached,
                "angle_err_deg": final["angle_err_deg"],
                "xy_err": final["xy_err"],
                "stage_xy_increases": increases,
                "stages": stage_xy,
            }
            rows.append(record)
            print(
                f"ep {index} id={row['episode_id']} ang={final['angle_err_deg']:.1f} "
                f"xy={final['xy_err']:.1f} ok={reached} inc={[round(v, 1) for v in increases]}",
                flush=True,
            )
            if len(rows) == 5 and all(all(inc > 30.0 for inc in item["stage_xy_increases"]) for item in rows):
                print("stop_rule every stage moved xy by more than 30 px", flush=True)
                break
    finally:
        env.close()

    n_ok = int(sum(item["reached"] for item in rows))
    payload = {
        "n": len(rows),
        "reached": n_ok,
        "f13_mean": F13_MEAN.tolist(),
        "f13_std": F13_STD.tolist(),
        "action_scale": 1,
        "torch_seed": TORCH_SEED,
        "stage_budget": STAGE_BUDGET,
        "episodes": rows,
    }
    out = ROOT / "research" / "artifacts" / "012" / "staged.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", out, "reached", n_ok, "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
