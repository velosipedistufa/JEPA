#!/usr/bin/env python3
"""Cycle 011: live reset goal under the official action z-score.

Does not open Lance. Does not multiply actions by 5. The scored bit is
eval_state, not the terminated flag.
"""

from __future__ import annotations

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
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

import eval_official_cem as official

# F13 from reports/gpu_4090/logs/baseline_official.json. Not re-estimated.
F13_MEAN = np.array([-0.007812564379916167, 0.006860687229453028], dtype=np.float64)
F13_STD = np.array([0.2084674874515821, 0.20674867061239938], dtype=np.float64)
N_EPISODES = 20
TORCH_SEED = 42


def run_live(env, model, transform, mean, std, goal_state, goal_frame) -> dict:
    """CEM from the current reset. Does not call reset again."""
    env.unwrapped._set_goal_state(np.asarray(goal_state, dtype=np.float64))
    goal_pixels = official.to_pixels(transform, official.hwc(goal_frame)).unsqueeze(0).cuda()
    with torch.inference_mode():
        z_g = model.encode({"pixels": goal_pixels})["emb"][0, -1]

    executed = 0
    eval_success = False
    final = None
    plan_norm = None
    while executed < official.EVAL_BUDGET and not eval_success:
        frame = official.hwc(np.asarray(env.render()))
        pixels = official.to_pixels(transform, frame).unsqueeze(0).cuda()

        def cost(cand, pixels=pixels):
            z_h = official.rollout_last(model, pixels, cand)
            return ((z_h - z_g) ** 2).sum(-1)

        mean_norm = official.cem(cost)
        if plan_norm is None:
            plan_norm = mean_norm.detach().cpu().numpy()
        env_actions = official.denormalize_plan(mean_norm.detach().cpu().numpy(), mean, std)
        n_exec = min(len(env_actions), official.EVAL_BUDGET - executed)
        for action in env_actions[:n_exec]:
            _obs, _reward, terminated, truncated, _info = env.step(action)
            executed += 1
            eval_success, distance = env.unwrapped.eval_state(
                env.unwrapped.goal_state, env.unwrapped._get_obs()
            )
            final = {
                "eval_success": bool(eval_success),
                "terminated": bool(terminated),
                "state_dist": float(distance),
            }
            if eval_success or truncated or terminated:
                break
    pose = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    goal = np.asarray(goal_state, dtype=np.float64)
    pos = float(np.linalg.norm(goal[:4] - pose[:4]))
    final = final or {"eval_success": False, "terminated": False, "state_dist": None}
    final.update({"steps": executed, "pos_l2": pos})
    return final


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
    print("f13_mean", F13_MEAN.tolist(), "f13_std", F13_STD.tolist(), flush=True)
    model = official.load_model()
    transform = official.make_transform()
    env = official.make_env()
    rows = []
    try:
        _obs, info0 = env.reset(seed=0)
        pix = official.to_pixels(transform, official.hwc(np.asarray(env.render()))).unsqueeze(0).cuda()
        with torch.inference_mode():
            z = model.encode({"pixels": pix})["emb"]
        print("sanity_z", tuple(z.shape), float(z.mean()), flush=True)
        goal0 = np.asarray(info0["goal"])
        print("goal0", goal0.shape, goal0.dtype, flush=True)

        for seed in range(N_EPISODES):
            obs, info = env.reset(seed=seed)
            state0 = np.asarray(obs["state"], dtype=np.float64)
            goal_state = np.asarray(info["goal_state"], dtype=np.float64)
            goal_frame = np.asarray(info["goal"])
            row = run_live(env, model, transform, F13_MEAN, F13_STD, goal_state, goal_frame)
            row["seed"] = seed
            row["start_state"] = state0.tolist()
            rows.append(row)
            print(
                f"ep {seed} eval_success={row['eval_success']} "
                f"pos={row['pos_l2']:.1f} steps={row['steps']}",
                flush=True,
            )
            if len(rows) == 10:
                n_ok = sum(item["eval_success"] for item in rows)
                mean_pos = float(np.mean([item["pos_l2"] for item in rows]))
                print(f"checkpoint10 successes={n_ok} mean_pos_l2={mean_pos:.4f}", flush=True)
                if n_ok == 0 and mean_pos > 100.0:
                    print("stop_rule_checked f13_matches=true continuing_to_20", flush=True)
    finally:
        env.close()

    n_ok = int(sum(item["eval_success"] for item in rows))
    payload = {
        "n": len(rows),
        "successes": n_ok,
        "seeds": list(range(len(rows))),
        "f13_mean": F13_MEAN.tolist(),
        "f13_std": F13_STD.tolist(),
        "action_scale": 1,
        "torch_seed": TORCH_SEED,
        "mean_pos_l2": float(np.mean([item["pos_l2"] for item in rows])) if rows else None,
        "protocol": {
            "horizon": official.HORIZON,
            "action_block": official.ACTION_BLOCK,
            "cem": [official.N_SAMPLES, official.N_STEPS, official.TOPK],
            "eval_budget": official.EVAL_BUDGET,
            "goal": "info['goal'] from reset",
            "scored_bit": "eval_state",
        },
        "episodes": rows,
    }
    out = ROOT / "research" / "artifacts" / "011" / "live_goal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", out, "successes", n_ok, "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
