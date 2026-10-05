#!/usr/bin/env python3
"""Official-style image-goal CEM for frozen quentinll/lewm-pusht.

Action inputs are z-scored with the Push-T column stats from
get_column_normalizer (torch mean / unbiased std). They are inverse-transformed
and clipped to the env box before env.step. No global action scale.
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
from torchvision.transforms import v2 as T

import stable_pretraining as spt
from stable_worldmodel.data.formats.lance import LanceDataset
from stable_worldmodel.wm.utils import load_pretrained

LOGS = ROOT / "logs"
DATASET = ROOT / "data" / "datasets" / "galilai-group--lewm-pusht"

HORIZON = 5
RECEDING = 5
ACTION_BLOCK = 5
ACT_DIM = 2
MODEL_ACT = ACTION_BLOCK * ACT_DIM
N_SAMPLES = 300
N_STEPS = 30
TOPK = 30
GOAL_OFFSET = 25
EVAL_BUDGET = 50
SEED = 42
# PlanConfig.history_len default used by le-wm/eval.py. The checkpoint
# predictor.num_frames is 3 and is the attention window inside rollout.
HISTORY = 1


def make_transform():
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(size=224),
        ]
    )


def find_lance_dir() -> Path:
    if not DATASET.exists():
        raise SystemExit(f"missing dataset dir {DATASET}")
    hits = list(DATASET.rglob("*.lance"))
    hits = [p for p in hits if p.is_dir()]
    if not hits:
        raise SystemExit(f"no .lance table under {DATASET}")
    # Prefer the expert train table.
    for p in hits:
        if "pusht_expert_train" in p.name:
            return p
    return hits[0]


def open_dataset() -> LanceDataset:
    path = find_lance_dir()
    print("lance", path, flush=True)
    return LanceDataset(
        path,
        frameskip=1,
        num_steps=1,
        keys_to_load=["pixels", "action", "state", "proprio"],
        keys_to_cache=["action", "state"],
    )


def action_stats(ds: LanceDataset) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.asarray(ds.get_col_data("action"), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    mask = ~np.isnan(raw).any(axis=1)
    data = torch.from_numpy(raw[mask])
    mean = data.mean(0)
    std = data.std(0, unbiased=True).clamp_min(1e-6)
    info = {
        "shape": list(raw.shape),
        "n": int(mask.sum()),
        "mean": mean.tolist(),
        "std_unbiased": std.tolist(),
        "std_population": data.std(0, unbiased=False).tolist(),
        "raw_min": data.min(0).values.tolist(),
        "raw_max": data.max(0).values.tolist(),
    }
    print("action_stats", json.dumps(info), flush=True)
    return mean.numpy(), std.numpy(), info


def normalize_block(raw_5x2: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((raw_5x2 - mean) / std).reshape(-1).astype(np.float32)


def denormalize_plan(norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(horizon, 10) normalized -> (horizon * 5, 2) clipped env actions."""
    raw = norm.reshape(-1, ACTION_BLOCK, ACT_DIM) * std + mean
    return np.clip(raw, -1.0, 1.0).reshape(-1, ACT_DIM).astype(np.float32)


def to_pixels(transform, frame: np.ndarray) -> torch.Tensor:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    chw = transform(frame)
    return chw.unsqueeze(0)


def load_model():
    model = load_pretrained("quentinll/lewm-pusht")
    model.eval()
    model.requires_grad_(False)
    model.cuda()
    return model


def rollout_last(model, pixels: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """pixels (1, H, 3, 224, 224), actions (S, T, 10) already normalized."""
    samples = actions.shape[0]
    pix = pixels.unsqueeze(1).expand(1, samples, *pixels.shape[1:]).contiguous()
    with torch.inference_mode():
        out = model.rollout({"pixels": pix}, actions.unsqueeze(0))
    return out["predicted_emb"][0, :, -1]


def cem(cost_fn) -> torch.Tensor:
    device = "cuda"
    mean = torch.zeros(HORIZON, MODEL_ACT, device=device)
    std = torch.ones(HORIZON, MODEL_ACT, device=device)
    for _ in range(N_STEPS):
        noise = torch.randn(N_SAMPLES, HORIZON, MODEL_ACT, device=device)
        cand = noise * std + mean
        cand[0] = mean
        costs = cost_fn(cand)
        _vals, idx = torch.topk(costs, k=TOPK, largest=False)
        elite = cand[idx]
        mean = elite.mean(0)
        std = elite.std(0).clamp_min(1e-3)
    return mean


def make_env():
    import gymnasium as gym

    return gym.make("swm/PushT-v1", render_mode="rgb_array")


def valid_starts(ds: LanceDataset):
    pairs = []
    for ep, length in enumerate(ds.lengths.tolist()):
        last = int(length) - GOAL_OFFSET - 1
        if last < 0:
            continue
        for step in range(last + 1):
            pairs.append((ep, step))
    return pairs


def sample_starts(ds: LanceDataset, n: int):
    pairs = valid_starts(ds)
    g = np.random.default_rng(SEED)
    # Match eval.py: sample row indices then sort them.
    pick = np.sort(g.choice(len(pairs), size=n, replace=False))
    return [pairs[int(i)] for i in pick]


def load_window(ds: LanceDataset, ep: int, start: int) -> dict:
    # frameskip 1: goal frame is exactly `goal_offset` steps ahead.
    chunk = ds.load_chunk(
        np.array([ep]),
        np.array([start]),
        np.array([start + GOAL_OFFSET + 1]),
    )[0]
    return chunk


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def hwc(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return frame


def global_index(ds: LanceDataset, ep: int, step: int) -> int:
    return int(ds.offsets[ep] + step)


def one_step_match(ds, model, transform, mean, std, env, ep, start) -> dict:
    """Compare one dataset macro-action's imagined Δz with the real transition."""
    g = global_index(ds, ep, start)
    actions = np.asarray(ds.get_col_data("action"), dtype=np.float64)
    raw = actions[g : g + ACTION_BLOCK]
    if raw.shape != (ACTION_BLOCK, ACT_DIM):
        return {"error": f"unexpected action slice {raw.shape}"}
    norm = ((raw - mean) / std).reshape(1, 1, MODEL_ACT).astype(np.float32)
    window = load_window(ds, ep, start)
    state0 = as_numpy(window["state"][0]).astype(np.float64)
    env.reset(seed=0)
    env.unwrapped._set_state(state0)
    pixels = to_pixels(transform, hwc(np.asarray(env.render()))).unsqueeze(0).cuda()
    with torch.inference_mode():
        z0 = model.encode({"pixels": pixels})["emb"][0, -1]
    pred = rollout_last(model, pixels, torch.tensor(norm, device="cuda"))[0]
    for a in raw:
        env.step(np.clip(a, -1.0, 1.0).astype(np.float32))
    pixels_1 = to_pixels(transform, hwc(np.asarray(env.render()))).unsqueeze(0).cuda()
    with torch.inference_mode():
        z1 = model.encode({"pixels": pixels_1})["emb"][0, -1]
    real = z1 - z0
    imag = pred - z0
    return {
        "real_l2": float(real.norm()),
        "imag_l2": float(imag.norm()),
        "pred_vs_real_l2": float((pred - z1).norm()),
        "dataset_action_raw": raw.tolist(),
        "dataset_action_norm": norm.reshape(ACTION_BLOCK, ACT_DIM).tolist(),
    }


def run_episode(env, model, transform, mean, std, state0, goal_state, goal_frame) -> dict:
    env.reset(seed=0)
    env.unwrapped._set_state(np.asarray(state0, dtype=np.float64))
    env.unwrapped._set_goal_state(np.asarray(goal_state, dtype=np.float64))
    goal_pixels = to_pixels(transform, hwc(goal_frame)).unsqueeze(0).cuda()
    with torch.inference_mode():
        z_g = model.encode({"pixels": goal_pixels})["emb"][0, -1]

    executed = 0
    success = False
    final = None
    plan_norm = None
    while executed < EVAL_BUDGET and not success:
        frame = hwc(np.asarray(env.render()))
        pixels = to_pixels(transform, frame).unsqueeze(0).cuda()

        def cost(cand, pixels=pixels):
            z_h = rollout_last(model, pixels, cand)
            return ((z_h - z_g) ** 2).sum(-1)

        mean_norm = cem(cost)
        if plan_norm is None:
            plan_norm = mean_norm.detach().cpu().numpy()
        env_actions = denormalize_plan(mean_norm.detach().cpu().numpy(), mean, std)
        # receding_horizon == horizon, so execute the whole 25-step plan.
        n_exec = min(len(env_actions), EVAL_BUDGET - executed)
        for action in env_actions[:n_exec]:
            _obs, _rew, terminated, truncated, _info = env.step(action)
            executed += 1
            success, distance = env.unwrapped.eval_state(
                env.unwrapped.goal_state, env.unwrapped._get_obs()
            )
            final = {
                "success": bool(success or terminated),
                "state_dist": float(distance),
                "terminated": bool(terminated),
            }
            if final["success"] or truncated:
                break
    pose = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    goal = np.asarray(goal_state, dtype=np.float64)
    pos = float(np.linalg.norm(goal[:4] - pose[:4]))
    final = final or {"success": False, "state_dist": None, "terminated": False}
    final.update(
        {
            "steps": executed,
            "pos_l2": pos,
            "first_plan_norm": plan_norm.reshape(HORIZON, MODEL_ACT).tolist()
            if plan_norm is not None
            else None,
            "first_plan_raw": denormalize_plan(plan_norm, mean, std)[:ACTION_BLOCK].tolist()
            if plan_norm is not None
            else None,
        }
    )
    return final


def episode_split(episode_ids, kinds):
    rng = np.random.default_rng(0)
    test_eps, train_eps = [], []
    for kind in ("rotation", "random_only"):
        eps = np.unique(episode_ids[kinds == kind])
        rng.shuffle(eps)
        n_test = max(1, int(round(len(eps) * 0.25)))
        test_eps.extend(eps[:n_test].tolist())
        train_eps.extend(eps[n_test:].tolist())
    return np.asarray(train_eps), np.asarray(test_eps)


def fit_angle_ridge(data):
    from sklearn.linear_model import Ridge

    train_eps, test_eps = episode_split(data["episode_id"], data["episode_kind"])
    train = np.isin(data["episode_id"], train_eps)
    z = data["z"].astype(np.float64)
    theta = data["theta_rad"].astype(np.float64)
    model = Ridge(alpha=1.0).fit(
        z[train], np.stack([np.cos(theta[train]), np.sin(theta[train])], 1)
    )
    weight = torch.tensor(model.coef_, dtype=torch.float32, device="cuda")
    bias = torch.tensor(model.intercept_, dtype=torch.float32, device="cuda")
    return weight, bias, test_eps


def probe_cost_run(ds_unused, model, transform, mean, std):
    del ds_unused
    data = np.load(ROOT / "data" / "probes" / "pusht_theta_probe.npz")
    weight, bias, test_eps = fit_angle_ridge(data)
    eid = data["episode_id"]
    kind = data["episode_kind"]
    tag = data["tag"]
    indices = []
    for ep in test_eps.tolist():
        hit = np.where((eid == ep) & (kind == "rotation") & (tag == "random"))[0]
        if len(hit) == 1:
            indices.append(int(hit[0]))
    commands = (("rotate T 90", 0.5 * np.pi), ("rotate T 180", np.pi))
    env = make_env()
    rows = []
    try:
        for index in indices:
            row = {key: data[key][index].item() for key in ("episode_id", "agent_x", "agent_y", "x", "y", "raw_angle_rad")}
            for name, delta in commands:
                env.reset(seed=0)
                state = np.array(
                    [row["agent_x"], row["agent_y"], row["x"], row["y"], row["raw_angle_rad"], 0.0, 0.0],
                    dtype=np.float64,
                )
                env.unwrapped._set_state(state)
                theta_now = float(env.unwrapped._get_obs()[4])
                frame = hwc(np.asarray(env.render()))
                pixels = to_pixels(transform, frame).unsqueeze(0).cuda()

                def cost(cand, pixels=pixels, theta_now=theta_now, delta=delta):
                    z_h = rollout_last(model, pixels, cand)
                    cs = z_h @ weight.T + bias
                    theta_hat = torch.atan2(cs[:, 1], cs[:, 0])
                    err = torch.atan2(
                        torch.sin(theta_hat - theta_now - delta),
                        torch.cos(theta_hat - theta_now - delta),
                    )
                    return err ** 2

                plan = cem(cost).detach().cpu().numpy()
                actions = denormalize_plan(plan, mean, std)
                for action in actions:
                    env.step(action)
                theta = float(env.unwrapped._get_obs()[4])
                err = abs((theta - (theta_now + delta) + np.pi) % (2 * np.pi) - np.pi)
                pose = np.asarray(env.unwrapped._get_obs()[:4], dtype=np.float64)
                xy = float(np.linalg.norm(pose[2:4] - np.array([row["x"], row["y"]])))
                rec = {
                    "episode_id": int(row["episode_id"]),
                    "command": name,
                    "angle_err_deg": float(np.degrees(err)),
                    "xy_err": xy,
                    "reached": bool(err < np.pi / 9),
                }
                rows.append(rec)
                print(
                    f"probe ep={rec['episode_id']} {name} ang={rec['angle_err_deg']:.1f} "
                    f"xy={xy:.1f} ok={rec['reached']}",
                    flush=True,
                )
    finally:
        env.close()
    by = {}
    for name, _delta in commands:
        chosen = [r for r in rows if r["command"] == name]
        by[name] = {
            "n": len(chosen),
            "mean_angle_err_deg": float(np.mean([r["angle_err_deg"] for r in chosen])),
            "reach_rate": float(np.mean([r["reached"] for r in chosen])),
            "mean_xy_err": float(np.mean([r["xy_err"] for r in chosen])),
        }
    payload = {"by_command": by, "episodes": rows, "n_episodes": len(indices)}
    (LOGS / "probe_cost_official.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("PROBE", json.dumps(by), flush=True)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        ds = open_dataset()
        mean, std, _stats = action_stats(ds)
        model = load_model()
        transform = make_transform()
        probe_cost_run(ds, model, transform, mean, std)
        return

    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ds = open_dataset()
    mean, std, stats = action_stats(ds)
    print("n_episodes", len(ds.lengths), "columns", ds.column_names, flush=True)
    starts = sample_starts(ds, 50)
    model = load_model()
    transform = make_transform()
    env = make_env()
    ep0, st0 = starts[0]
    match = one_step_match(ds, model, transform, mean, std, env, ep0, st0)
    print("transition_match", json.dumps(match), flush=True)

    rows = []
    try:
        for i, (ep, start) in enumerate(starts[:n_limit]):
            window = load_window(ds, ep, start)
            state0 = as_numpy(window["state"][0])
            goal_state = as_numpy(window["state"][-1])
            goal_frame = as_numpy(window["pixels"][-1])
            row = run_episode(env, model, transform, mean, std, state0, goal_state, goal_frame)
            row["episode"] = int(ep)
            row["start"] = int(start)
            # Keep the plan only for the first episode in the saved file.
            if i > 0:
                row.pop("first_plan_norm", None)
                row.pop("first_plan_raw", None)
            rows.append(row)
            print(
                f"ep {i} id={ep} start={start} success={row['success']} "
                f"pos={row['pos_l2']:.1f} dist={row['state_dist']}",
                flush=True,
            )
    finally:
        env.close()

    n_ok = int(sum(r["success"] for r in rows))
    payload = {
        "n": len(rows),
        "successes": n_ok,
        "success_rate": n_ok / len(rows) if rows else None,
        "success_percent": 100.0 * n_ok / len(rows) if rows else None,
        "mean_pos_l2": float(np.mean([r["pos_l2"] for r in rows])),
        "mean_state_dist": float(np.mean([r["state_dist"] for r in rows if r["state_dist"] is not None])),
        "action_stats": stats,
        "transition_match": {k: v for k, v in match.items() if "action" not in k},
        "history_len": HISTORY,
        "predictor_num_frames": 3,
        "protocol": {
            "horizon": HORIZON,
            "receding_horizon": RECEDING,
            "action_block": ACTION_BLOCK,
            "cem": [N_SAMPLES, N_STEPS, TOPK],
            "goal_offset_steps": GOAL_OFFSET,
            "eval_budget": EVAL_BUDGET,
            "seed": SEED,
            "action_norm": "get_column_normalizer torch mean / unbiased std",
            "dataset": str(find_lance_dir()),
        },
        "episodes": rows,
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / ("baseline_official.json" if n_limit >= 50 else f"baseline_official_n{n_limit}.json")
    # Always keep the 10-ep file, and write the canonical name for a 50-ep run.
    if n_limit >= 50:
        (LOGS / "baseline_official.json").write_text(json.dumps(payload, indent=2) + "\n")
    else:
        out.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", out if n_limit < 50 else LOGS / "baseline_official.json", "successes", n_ok, flush=True)
    if n_ok == 0 and rows and rows[0].get("first_plan_raw") is not None:
        print("PLAN_RAW", rows[0]["first_plan_raw"], flush=True)
        print("DATA_RAW", match.get("dataset_action_raw"), flush=True)
        print("DATA_NORM", match.get("dataset_action_norm"), flush=True)
        print("PLAN_NORM0", rows[0].get("first_plan_norm", [None])[0], flush=True)


if __name__ == "__main__":
    main()
