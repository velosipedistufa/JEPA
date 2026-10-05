#!/usr/bin/env python3
"""Frozen-LeWM GPU jobs: image-goal CEM, probe costs, and a z-conditioned skill head.

Does not train LeWM. Does not call eval.py (that path needs the HDF5 dataset).
"""

from __future__ import annotations

import json
import os
import time
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torchvision.transforms import v2 as T

import stable_pretraining as spt
from stable_worldmodel.wm.utils import load_pretrained

LOGS = ROOT / "logs"
FIGS = ROOT / "reports" / "figures_gpu"
HEADS = ROOT / "data" / "heads"
PROBE_NPZ = ROOT / "data" / "probes" / "pusht_theta_probe.npz"

HORIZON = 5
ACTION_BLOCK = 5
ACT_DIM = 2
MODEL_ACT = ACTION_BLOCK * ACT_DIM  # 10, matches the checkpoint action encoder
N_SAMPLES = 300
N_STEPS = 30
TOPK = 30
EVAL_BUDGET = 50
# Live pushes match the frozen predictor best when env actions in [-1, 1]
# are divided by 0.2 before the action encoder (8-trial frameskip check).
ACTION_MODEL_SCALE = 5.0
RIDGE_ALPHA = 1.0
COMMANDS = ("rotate T 90", "rotate T 180", "rotate T -90", "put T on the target")
CMD_DELTA = {
    "rotate T 90": 0.5 * np.pi,
    "rotate T 180": np.pi,
    "rotate T -90": -0.5 * np.pi,
}
TARGET_XY = np.array([256.0, 256.0], dtype=np.float64)
TARGET_ANGLE = 0.25 * np.pi


def wrap_np(angle):
    a = np.asarray(angle, dtype=np.float64)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_t(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def make_transform():
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(size=224),
        ]
    )


def to_pixels(transform, frame: np.ndarray) -> torch.Tensor:
    chw = transform(np.asarray(frame))
    return chw.unsqueeze(0).unsqueeze(0)


def load_model():
    model = load_pretrained("quentinll/lewm-pusht")
    model.eval()
    model.requires_grad_(False)
    model.to("cuda")
    return model


def encode_frame(model, transform, frame) -> torch.Tensor:
    pixels = to_pixels(transform, frame).cuda()
    with torch.inference_mode():
        z = model.encode({"pixels": pixels})["emb"]
    return pixels, z


def rollout_zh(model, pixels: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Env-space actions (S, T, 10) in [-1, 1] -> final latent (S, 192)."""
    actions = actions * ACTION_MODEL_SCALE
    samples, _steps, _ = actions.shape
    pix = pixels.unsqueeze(1).expand(1, samples, *pixels.shape[1:]).contiguous()
    info = {"pixels": pix}
    with torch.inference_mode():
        out = model.rollout(info, actions.unsqueeze(0))
    return out["predicted_emb"][0, :, -1]


def cem(cost_fn, device="cuda") -> tuple[torch.Tensor, float]:
    mean = torch.zeros(HORIZON, MODEL_ACT, device=device)
    std = torch.ones(HORIZON, MODEL_ACT, device=device)
    last = None
    started = time.perf_counter()
    for _step in range(N_STEPS):
        noise = torch.randn(N_SAMPLES, HORIZON, MODEL_ACT, device=device)
        cand = (noise * std.unsqueeze(0) + mean.unsqueeze(0)).clamp(-1.0, 1.0)
        cand[0] = mean.clamp(-1.0, 1.0)
        costs = cost_fn(cand)
        _vals, idx = torch.topk(costs, k=TOPK, largest=False)
        elite = cand[idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(1e-3)
        last = float(_vals.mean())
    return mean, last if last is not None else float("nan")


def plan_to_actions(mean: torch.Tensor) -> np.ndarray:
    actions = mean.view(HORIZON, ACTION_BLOCK, ACT_DIM).reshape(-1, ACT_DIM)
    return actions.clamp(-1.0, 1.0).detach().cpu().numpy().astype(np.float32)


def make_env():
    import gymnasium as gym

    return gym.make("swm/PushT-v1", render_mode="rgb_array")


def obs_state(env):
    return np.asarray(env.unwrapped._get_obs(), dtype=np.float64)


def block_pose(env):
    return np.asarray(env.unwrapped._get_info()["block_pose"], dtype=np.float64)


def success_vs_goal(env) -> dict:
    state = obs_state(env)
    goal = np.asarray(env.unwrapped.goal_state, dtype=np.float64)
    pos = float(np.linalg.norm(goal[:4] - state[:4]))
    ang = float(abs(wrap_np(goal[4] - state[4])))
    return {
        "pos_l2": pos,
        "angle_err_rad": ang,
        "angle_err_deg": float(np.degrees(ang)),
        "success": bool(pos < 20.0 and ang < np.pi / 9),
    }


def image_goal_cost(model, pixels, goal_pixels):
    with torch.inference_mode():
        z_g = model.encode({"pixels": goal_pixels})["emb"][0, -1]

    def cost(actions):
        z_h = rollout_zh(model, pixels, actions)
        return ((z_h - z_g) ** 2).sum(dim=-1)

    return cost


def run_episode_plan(env, model, transform, cost_builder, budget=EVAL_BUDGET):
    """Replan every HORIZON*ACTION_BLOCK steps. cost_builder(pixels, env) -> cost fn."""
    errors = []
    solves = 0
    solve_s = 0.0
    steps = 0
    plan = np.zeros((0, ACT_DIM), dtype=np.float32)
    cursor = 0
    for _ in range(budget):
        if cursor >= len(plan):
            frame = np.asarray(env.render())
            pixels, _z = encode_frame(model, transform, frame)
            started = time.perf_counter()
            mean, _c = cem(cost_builder(pixels, env))
            solve_s += time.perf_counter() - started
            solves += 1
            plan = plan_to_actions(mean)
            cursor = 0
        action = plan[cursor]
        cursor += 1
        _obs, _r, terminated, truncated, _info = env.step(action)
        steps += 1
        stat = success_vs_goal(env)
        errors.append(stat["pos_l2"])
        if stat["success"] or terminated or truncated:
            break
    final = success_vs_goal(env)
    final.update(
        {
            "steps": steps,
            "solves": solves,
            "solve_seconds": solve_s,
            "pos_trace": errors,
        }
    )
    return final


def stage_sanity():
    LOGS.mkdir(parents=True, exist_ok=True)
    transform = make_transform()
    model = load_model()
    env = make_env()
    try:
        _obs, info = env.reset(seed=0)
        frame = np.asarray(env.render())
        goal = np.asarray(info["goal"])
    finally:
        env.close()
    started = time.perf_counter()
    pixels, z = encode_frame(model, transform, frame)
    elapsed = time.perf_counter() - started
    flat = z.detach().float().cpu()
    lines = [
        f"torch {torch.__version__}",
        f"cuda {torch.cuda.is_available()}",
        f"gpu {torch.cuda.get_device_name(0)}",
        f"torch.version.cuda {torch.version.cuda}",
        f"encode_s {elapsed:.6f}",
        f"z_shape {tuple(z.shape)}",
        f"z_mean {float(flat.mean())}",
        f"z_std {float(flat.std(unbiased=False))}",
        f"z_min {float(flat.min())}",
        f"z_max {float(flat.max())}",
        f"frame {frame.shape} {frame.dtype}",
        f"goal {goal.shape} {goal.dtype}",
        f"pixels {tuple(pixels.shape)}",
    ]
    text = "\n".join(lines) + "\n"
    (LOGS / "sanity_gpu.txt").write_text(text)
    print(text, end="")
    if tuple(z.shape) != (1, 1, 192):
        raise SystemExit(f"unexpected z shape {tuple(z.shape)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA became unavailable")
    return model, transform


def stage_baseline(model, transform, n_episodes=20):
    env = make_env()
    rows = []
    t0 = time.perf_counter()
    try:
        for seed in range(n_episodes):
            _obs, info = env.reset(seed=seed)
            goal = np.asarray(info["goal"])
            goal_pixels = to_pixels(transform, goal).cuda()

            def builder(pixels, _env, gp=goal_pixels):
                return image_goal_cost(model, pixels, gp)

            row = run_episode_plan(env, model, transform, builder)
            row["seed"] = seed
            row.pop("pos_trace", None)
            rows.append(row)
            print(
                f"baseline {seed} success={row['success']} pos={row['pos_l2']:.1f} "
                f"ang={row['angle_err_deg']:.1f} solve_s={row['solve_seconds']:.2f}",
                flush=True,
            )
    finally:
        env.close()
    payload = {
        "n": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else None,
        "mean_pos_l2": float(np.mean([r["pos_l2"] for r in rows])) if rows else None,
        "mean_angle_err_deg": float(np.mean([r["angle_err_deg"] for r in rows])) if rows else None,
        "wall_s": time.perf_counter() - t0,
        "protocol": {
            "horizon": HORIZON,
            "action_block": ACTION_BLOCK,
            "num_samples": N_SAMPLES,
            "n_steps": N_STEPS,
            "topk": TOPK,
            "eval_budget": EVAL_BUDGET,
            "cost": "sum of squares between z_H and encode(goal image)",
            "goal": "info['goal'] rendered by PushT.reset, not a language image",
            "dataset": "not used; eval.py requires pusht_expert_train hdf5",
            "action_model_scale": ACTION_MODEL_SCALE,
            "action_scaler": "dataset z-score unavailable; env actions are multiplied by action_model_scale inside the predictor only",
        },
        "episodes": rows,
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    (LOGS / "baseline_cem.json").write_text(text)
    (LOGS / f"baseline_cem_n{n_episodes}.json").write_text(text)
    print("baseline_success_rate", payload["success_rate"], "wall", payload["wall_s"])
    return payload


def episode_split(episode_ids, kinds):
    rng = np.random.default_rng(0)
    test_eps, train_eps = [], []
    for kind in ("rotation", "random_only"):
        eps = np.unique(episode_ids[kinds == kind])
        rng.shuffle(eps)
        n_test = max(1, int(round(len(eps) * 0.25)))
        test_eps.extend(eps[:n_test].tolist())
        train_eps.extend(eps[n_test:].tolist())
    return np.asarray(train_eps, dtype=np.int32), np.asarray(test_eps, dtype=np.int32)


def fit_probes(data):
    train_eps, test_eps = episode_split(data["episode_id"], data["episode_kind"])
    train = np.isin(data["episode_id"], train_eps)
    z = data["z"].astype(np.float64)
    theta = data["theta_rad"].astype(np.float64)
    xy = np.stack([data["x"], data["y"]], axis=1).astype(np.float64)
    ang = Ridge(alpha=RIDGE_ALPHA).fit(z[train], np.stack([np.cos(theta[train]), np.sin(theta[train])], 1))
    loc = Ridge(alpha=RIDGE_ALPHA).fit(z[train], xy[train])
    return ang, loc, train_eps, test_eps


def torch_ridge(model_sk, device):
    weight = torch.tensor(model_sk.coef_, dtype=torch.float32, device=device)
    bias = torch.tensor(model_sk.intercept_, dtype=torch.float32, device=device)
    return weight, bias


def apply_start(env, row):
    env.reset(seed=int(row["episode_id"]) + 17)
    state = np.array(
        [
            row["agent_x"],
            row["agent_y"],
            row["x"],
            row["y"],
            row["raw_angle_rad"],
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    env.unwrapped._set_state(state)
    pose = block_pose(env)
    return float(wrap_np(pose[2])), pose[:2].copy()


def probe_cost_fn(model, pixels, ang_w, ang_b, loc_w, loc_b, command, theta_now):
    if command == "put T on the target":
        target = torch.tensor(TARGET_XY, dtype=torch.float32, device=pixels.device)

        def cost(actions):
            z_h = rollout_zh(model, pixels, actions)
            xy = z_h @ loc_w.T + loc_b
            return ((xy - target) ** 2).sum(dim=-1)

        return cost

    delta = float(CMD_DELTA[command])
    theta_now_t = torch.tensor(theta_now, dtype=torch.float32, device=pixels.device)

    def cost(actions):
        z_h = rollout_zh(model, pixels, actions)
        cs = z_h @ ang_w.T + ang_b
        theta_hat = torch.atan2(cs[:, 1], cs[:, 0])
        err = wrap_t(theta_hat - theta_now_t - delta)
        return err ** 2

    return cost


def language_metrics(env, command, theta_now, xy_now):
    pose = block_pose(env)
    theta = float(wrap_np(pose[2]))
    xy = pose[:2]
    if command == "put T on the target":
        ang = float(abs(wrap_np(theta - TARGET_ANGLE)))
        xy_err = float(np.linalg.norm(xy - TARGET_XY))
        ok = bool(xy_err < 20.0 and ang < np.pi / 9)
    else:
        target = float(wrap_np(theta_now + CMD_DELTA[command]))
        ang = float(abs(wrap_np(theta - target)))
        xy_err = float(np.linalg.norm(xy - xy_now))
        ok = bool(ang < np.pi / 9)
    return {
        "angle_err_rad": ang,
        "angle_err_deg": float(np.degrees(ang)),
        "xy_err": xy_err,
        "reached": ok,
        "theta_deg": float(np.degrees(theta)),
        "xy": [float(xy[0]), float(xy[1])],
    }


def run_language_episode(env, model, transform, row, command, cost_builder):
    theta_now, xy_now = apply_start(env, row)
    started = time.perf_counter()

    def builder(pixels, _env):
        return cost_builder(pixels, command, theta_now)

    # Language tasks are scored on angle/xy, not the env's random goal_state.
    trace = []
    plan = np.zeros((0, ACT_DIM), dtype=np.float32)
    cursor = 0
    solves = 0
    solve_s = 0.0
    for _ in range(EVAL_BUDGET):
        if cursor >= len(plan):
            frame = np.asarray(env.render())
            pixels, _z = encode_frame(model, transform, frame)
            t1 = time.perf_counter()
            mean, _c = cem(builder(pixels, env))
            solve_s += time.perf_counter() - t1
            solves += 1
            plan = plan_to_actions(mean)
            cursor = 0
        _obs, _r, term, trunc, _info = env.step(plan[cursor])
        cursor += 1
        if term or trunc:
            break
    metrics = language_metrics(env, command, theta_now, xy_now)
    metrics.update(
        {
            "episode_id": int(row["episode_id"]),
            "command": command,
            "theta_now_deg": float(np.degrees(theta_now)),
            "solves": solves,
            "solve_seconds": solve_s,
            "wall_s": time.perf_counter() - started,
            "trace_unused": trace,
        }
    )
    metrics.pop("trace_unused")
    return metrics


def stage_probe(model, transform):
    data = np.load(PROBE_NPZ, allow_pickle=False)
    ang, loc, _train_eps, test_eps = fit_probes(data)
    device = "cuda"
    ang_w, ang_b = torch_ridge(ang, device)
    loc_w, loc_b = torch_ridge(loc, device)
    kind = data["episode_kind"]
    eid = data["episode_id"]
    tag = data["tag"]
    rot_test = []
    for ep in test_eps.tolist():
        idx = np.where((eid == ep) & (kind == "rotation") & (tag == "random"))[0]
        if len(idx) == 1:
            rot_test.append(int(idx[0]))
    rows = []
    env = make_env()
    commands = ("rotate T 90", "rotate T 180", "rotate T -90", "put T on the target")
    # Cap put-T to the same episodes. Rotation commands use the scripted start.
    t0 = time.perf_counter()
    try:
        for index in rot_test:
            row = {key: data[key][index].item() if data[key].ndim == 1 else data[key][index] for key in (
                "episode_id", "agent_x", "agent_y", "x", "y", "raw_angle_rad"
            )}
            for command in commands:
                def cost_builder(pixels, cmd, theta_now, _row=row):
                    return probe_cost_fn(model, pixels, ang_w, ang_b, loc_w, loc_b, cmd, theta_now)

                metrics = run_language_episode(env, model, transform, row, command, cost_builder)
                rows.append(metrics)
                print(
                    f"probe ep={metrics['episode_id']} {command} "
                    f"ang={metrics['angle_err_deg']:.1f} xy={metrics['xy_err']:.1f} "
                    f"ok={metrics['reached']}",
                    flush=True,
                )
    finally:
        env.close()
    summary = _summarize_language(rows, time.perf_counter() - t0)
    summary["n_test_episodes"] = len(rot_test)
    summary["method"] = "ridge probe cost on imagined z_H"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "probe_cost_cem.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_language(rows, FIGS / "probe_cost_angle_error.pdf", "Probe-cost CEM")
    print("probe_cost written", summary["by_command"])
    return summary, data, ang, loc, rot_test


def _summarize_language(rows, wall_s):
    by = {}
    for command in COMMANDS:
        chosen = [r for r in rows if r["command"] == command]
        if not chosen:
            by[command] = {"n": 0, "status": "not measured"}
            continue
        by[command] = {
            "n": len(chosen),
            "mean_angle_err_deg": float(np.mean([r["angle_err_deg"] for r in chosen])),
            "median_angle_err_deg": float(np.median([r["angle_err_deg"] for r in chosen])),
            "mean_xy_err": float(np.mean([r["xy_err"] for r in chosen])),
            "reach_rate": float(np.mean([r["reached"] for r in chosen])),
            "mean_solve_seconds": float(np.mean([r["solve_seconds"] for r in chosen])),
        }
    return {"wall_s": wall_s, "by_command": by, "episodes": rows}


def _plot_language(rows, path: Path, title: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    labels, means = [], []
    for command in COMMANDS:
        chosen = [r["angle_err_deg"] for r in rows if r["command"] == command]
        if not chosen:
            continue
        labels.append(command)
        means.append(float(np.mean(chosen)))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(range(len(means)), means, color="#1f4e79")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("mean final |angle error| (deg)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


class SkillHead(nn.Module):
    def __init__(self, z_dim=192, n_cmd=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + n_cmd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, e):
        return self.net(torch.cat([z, e], dim=-1))


def one_hot(indices, n=4, device="cuda"):
    e = torch.zeros(*indices.shape, n, device=device)
    return e.scatter_(-1, indices.unsqueeze(-1), 1.0)


def collect_extra(env, model, transform, n_episodes=200, seed0=50_000):
    records = []
    pending_frames = []
    pending = []

    def flush():
        if not pending_frames:
            return
        pixels = torch.cat([to_pixels(transform, f) for f in pending_frames], 0).cuda()
        with torch.inference_mode():
            emb = model.encode({"pixels": pixels})["emb"][:, 0].detach().cpu().numpy()
        for vec, item in zip(emb, pending):
            item["z"] = vec.astype(np.float32)
            records.append(item)
        pending_frames.clear()
        pending.clear()

    for i in range(n_episodes):
        _obs, info = env.reset(seed=seed0 + i)
        base = np.asarray(info["block_pose"], dtype=np.float64)
        agent = np.asarray(info["pos_agent"], dtype=np.float64)
        vel = np.asarray(info["vel_agent"], dtype=np.float64)
        base_state = np.array(
            [agent[0], agent[1], base[0], base[1], base[2], vel[0], vel[1]],
            dtype=np.float64,
        )
        shots = [("random", np.asarray(env.render()), 0.0)]
        for name, delta, cmd_i in (
            ("rot90", 0.5 * np.pi, 0),
            ("rot180", np.pi, 1),
            ("rot_m90", -0.5 * np.pi, 2),
        ):
            state = base_state.copy()
            state[4] = base[2] + delta
            state[5:] = 0.0
            env.unwrapped._set_state(state)
            shots.append((name, np.asarray(env.render()), cmd_i))
        for name, frame, cmd_i in shots:
            pending_frames.append(frame)
            pending.append({"ep": i, "name": name, "cmd": cmd_i})
        if len(pending_frames) >= 32:
            flush()
        if (i + 1) % 50 == 0:
            print(f"extra_episodes {i + 1}/{n_episodes}", flush=True)
    flush()
    by_ep = {}
    for item in records:
        by_ep.setdefault(item["ep"], {})[item["name"]] = item
    z_t, z_h, cmd = [], [], []
    for group in by_ep.values():
        if "random" not in group:
            continue
        base_z = group["random"]["z"]
        for name, cmd_i in (("rot90", 0), ("rot180", 1), ("rot_m90", 2)):
            if name not in group:
                continue
            z_t.append(base_z)
            z_h.append(group[name]["z"])
            cmd.append(cmd_i)
    return (
        np.stack(z_t).astype(np.float32),
        np.stack(z_h).astype(np.float32),
        np.asarray(cmd, dtype=np.int64),
    )


def existing_pairs(data, train_eps):
    z = data["z"].astype(np.float32)
    eid = data["episode_id"]
    tag = data["tag"]
    kind = data["episode_kind"]
    tag_cmd = {"rot90": 0, "rot180": 1, "rot_m90": 2}
    z_t, z_h, cmd = [], [], []
    train_set = set(int(x) for x in train_eps.tolist())
    for ep in np.unique(eid):
        if int(ep) not in train_set:
            continue
        idx = np.where(eid == ep)[0]
        base = [i for i in idx if tag[i] == "random"]
        if len(base) != 1:
            continue
        if not np.any(kind[idx] == "rotation"):
            continue
        b = base[0]
        for i in idx:
            c = tag_cmd.get(str(tag[i]))
            if c is None:
                continue
            z_t.append(z[b])
            z_h.append(z[i])
            cmd.append(c)
    if not z_t:
        return (
            np.zeros((0, 192), np.float32),
            np.zeros((0, 192), np.float32),
            np.zeros((0,), np.int64),
        )
    return np.stack(z_t), np.stack(z_h), np.asarray(cmd, dtype=np.int64)


def train_head(z_t, z_h, cmd, steps=400):
    device = "cuda"
    head = SkillHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    zt = torch.tensor(z_t, device=device)
    zh = torch.tensor(z_h, device=device)
    c = torch.tensor(cmd, device=device)
    n = zt.shape[0]
    losses = []
    for step in range(steps):
        sel = torch.randint(0, n, (256,), device=device)
        e = one_hot(c[sel])
        pred = head(zt[sel], e)
        loss = ((pred - (zh[sel] - zt[sel])) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            losses.append({"step": step, "mse": float(loss.detach())})
            print(f"skill_step {step} mse {float(loss.detach()):.5f}", flush=True)
    head.eval()
    return head, losses


def stage_skill(model, transform, data, test_rows_index):
    _ang, _loc, train_eps, _test_eps = fit_probes(data)
    base_t, base_h, base_c = existing_pairs(data, train_eps)
    env = make_env()
    try:
        extra_t, extra_h, extra_c = collect_extra(env, model, transform, n_episodes=200)
    finally:
        env.close()
    z_t = np.concatenate([base_t, extra_t], 0)
    z_h = np.concatenate([base_h, extra_h], 0)
    cmd = np.concatenate([base_c, extra_c], 0)
    print(f"skill_pairs {len(cmd)} existing {len(base_c)} extra {len(extra_c)}", flush=True)
    head, losses = train_head(z_t, z_h, cmd)
    HEADS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": head.state_dict(), "commands": COMMANDS, "predicts": "delta_z"},
        HEADS / "skill_mlp.pt",
    )
    # held-out pair residual on original test rotation pairs
    test_eps = _test_eps
    # rebuild test pairs by flipping the filter: call existing_pairs logic inline
    hold_t, hold_h, hold_c = _pairs_for_episodes(data, test_eps)
    with torch.inference_mode():
        if len(hold_c):
            e = one_hot(torch.tensor(hold_c, device="cuda"))
            pred = head(torch.tensor(hold_t, device="cuda"), e)
            target = torch.tensor(hold_h - hold_t, device="cuda")
            hold_mse = float(((pred - target) ** 2).mean())
            hold_l2 = float((pred - target).norm(dim=-1).mean())
            target_l2 = float(target.norm(dim=-1).mean())
        else:
            hold_mse = hold_l2 = target_l2 = None

    def cost_builder(pixels, command, theta_now):
        del theta_now
        cmd_i = COMMANDS.index(command)
        with torch.inference_mode():
            z_t_now = model.encode({"pixels": pixels})["emb"][0, -1]
        e = one_hot(torch.tensor([cmd_i], device="cuda"))[0]

        def cost(actions):
            z_h = rollout_zh(model, pixels, actions)
            delta = head(z_t_now.expand(z_h.shape[0], -1), e.expand(z_h.shape[0], -1))
            return ((z_h - z_t_now - delta) ** 2).sum(-1)

        return cost

    rows = []
    env = make_env()
    t0 = time.perf_counter()
    try:
        for index in test_rows_index:
            row = {
                key: data[key][index].item()
                for key in ("episode_id", "agent_x", "agent_y", "x", "y", "raw_angle_rad")
            }
            for command in ("rotate T 90", "rotate T 180", "rotate T -90"):
                metrics = run_language_episode(env, model, transform, row, command, cost_builder)
                rows.append(metrics)
                print(
                    f"skill ep={metrics['episode_id']} {command} "
                    f"ang={metrics['angle_err_deg']:.1f} xy={metrics['xy_err']:.1f} "
                    f"ok={metrics['reached']}",
                    flush=True,
                )
    finally:
        env.close()
    summary = _summarize_language(rows, time.perf_counter() - t0)
    summary["method"] = "CEM chases pred delta z = MLP(z_t, one_hot)"
    summary["n_train_pairs"] = int(len(cmd))
    summary["n_existing_train_pairs"] = int(len(base_c))
    summary["n_extra_pairs"] = int(len(extra_c))
    summary["train_loss_trace"] = losses
    summary["heldout_pair_mse"] = hold_mse
    summary["heldout_pair_mean_l2"] = hold_l2
    summary["heldout_target_mean_l2"] = target_l2
    summary["put_T_on_the_target"] = "not measured; no same-xy pairs for that command"
    (LOGS / "skill_head_cem.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_language(rows, FIGS / "skill_head_angle_error.pdf", "Skill-head CEM")
    _compare_plot()
    print("skill written", summary["by_command"])
    return summary


def _pairs_for_episodes(data, episodes):
    z = data["z"].astype(np.float32)
    eid = data["episode_id"]
    tag = data["tag"]
    wanted = set(int(x) for x in episodes.tolist())
    tag_cmd = {"rot90": 0, "rot180": 1, "rot_m90": 2}
    z_t, z_h, cmd = [], [], []
    for ep in np.unique(eid):
        if int(ep) not in wanted:
            continue
        idx = np.where(eid == ep)[0]
        base = [i for i in idx if tag[i] == "random"]
        if len(base) != 1:
            continue
        b = base[0]
        for i in idx:
            c = tag_cmd.get(str(tag[i]))
            if c is None:
                continue
            z_t.append(z[b])
            z_h.append(z[i])
            cmd.append(c)
    if not z_t:
        return (
            np.zeros((0, 192), np.float32),
            np.zeros((0, 192), np.float32),
            np.zeros((0,), np.int64),
        )
    return np.stack(z_t), np.stack(z_h), np.asarray(cmd, np.int64)


def _compare_plot():
    probe_path = LOGS / "probe_cost_cem.json"
    skill_path = LOGS / "skill_head_cem.json"
    if not probe_path.is_file() or not skill_path.is_file():
        return
    probe = json.loads(probe_path.read_text())
    skill = json.loads(skill_path.read_text())
    labels = ["rotate T 90", "rotate T 180", "rotate T -90"]
    p = [probe["by_command"][k]["mean_angle_err_deg"] for k in labels]
    s = [skill["by_command"][k]["mean_angle_err_deg"] for k in labels]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, p, width=0.36, label="probe cost", color="#1f4e79")
    ax.bar(x + 0.18, s, width=0.36, label="skill head", color="#c44536")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean final |angle error| (deg)")
    ax.set_title("Same held-out scripted starts")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "probe_vs_skill_angle.pdf")
    plt.close(fig)


def test_rotation_indices(data):
    _ang, _loc, _train, test_eps = fit_probes(data)
    eid = data["episode_id"]
    kind = data["episode_kind"]
    tag = data["tag"]
    out = []
    for ep in test_eps.tolist():
        idx = np.where((eid == ep) & (kind == "rotation") & (tag == "random"))[0]
        if len(idx) == 1:
            out.append(int(idx[0]))
    return out


def main():
    import sys

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    model, transform = stage_sanity()
    if stage == "sanity":
        return
    if stage in ("baseline", "all"):
        small = stage_baseline(model, transform, n_episodes=10)
        if small["wall_s"] < 900:
            stage_baseline(model, transform, n_episodes=20)
            # 50 only if the 20-episode run stays inside the same budget scale
            per = small["wall_s"] / max(small["n"], 1)
            if per * 50 < 2400:
                stage_baseline(model, transform, n_episodes=50)
    if stage in ("probe", "all"):
        summary, data, _ang, _loc, indices = stage_probe(model, transform)
        del summary
    else:
        data = np.load(PROBE_NPZ, allow_pickle=False)
        indices = test_rotation_indices(data)
    if stage in ("skill", "all"):
        stage_skill(model, transform, data, indices)


if __name__ == "__main__":
    main()
