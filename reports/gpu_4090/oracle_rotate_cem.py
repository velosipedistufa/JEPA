#!/usr/bin/env python3
"""In-place rotation CEM with a rendered goal image. Same planner as eval_official_cem."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

import eval_official_cem as official

LOGS = ROOT / "logs"
HEADS = ROOT / "data" / "heads"
PROBE = ROOT / "data" / "probes" / "pusht_theta_probe.npz"
DELTAS = (("rotate_90", 0.5 * np.pi), ("rotate_180", np.pi))
XY_LIMIT = 15.0
ANGLE_LIMIT = np.pi / 9.0


def wrap_diff(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def episode_split(episode_ids, kinds):
    rng = np.random.default_rng(0)
    test_eps, train_eps = [], []
    for kind in ("rotation", "random_only"):
        eps = np.unique(episode_ids[kinds == kind])
        rng.shuffle(eps)
        n_test = max(1, int(round(len(eps) * 0.25)))
        test_eps.extend(int(x) for x in eps[:n_test])
        train_eps.extend(int(x) for x in eps[n_test:])
    return train_eps, test_eps


def test_starts(data):
    train_eps, test_eps = episode_split(data["episode_id"], data["episode_kind"])
    eid = data["episode_id"]
    kind = data["episode_kind"]
    tag = data["tag"]
    rows = []
    for ep in test_eps:
        hit = np.where((eid == ep) & (kind == "rotation") & (tag == "random"))[0]
        if len(hit) == 1:
            i = int(hit[0])
            rows.append(
                {
                    "episode_id": int(data["episode_id"][i]),
                    "agent_x": float(data["agent_x"][i]),
                    "agent_y": float(data["agent_y"][i]),
                    "x": float(data["x"][i]),
                    "y": float(data["y"][i]),
                    "raw_angle_rad": float(data["raw_angle_rad"][i]),
                }
            )
    return train_eps, rows


def apply_state(env, row, angle, vel=(0.0, 0.0)):
    state = np.array(
        [
            row["agent_x"],
            row["agent_y"],
            row["x"],
            row["y"],
            angle,
            vel[0],
            vel[1],
        ],
        dtype=np.float64,
    )
    env.unwrapped._set_state(state)
    obs = env.unwrapped._get_obs()
    return obs


def encode_render(model, transform, frame):
    pixels = official.to_pixels(transform, official.hwc(frame)).unsqueeze(0).cuda()
    with torch.inference_mode():
        z = model.encode({"pixels": pixels})["emb"][0, -1]
    return pixels, z


def coverage_of(env):
    info = env.unwrapped._get_info()
    found = {}
    for key, value in info.items():
        if "cover" in key.lower():
            if isinstance(value, np.ndarray):
                found[key] = value.tolist() if value.size <= 8 else {"shape": list(value.shape)}
            else:
                found[key] = value if isinstance(value, (int, float, str)) else str(type(value))
    if hasattr(env.unwrapped, "coverage_arr"):
        arr = np.asarray(env.unwrapped.coverage_arr)
        found["coverage_arr_len"] = int(arr.size)
    return found or None


def run_cem_budget(env, model, transform, mean, std, z_g, budget):
    executed = 0
    best_ang = None
    while executed < budget:
        pixels, _z = encode_render(model, transform, np.asarray(env.render()))

        def cost(cand, pixels=pixels):
            z_h = official.rollout_last(model, pixels, cand)
            return ((z_h - z_g) ** 2).sum(-1)

        plan = official.cem(cost).detach().cpu().numpy()
        actions = official.denormalize_plan(plan, mean, std)
        for action in actions:
            if executed >= budget:
                break
            env.step(action)
            executed += 1
            obs = env.unwrapped._get_obs()
            # best tracked by caller after the fact; keep stepping the full budget
            del obs
    return executed


def measure(env, theta0, xy0, delta):
    obs = env.unwrapped._get_obs()
    ang = abs(wrap_diff(float(obs[4]), theta0 + delta))
    xy = float(np.linalg.norm(obs[2:4] - xy0))
    return {
        "angle_err_deg": float(np.degrees(ang)),
        "xy_err": xy,
        "reached": bool(ang < ANGLE_LIMIT and xy < XY_LIMIT),
        "coverage": coverage_of(env),
    }


def summarize(rows):
    out = {}
    for row in rows:
        key = f"{row['command']}_budget{row['budget']}"
        out.setdefault(key, []).append(row)
    table = {}
    for key, group in out.items():
        table[key] = {
            "n": len(group),
            "mean_abs_dtheta_deg": float(np.mean([r["angle_err_deg"] for r in group])),
            "mean_dxy": float(np.mean([r["xy_err"] for r in group])),
            "reach_rate": float(np.mean([r["reached"] for r in group])),
            "wall_s": float(np.sum([r["wall_s"] for r in group])),
            "mean_zg_zt_l2": float(np.mean([r["zg_zt_l2"] for r in group])),
        }
    return table


def oracle(model, transform, mean, std):
    data = np.load(PROBE)
    _train, starts = test_starts(data)
    if len(starts) != 20:
        print("warning_n_starts", len(starts), flush=True)
    env = official.make_env()
    rows = []
    try:
        for row in starts:
            env.reset(seed=0)
            apply_state(env, row, row["raw_angle_rad"])
            theta0 = float(env.unwrapped._get_obs()[4])
            xy0 = env.unwrapped._get_obs()[2:4].copy()
            _pix_t, z_t = encode_render(model, transform, np.asarray(env.render()))
            goals = {}
            for name, delta in DELTAS:
                apply_state(env, row, row["raw_angle_rad"] + delta)
                goal_obs = env.unwrapped._get_obs()
                pix_g, z_g = encode_render(model, transform, np.asarray(env.render()))
                goals[name] = {
                    "delta": delta,
                    "z_g": z_g,
                    "zg_zt_l2": float((z_g - z_t).norm()),
                    "goal_dxy": float(np.linalg.norm(goal_obs[2:4] - xy0)),
                    "goal_dtheta_deg": float(
                        np.degrees(abs(wrap_diff(float(goal_obs[4]), theta0 + delta)))
                    ),
                }
            jobs = [
                ("rotate_90", 50),
                ("rotate_180", 50),
                ("rotate_180", 100),
            ]
            for name, budget in jobs:
                spec = goals[name]
                t0 = time.perf_counter()
                apply_state(env, row, row["raw_angle_rad"])
                run_cem_budget(env, model, transform, mean, std, spec["z_g"], budget)
                metrics = measure(env, theta0, xy0, spec["delta"])
                metrics.update(
                    {
                        "episode_id": row["episode_id"],
                        "command": name,
                        "budget": budget,
                        "zg_zt_l2": spec["zg_zt_l2"],
                        "goal_render_dxy": spec["goal_dxy"],
                        "goal_render_dtheta_deg": spec["goal_dtheta_deg"],
                        "wall_s": time.perf_counter() - t0,
                    }
                )
                rows.append(metrics)
                print(
                    f"oracle ep={row['episode_id']} {name} b={budget} "
                    f"ang={metrics['angle_err_deg']:.1f} xy={metrics['xy_err']:.1f} "
                    f"ok={metrics['reached']} zg={spec['zg_zt_l2']:.2f}",
                    flush=True,
                )
    finally:
        env.close()
    table = summarize(rows)
    payload = {
        "n_starts": len(starts),
        "table": table,
        "rows": rows,
        "decision": None,
    }
    rate_180_100 = table.get("rotate_180_budget100", {}).get("reach_rate", 0.0)
    if rate_180_100 > 0.30:
        payload["decision"] = "oracle 180 at budget 100 is above 0.30; a z_g MLP is allowed"
    else:
        payload["decision"] = (
            "CEM cannot execute in-place 180° even with a true goal image under this horizon/budget."
        )
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "oracle_rotate.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (LOGS / "oracle_rotate.csv").open("w", newline="") as handle:
        fields = [
            "episode_id",
            "command",
            "budget",
            "angle_err_deg",
            "xy_err",
            "reached",
            "zg_zt_l2",
            "goal_render_dxy",
            "goal_render_dtheta_deg",
            "wall_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("TABLE", json.dumps(table), flush=True)
    print("DECISION", payload["decision"], flush=True)
    return payload, data


class ZGMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(192 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 192),
        )

    def forward(self, z, e):
        return self.net(torch.cat([z, e], dim=-1))


def collect_pairs(env, model, transform, n_extra=250):
    """Same-xy turns. Returns z_t, z_g, command index 0=+90 1=+180."""
    zs_t, zs_g, cmd = [], [], []
    frames = []
    meta = []

    def flush():
        if not frames:
            return
        batch = torch.cat(
            [official.to_pixels(transform, official.hwc(f)).unsqueeze(0) for f in frames],
            0,
        ).cuda()
        with torch.inference_mode():
            emb = model.encode({"pixels": batch})["emb"][:, 0].detach().cpu()
        for vec, item in zip(emb, meta):
            item["z"] = vec
        frames.clear()
        meta.clear()

    data = np.load(PROBE)
    train_eps, _starts = test_starts(data)
    eid = data["episode_id"]
    kind = data["episode_kind"]
    tag = data["tag"]
    for ep in train_eps:
        hit = np.where((eid == ep) & (kind == "rotation") & (tag == "random"))[0]
        if len(hit) != 1:
            continue
        i = int(hit[0])
        row = {
            "agent_x": float(data["agent_x"][i]),
            "agent_y": float(data["agent_y"][i]),
            "x": float(data["x"][i]),
            "y": float(data["y"][i]),
            "raw_angle_rad": float(data["raw_angle_rad"][i]),
        }
        env.reset(seed=0)
        apply_state(env, row, row["raw_angle_rad"])
        shots = [("t", np.asarray(env.render()), None)]
        for name, delta, idx in (("g90", 0.5 * np.pi, 0), ("g180", np.pi, 1)):
            apply_state(env, row, row["raw_angle_rad"] + delta)
            shots.append((name, np.asarray(env.render()), idx))
        for name, frame, idx in shots:
            frames.append(frame)
            meta.append({"ep": ep, "name": name, "cmd": idx})
        if len(frames) >= 24:
            flush()
    flush()
    # The flush stored z on dicts that were appended then cleared from meta,
    # but the dict objects in the local shots loop were only in meta.
    # Re-collect properly below if the above lost references.
    return _collect_clean(env, model, transform, n_extra)


def _collect_clean(env, model, transform, n_extra):
    data = np.load(PROBE)
    train_eps, _starts = test_starts(data)
    records = []

    def encode_many(frames):
        if not frames:
            return []
        batch = torch.cat(
            [official.to_pixels(transform, official.hwc(f)).unsqueeze(0) for f in frames],
            0,
        ).cuda()
        with torch.inference_mode():
            emb = model.encode({"pixels": batch})["emb"][:, 0].detach().cpu().numpy()
        return list(emb)

    pending_f, pending_m = [], []

    def flush():
        vecs = encode_many(pending_f)
        for vec, item in zip(vecs, pending_m):
            item["z"] = vec
            records.append(item)
        pending_f.clear()
        pending_m.clear()

    for ep in train_eps:
        eid = data["episode_id"]
        kind = data["episode_kind"]
        tag = data["tag"]
        hit = np.where((eid == ep) & (kind == "rotation") & (tag == "random"))[0]
        if len(hit) != 1:
            continue
        i = int(hit[0])
        row = {
            "agent_x": float(data["agent_x"][i]),
            "agent_y": float(data["agent_y"][i]),
            "x": float(data["x"][i]),
            "y": float(data["y"][i]),
            "raw_angle_rad": float(data["raw_angle_rad"][i]),
        }
        env.reset(seed=1)
        apply_state(env, row, row["raw_angle_rad"])
        shots = [(np.asarray(env.render()), "t", None)]
        apply_state(env, row, row["raw_angle_rad"] + 0.5 * np.pi)
        shots.append((np.asarray(env.render()), "g", 0))
        apply_state(env, row, row["raw_angle_rad"] + np.pi)
        shots.append((np.asarray(env.render()), "g", 1))
        for frame, kind_name, cmd in shots:
            pending_f.append(frame)
            pending_m.append({"ep": int(ep), "kind": kind_name, "cmd": cmd})
        if len(pending_f) >= 24:
            flush()
    # extra random resets
    for i in range(n_extra):
        _obs, info = env.reset(seed=80_000 + i)
        pose = np.asarray(info["block_pose"], dtype=np.float64)
        agent = np.asarray(info["pos_agent"], dtype=np.float64)
        row = {
            "agent_x": float(agent[0]),
            "agent_y": float(agent[1]),
            "x": float(pose[0]),
            "y": float(pose[1]),
            "raw_angle_rad": float(pose[2]),
        }
        shots = [(np.asarray(env.render()), "t", None)]
        apply_state(env, row, row["raw_angle_rad"] + 0.5 * np.pi)
        shots.append((np.asarray(env.render()), "g", 0))
        apply_state(env, row, row["raw_angle_rad"] + np.pi)
        shots.append((np.asarray(env.render()), "g", 1))
        for frame, kind_name, cmd in shots:
            pending_f.append(frame)
            pending_m.append({"ep": 100_000 + i, "kind": kind_name, "cmd": cmd})
        if len(pending_f) >= 24:
            flush()
    flush()
    by = {}
    for item in records:
        by.setdefault(item["ep"], {})[item["kind"] if item["cmd"] is None else f"g{item['cmd']}"] = item
    # rebuild with explicit keys
    grouped = {}
    for item in records:
        slot = grouped.setdefault(item["ep"], {})
        if item["cmd"] is None:
            slot["t"] = item["z"]
        else:
            slot[item["cmd"]] = item["z"]
    zt, zg, cc = [], [], []
    for slot in grouped.values():
        if "t" not in slot:
            continue
        for cmd in (0, 1):
            if cmd not in slot:
                continue
            zt.append(slot["t"])
            zg.append(slot[cmd])
            cc.append(cmd)
    return np.stack(zt).astype(np.float32), np.stack(zg).astype(np.float32), np.asarray(cc, np.int64)


def train_mlp(z_t, z_g, cmd, steps=2000):
    net = ZGMLP().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    zt = torch.tensor(z_t, device="cuda")
    zg = torch.tensor(z_g, device="cuda")
    c = torch.tensor(cmd, device="cuda")
    n = zt.shape[0]
    trace = []
    for step in range(steps):
        sel = torch.randint(0, n, (256,), device="cuda")
        e = torch.nn.functional.one_hot(c[sel], 2).float()
        pred = net(zt[sel], e)
        loss = ((pred - zg[sel]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            trace.append({"step": step, "mse": float(loss.detach())})
            print(f"zg_step {step} mse {float(loss.detach()):.5f}", flush=True)
    net.eval()
    return net, trace


def pred_zg_cem(model, transform, mean, std, net):
    data = np.load(PROBE)
    _train, starts = test_starts(data)
    env = official.make_env()
    rows = []
    try:
        for row in starts:
            for cmd, (name, delta) in enumerate(DELTAS):
                for budget in (50, 100):
                    env.reset(seed=0)
                    apply_state(env, row, row["raw_angle_rad"])
                    theta0 = float(env.unwrapped._get_obs()[4])
                    xy0 = env.unwrapped._get_obs()[2:4].copy()
                    _pix, z_t = encode_render(model, transform, np.asarray(env.render()))
                    e = torch.nn.functional.one_hot(torch.tensor([cmd], device="cuda"), 2).float()
                    with torch.inference_mode():
                        z_hat = net(z_t.view(1, -1), e)[0]
                    t0 = time.perf_counter()
                    run_cem_budget(env, model, transform, mean, std, z_hat, budget)
                    metrics = measure(env, theta0, xy0, delta)
                    metrics.update(
                        {
                            "episode_id": row["episode_id"],
                            "command": name,
                            "budget": budget,
                            "wall_s": time.perf_counter() - t0,
                            "zg_zt_l2": float((z_hat - z_t).norm()),
                        }
                    )
                    rows.append(metrics)
                    print(
                        f"pred ep={row['episode_id']} {name} b={budget} "
                        f"ang={metrics['angle_err_deg']:.1f} ok={metrics['reached']}",
                        flush=True,
                    )
    finally:
        env.close()
    table = summarize(rows)
    payload = {"table": table, "rows": rows}
    (LOGS / "pred_zg_cem.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("PRED_TABLE", json.dumps(table), flush=True)
    return payload


def main():
    print("torch_check", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    ds = official.open_dataset()
    mean, std, stats = official.action_stats(ds)
    del ds
    model = official.load_model()
    transform = official.make_transform()
    # one-frame encode
    env = official.make_env()
    env.reset(seed=0)
    pix, z = encode_render(model, transform, np.asarray(env.render()))
    env.close()
    print("sanity_z", tuple(pix.shape), tuple(z.shape), float(z.mean()), flush=True)
    payload, _data = oracle(model, transform, mean, std)
    rate = payload["table"].get("rotate_180_budget100", {}).get("reach_rate", 0.0)
    (LOGS / "oracle_action_stats.json").write_text(json.dumps(stats) + "\n")
    if rate > 0.30:
        env = official.make_env()
        try:
            z_t, z_g, cmd = _collect_clean(env, model, transform, n_extra=250)
        finally:
            env.close()
        print("pairs", len(cmd), flush=True)
        if len(cmd) < 500:
            raise SystemExit(f"only {len(cmd)} pairs")
        net, trace = train_mlp(z_t, z_g, cmd)
        HEADS.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "commands": ["rotate_90", "rotate_180"]}, HEADS / "zg_mlp.pt")
        pred = pred_zg_cem(model, transform, mean, std, net)
        pred["train_trace"] = trace
        pred["n_pairs"] = int(len(cmd))
        (LOGS / "pred_zg_cem.json").write_text(json.dumps(pred, indent=2) + "\n")
    else:
        (LOGS / "pred_zg_cem.json").write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": payload["decision"],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
