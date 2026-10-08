#!/usr/bin/env python3
"""Checks 1 and 2 of HANDOFF_5090.md: flat CEM against demo-breadcrumb CEM.

Flat: the official planner chases one goal image and replans every 25 steps,
like eval_official_cem, but with a longer budget.

Crumbs: a graph over expert demo states (train episodes only). Nodes are demo
states every 5 steps. Edges follow each demo forward, and also link states from
different demos that are close. From the current state the graph gives a chain
of real demo states to the goal. CEM chases a rendered frame of the state about
25 demo steps down that chain, then the graph is read again from wherever the
block ended up. The last hop chases the goal image itself.

LeWM stays frozen. Same F13 z-score, same CEM 300/30/top 30, horizon 5,
action block 5. Retrieval uses the env state (agent xy, block xy, angle). On a
real robot that state would come from a tag on the block.

  python scripts/demo_breadcrumbs.py --task dataset --offset 50 --planner flat --n 50
  python scripts/demo_breadcrumbs.py --task dataset --offset 50 --planner crumbs --n 50
  python scripts/demo_breadcrumbs.py --task live --planner crumbs --n 20
  python scripts/demo_breadcrumbs.py --task live --dry --n 20     # graph only, no GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

OUT = ROOT / "research" / "artifacts" / "5090"

STRIDE = 5  # one graph node every 5 demo steps
HOP_STEPS = 25  # one CEM plan is horizon 5 x action block 5 env steps
ANGLE_PX = 60.0  # 1 rad of block angle counts as 60 px in the state distance
JUMP_RADIUS = 15.0  # demo states from different episodes closer than this are linked
JUMP_K = 8
ENTRY_K = 16
GOAL_RADIUS = 20.0
TEST_FRACTION = 0.1
SPLIT_SEED = 0
TASK_SEED = 42
TORCH_SEED = 42


def features(states) -> np.ndarray:
    """(N, 7) env states -> (N, 6): agent xy, block xy, angle on a circle of ANGLE_PX."""
    s = np.asarray(states, dtype=np.float64).reshape(-1, np.shape(states)[-1])
    return np.stack(
        [s[:, 0], s[:, 1], s[:, 2], s[:, 3], ANGLE_PX * np.cos(s[:, 4]), ANGLE_PX * np.sin(s[:, 4])],
        axis=1,
    )


def jump_cost(dist):
    return 1.0 + 2.0 * np.asarray(dist) / JUMP_RADIUS


def split_episodes(n_episodes: int):
    rng = np.random.default_rng(SPLIT_SEED)
    eps = rng.permutation(n_episodes)
    n_test = max(1, int(round(n_episodes * TEST_FRACTION)))
    return np.sort(eps[n_test:]), np.sort(eps[:n_test])


class DemoGraph:
    def __init__(self, states, offsets, lengths, episodes):
        eps, ts = [], []
        for ep in episodes:
            t = np.arange(0, int(lengths[ep]), STRIDE)
            eps.append(np.full(len(t), int(ep)))
            ts.append(t)
        self.ep = np.concatenate(eps)
        self.t = np.concatenate(ts)
        self.row = np.asarray(offsets)[self.ep].astype(np.int64) + self.t
        self.state = np.asarray(states, dtype=np.float64)[self.row]
        self.feat = features(self.state)
        n = len(self.row)

        fwd = np.nonzero(self.ep[1:] == self.ep[:-1])[0]
        src = [fwd]
        dst = [fwd + 1]
        weight = [np.ones(len(fwd))]

        self.tree = cKDTree(self.feat)
        dist, idx = self.tree.query(self.feat, k=JUMP_K + 1, distance_upper_bound=JUMP_RADIUS)
        rows = np.repeat(np.arange(n), JUMP_K + 1)
        dist, idx = dist.ravel(), idx.ravel()
        keep = np.isfinite(dist) & (idx < n)
        rows, idx, dist = rows[keep], idx[keep], dist[keep]
        keep = self.ep[rows] != self.ep[idx]
        rows, idx, dist = rows[keep], idx[keep], dist[keep]
        src.append(rows)
        dst.append(idx)
        weight.append(jump_cost(dist))

        # (src, dst) pairs are unique: forward edges stay in one episode, jumps never do.
        self.graph = csr_matrix(
            (np.concatenate(weight), (np.concatenate(src), np.concatenate(dst))), shape=(n, n)
        )
        self.reverse = self.graph.T.tocsr()
        self.n_nodes = n
        self.n_forward = int(len(fwd))
        self.n_jump = int(len(rows))

    def plan_to(self, goal_state) -> dict:
        """Distance to the goal from every node, and the next node toward it."""
        goal_feat = features(goal_state)[0]
        near = self.tree.query_ball_point(goal_feat, GOAL_RADIUS)
        if len(near) == 0:
            _d, near = self.tree.query(goal_feat, k=5)
        near = np.atleast_1d(np.asarray(near, dtype=np.int64))
        dist, nxt, _src = dijkstra(
            self.reverse, directed=True, indices=near, return_predecessors=True, min_only=True
        )
        gap = float(np.min(np.linalg.norm(self.feat[near] - goal_feat, axis=1)))
        return {"dist": dist, "next": nxt, "goal_nodes": set(near.tolist()), "goal_gap": gap}

    def entry(self, state, plan):
        """Node near the current state with the cheapest way to the goal."""
        d, idx = self.tree.query(features(state)[0], k=min(ENTRY_K, self.n_nodes))
        d, idx = np.atleast_1d(d), np.atleast_1d(idx)
        cost = plan["dist"][idx] + jump_cost(d)
        best = int(np.argmin(cost))
        return int(idx[best]), float(d[best]), float(cost[best])

    def is_forward(self, a: int, b: int) -> bool:
        return bool(self.ep[a] == self.ep[b] and self.t[b] == self.t[a] + STRIDE)

    def edge_steps(self, a: int, b: int) -> float:
        if self.is_forward(a, b):
            return float(STRIDE)
        return float(STRIDE * np.linalg.norm(self.feat[b] - self.feat[a]) / JUMP_RADIUS)

    def walk(self, node: int, plan, max_steps: float):
        """Follow the chain toward the goal for about max_steps demo steps."""
        steps, edges, jumps = 0.0, 0, 0
        while steps < max_steps and node not in plan["goal_nodes"]:
            nxt = int(plan["next"][node])
            if nxt < 0:
                break
            steps += self.edge_steps(node, nxt)
            jumps += not self.is_forward(node, nxt)
            edges += 1
            node = nxt
        return node, node in plan["goal_nodes"], steps, edges, jumps


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def end_metrics(env, goal_state) -> dict:
    pose = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    goal = np.asarray(goal_state, dtype=np.float64)
    return {
        "pos_l2": float(np.linalg.norm(goal[:4] - pose[:4])),
        "block_xy_err": float(np.linalg.norm(goal[2:4] - pose[2:4])),
        "agent_xy_err": float(np.linalg.norm(goal[0:2] - pose[0:2])),
        "angle_err_deg": float(np.degrees(abs(wrap(pose[4] - goal[4])))),
    }


def render_state(env, state) -> np.ndarray:
    import eval_official_cem as official

    env.reset(seed=0)
    env.unwrapped._set_state(np.asarray(state, dtype=np.float64))
    return official.hwc(np.asarray(env.render()))


def encode(model, transform, frame):
    import torch

    import eval_official_cem as official

    pixels = official.to_pixels(transform, official.hwc(frame)).unsqueeze(0).cuda()
    with torch.inference_mode():
        return model.encode({"pixels": pixels})["emb"][0, -1]


def run_episode(env, crumb_env, model, transform, mean, std, goal_state, goal_frame, budget, graph):
    import eval_official_cem as official

    env.unwrapped._set_goal_state(np.asarray(goal_state, dtype=np.float64))
    z_goal = encode(model, transform, goal_frame)
    plan = graph.plan_to(goal_state) if graph is not None else None

    executed = 0
    success = False
    stopped = None
    hops = []
    while executed < budget and not success:
        z_target = z_goal
        hop = {"target": "goal"}
        if graph is not None:
            obs = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
            node, gap, cost = graph.entry(obs, plan)
            if np.isfinite(cost):
                crumb, final, steps, edges, jumps = graph.walk(node, plan, HOP_STEPS)
                hop = {
                    "target": "goal" if final else "crumb",
                    "entry_gap": gap,
                    "path_cost": float(plan["dist"][node]),
                    "crumb_steps": steps,
                    "edges": edges,
                    "jumps": jumps,
                }
                if not final:
                    z_target = encode(model, transform, render_state(crumb_env, graph.state[crumb]))
            else:
                hop = {"target": "goal", "no_path": True, "entry_gap": gap}

        pixels = official.to_pixels(transform, official.hwc(np.asarray(env.render()))).unsqueeze(0).cuda()

        def cost_fn(cand, pixels=pixels, z_t=z_target):
            z_h = official.rollout_last(model, pixels, cand)
            return ((z_h - z_t) ** 2).sum(-1)

        plan_norm = official.cem(cost_fn).detach().cpu().numpy()
        actions = official.denormalize_plan(plan_norm, mean, std)
        for action in actions[: budget - executed]:
            _obs, _reward, terminated, truncated, _info = env.step(action)
            executed += 1
            ok, _dist = env.unwrapped.eval_state(env.unwrapped.goal_state, env.unwrapped._get_obs())
            success = bool(ok)
            if success or terminated or truncated:
                if not success:
                    stopped = "terminated" if terminated else "truncated"
                break
        hops.append(hop)
        if stopped:
            break

    row = {"success": success, "steps": executed, "stopped": stopped, "n_hops": len(hops)}
    row.update(end_metrics(env, goal_state))
    row["n_crumb_hops"] = sum(h["target"] == "crumb" for h in hops)
    row["n_no_path"] = sum(bool(h.get("no_path")) for h in hops)
    row["hops"] = hops
    return row


def live_tasks(env, n):
    for seed in range(n):
        _obs, info = env.reset(seed=seed)
        meta = {"seed": seed, "start_state": np.asarray(env.unwrapped._get_obs()).tolist()}
        yield meta, np.asarray(info["goal_state"], dtype=np.float64), np.asarray(info["goal"])


def dataset_tasks(env, ds, states, test_eps, offset, n):
    import eval_official_cem as official

    lengths = np.asarray(ds.lengths)
    offsets = np.asarray(ds.offsets)
    pairs = [(int(ep), t) for ep in test_eps for t in range(0, int(lengths[ep]) - offset)]
    if not pairs:
        raise SystemExit(f"no held-out episode is longer than offset {offset}")
    rng = np.random.default_rng(TASK_SEED)
    pick = np.sort(rng.choice(len(pairs), size=min(n, len(pairs)), replace=False))
    for i in pick:
        ep, t = pairs[int(i)]
        state0 = states[int(offsets[ep]) + t]
        goal_state = states[int(offsets[ep]) + t + offset]
        chunk = ds.load_chunk(np.array([ep]), np.array([t + offset]), np.array([t + offset + 1]))[0]
        goal_frame = official.as_numpy(chunk["pixels"][-1])
        env.reset(seed=0)
        env.unwrapped._set_state(np.asarray(state0, dtype=np.float64))
        yield {"episode": ep, "start": t, "start_state": state0.tolist()}, goal_state, goal_frame


def coverage_row(graph, env, goal_state) -> dict:
    """Graph only: is there a chain of demo states from here to the goal?"""
    plan = graph.plan_to(goal_state)
    obs = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    node, gap, cost = graph.entry(obs, plan)
    row = {"entry_gap": gap, "goal_gap": plan["goal_gap"], "path": bool(np.isfinite(cost))}
    if row["path"]:
        _end, reached, steps, edges, jumps = graph.walk(node, plan, np.inf)
        row.update({"reaches_goal_node": reached, "demo_steps": steps, "edges": edges, "jumps": jumps})
    return row


def summarize(rows, dry):
    if dry:
        paths = [r for r in rows if r["path"]]
        return {
            "n": len(rows),
            "with_path": len(paths),
            "median_entry_gap": float(np.median([r["entry_gap"] for r in rows])),
            "median_goal_gap": float(np.median([r["goal_gap"] for r in rows])),
            "median_demo_steps": float(np.median([r["demo_steps"] for r in paths])) if paths else None,
            "median_jumps": float(np.median([r["jumps"] for r in paths])) if paths else None,
        }
    n_ok = int(sum(r["success"] for r in rows))
    return {
        "n": len(rows),
        "successes": n_ok,
        "success_rate": n_ok / len(rows) if rows else None,
        "mean_pos_l2": float(np.mean([r["pos_l2"] for r in rows])),
        "median_block_xy_err": float(np.median([r["block_xy_err"] for r in rows])),
        "median_angle_err_deg": float(np.median([r["angle_err_deg"] for r in rows])),
        "stopped_early": int(sum(r["stopped"] is not None for r in rows)),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("dataset", "live"), required=True)
    p.add_argument("--planner", choices=("flat", "crumbs"), default="crumbs")
    p.add_argument("--offset", type=int, default=50, help="dataset task: goal is this many demo steps ahead")
    p.add_argument("--budget", type=int, default=300, help="env steps per episode")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--dry", action="store_true", help="graph coverage only, no model, no CEM")
    return p.parse_args()


def main():
    args = parse_args()
    import eval_official_cem as official  # sets STABLEWM_HOME and headless env vars

    np.random.seed(TORCH_SEED)
    ds = official.open_dataset()
    mean, std, stats = official.action_stats(ds)
    lengths = np.asarray(ds.lengths)
    offsets = np.asarray(ds.offsets)
    states = np.asarray(ds.get_col_data("state"), dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 5:
        raise SystemExit(f"unexpected state column shape {states.shape}")
    print("episodes", len(lengths), "rows", len(states), "state_dim", states.shape[1], flush=True)
    train_eps, test_eps = split_episodes(len(lengths))
    print("train_episodes", len(train_eps), "test_episodes", len(test_eps), flush=True)

    graph = None
    if args.planner == "crumbs" or args.dry:
        t0 = time.time()
        graph = DemoGraph(states, offsets, lengths, train_eps)
        print(
            f"graph nodes={graph.n_nodes} forward={graph.n_forward} jumps={graph.n_jump} "
            f"build_s={time.time() - t0:.1f}",
            flush=True,
        )

    env = official.make_env()
    if args.task == "live":
        tasks = live_tasks(env, args.n)
    else:
        tasks = dataset_tasks(env, ds, states, test_eps, args.offset, args.n)

    rows = []
    model = transform = crumb_env = None
    if not args.dry:
        import torch

        torch.manual_seed(TORCH_SEED)
        torch.cuda.manual_seed_all(TORCH_SEED)
        print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
        model = official.load_model()
        transform = official.make_transform()
        crumb_env = official.make_env()
    t_start = time.time()
    try:
        for i, (meta, goal_state, goal_frame) in enumerate(tasks):
            if args.dry:
                row = coverage_row(graph, env, goal_state)
                row.update(meta)
                print(
                    f"ep {i} path={row['path']} entry_gap={row['entry_gap']:.1f} "
                    f"goal_gap={row['goal_gap']:.1f} steps={row.get('demo_steps', float('nan')):.0f} "
                    f"jumps={row.get('jumps', -1)}",
                    flush=True,
                )
            else:
                row = run_episode(env, crumb_env, model, transform, mean, std, goal_state, goal_frame, args.budget, graph)
                row.update(meta)
                print(
                    f"ep {i} success={row['success']} steps={row['steps']} pos={row['pos_l2']:.1f} "
                    f"block_xy={row['block_xy_err']:.1f} ang={row['angle_err_deg']:.1f} "
                    f"crumb_hops={row['n_crumb_hops']}/{row['n_hops']}",
                    flush=True,
                )
            rows.append(row)
    finally:
        env.close()
        if crumb_env is not None:
            crumb_env.close()

    summary = summarize(rows, args.dry)
    payload = {
        "summary": summary,
        "args": vars(args),
        "wall_s": time.time() - t_start,
        "graph": None
        if graph is None
        else {"nodes": graph.n_nodes, "forward_edges": graph.n_forward, "jump_edges": graph.n_jump},
        "constants": {
            "stride": STRIDE,
            "hop_steps": HOP_STEPS,
            "angle_px": ANGLE_PX,
            "jump_radius": JUMP_RADIUS,
            "jump_k": JUMP_K,
            "entry_k": ENTRY_K,
            "goal_radius": GOAL_RADIUS,
            "test_fraction": TEST_FRACTION,
            "split_seed": SPLIT_SEED,
            "task_seed": TASK_SEED,
            "torch_seed": TORCH_SEED,
        },
        "planner": {
            "cem": [official.N_SAMPLES, official.N_STEPS, official.TOPK],
            "horizon": official.HORIZON,
            "action_block": official.ACTION_BLOCK,
            "action_mean": mean.tolist(),
            "action_std": std.tolist(),
        },
        "episodes": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = args.task + (str(args.offset) if args.task == "dataset" else "")
    name += "_dry" if args.dry else f"_{args.planner}_b{args.budget}"
    out = OUT / f"{name}_n{len(rows)}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
