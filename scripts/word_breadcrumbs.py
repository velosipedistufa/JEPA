#!/usr/bin/env python3
"""Check 3 of HANDOFF_5090.md: words pick the goal, the demo graph finds the route.

A command names a region of block poses, not one goal image:

  target      "put the T on the target"      block within 20 px and 20 deg of (256, 256, pi/4)
  turn_cw     "turn the T 90 deg clockwise"  block angle start + 90 deg (20 deg), block xy within 40 px of start
  turn_ccw    "turn the T 90 deg counter-clockwise"  same with start - 90 deg
  push_left / push_right / push_up / push_down   block xy within 20 px of start + 80 px that way, angle free

Screen y points down and a positive block angle turns the T clockwise on screen
(checked on the env render). The 20 px / 20 deg tolerances are the env's own
eval_state ones. Success is scored on the block only, at any step.

Goal nodes are the demo-graph nodes (same graph as check 2) whose block pose is
in the region. Both planners chase the same goal image: the rendered frame of
the goal node the graph route from the start ends at.

  flat    chases that one image for the whole budget (words name a reachable goal)
  crumbs  check-2 breadcrumbs toward the goal nodes; the last hop chases the
          rendered goal node the route currently ends at

Starts are live env resets, seeds 0..n-1. A push whose target point leaves the
arena box is dropped before anything runs.

  python scripts/word_breadcrumbs.py --dry --n 10
  python scripts/word_breadcrumbs.py --planner flat --n 10
  python scripts/word_breadcrumbs.py --planner crumbs --n 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.sparse.csgraph import dijkstra

import demo_breadcrumbs as crumbs
from demo_breadcrumbs import HOP_STEPS, OUT, TORCH_SEED, DemoGraph, encode, features, render_state, split_episodes, wrap

TOL_XY = 20.0
TOL_ANG = np.pi / 9
TURN_DRIFT = 40.0
PUSH_PX = 80.0
TARGET = np.array([256.0, 256.0, np.pi / 4])
ARENA = (80.0, 432.0)  # a push target point must stay inside this box
IN_PLACE_XY = 15.0  # side question: block moves less than this ...
IN_PLACE_ANG = np.pi / 2  # ... while turning more than this ...
IN_PLACE_STEPS = 50  # ... within this many demo steps

COMMANDS = {
    "target": "put the T on the target",
    "turn_cw": "turn the T 90 degrees clockwise",
    "turn_ccw": "turn the T 90 degrees counter-clockwise",
    "push_left": "push the T left",
    "push_right": "push the T right",
    "push_up": "push the T up",
    "push_down": "push the T down",
}
DIRS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}


def angle_diff(a, b):
    return np.abs(wrap(np.asarray(a) - np.asarray(b)))


def push_point(cmd, start):
    return np.asarray(start[2:4], dtype=np.float64) + PUSH_PX * np.asarray(DIRS[cmd.split("_")[1]])


def valid(cmd, start) -> bool:
    if not cmd.startswith("push"):
        return True
    p = push_point(cmd, start)
    return bool(np.all((p >= ARENA[0]) & (p <= ARENA[1])))


def in_region(cmd, start, states) -> np.ndarray:
    """Block-only test of the command on (N, >=5) states."""
    s = np.atleast_2d(np.asarray(states, dtype=np.float64))
    xy, th = s[:, 2:4], s[:, 4]
    start = np.asarray(start, dtype=np.float64)
    if cmd == "target":
        return (np.linalg.norm(xy - TARGET[:2], axis=1) < TOL_XY) & (angle_diff(th, TARGET[2]) < TOL_ANG)
    if cmd.startswith("turn"):
        sign = 1.0 if cmd == "turn_cw" else -1.0
        return (np.linalg.norm(xy - start[2:4], axis=1) < TURN_DRIFT) & (
            angle_diff(th, start[4] + sign * np.pi / 2) < TOL_ANG
        )
    return np.linalg.norm(xy - push_point(cmd, start), axis=1) < TOL_XY


def plan_to_nodes(graph, goal_nodes) -> dict:
    """Like DemoGraph.plan_to, but the goal is a set of nodes picked by the command."""
    dist, nxt, _src = dijkstra(
        graph.reverse, directed=True, indices=goal_nodes, return_predecessors=True, min_only=True
    )
    return {"dist": dist, "next": nxt, "goal_nodes": set(goal_nodes.tolist())}


def goal_image_node(graph, plan, goal_nodes, start):
    """Goal node the route from the start ends at; nearest goal node if there is no route."""
    node, gap, cost = graph.entry(start, plan)
    if np.isfinite(cost):
        end, reached, steps, _edges, jumps = graph.walk(node, plan, np.inf)
        if reached:
            return end, {"route": True, "entry_gap": gap, "demo_steps": steps, "jumps": jumps}
    d = np.linalg.norm(graph.feat[goal_nodes] - features(start)[0], axis=1)
    return int(goal_nodes[int(np.argmin(d))]), {"route": False, "entry_gap": gap}


def in_place_turns(graph) -> dict:
    """Side question: do the demos ever turn the T in place?"""
    hit = np.zeros(graph.n_nodes, dtype=bool)
    for k in range(1, IN_PLACE_STEPS // crumbs.STRIDE + 1):
        i = np.arange(graph.n_nodes - k)
        j = i + k
        same = graph.ep[i] == graph.ep[j]
        i, j = i[same], j[same]
        moved = np.linalg.norm(graph.state[j, 2:4] - graph.state[i, 2:4], axis=1)
        turned = angle_diff(graph.state[j, 4], graph.state[i, 4])
        hit[i[(moved < IN_PLACE_XY) & (turned > IN_PLACE_ANG)]] = True
    return {
        "nodes": int(hit.sum()),
        "of_nodes": int(graph.n_nodes),
        "episodes": int(len(np.unique(graph.ep[hit]))),
        "of_episodes": int(len(np.unique(graph.ep))),
        "rule": f"block moves < {IN_PLACE_XY} px while turning > 90 deg within {IN_PLACE_STEPS} demo steps",
    }


def run_episode(env, crumb_env, model, transform, mean, std, cmd, start, graph, plan, goal_node, budget, planner):
    import eval_official_cem as official

    env.unwrapped._set_goal_state(np.asarray(graph.state[goal_node], dtype=np.float64))
    z_goal = encode(model, transform, render_state(crumb_env, graph.state[goal_node]))

    executed = 0
    success = False
    hops = []
    while executed < budget and not success:
        z_target = z_goal
        hop = {"target": "goal_image"}
        if planner == "crumbs":
            obs = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
            node, gap, cost = graph.entry(obs, plan)
            if np.isfinite(cost):
                crumb, final, steps, edges, jumps = graph.walk(node, plan, HOP_STEPS)
                hop = {
                    "target": "goal_node" if final else "crumb",
                    "entry_gap": gap,
                    "path_cost": float(plan["dist"][node]),
                    "crumb_steps": steps,
                    "edges": edges,
                    "jumps": jumps,
                }
                if crumb != goal_node:
                    z_target = encode(model, transform, render_state(crumb_env, graph.state[crumb]))
            else:
                hop = {"target": "goal_image", "no_path": True, "entry_gap": gap}

        pixels = official.to_pixels(transform, official.hwc(np.asarray(env.render()))).unsqueeze(0).cuda()

        def cost_fn(cand, pixels=pixels, z_t=z_target):
            z_h = official.rollout_last(model, pixels, cand)
            return ((z_h - z_t) ** 2).sum(-1)

        plan_norm = official.cem(cost_fn).detach().cpu().numpy()
        actions = official.denormalize_plan(plan_norm, mean, std)
        # env "terminated" compares agent and block to the goal image state; only the
        # block test of the command counts here, so it is not used to stop.
        for action in actions[: budget - executed]:
            env.step(action)
            executed += 1
            if in_region(cmd, start, env.unwrapped._get_obs())[0]:
                success = True
                break
        hops.append(hop)

    pose = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    row = {"success": success, "steps": executed, "n_hops": len(hops)}
    row["end_block"] = pose[2:5].tolist()
    row["block_moved_px"] = float(np.linalg.norm(pose[2:4] - np.asarray(start[2:4])))
    row["block_turned_deg"] = float(np.degrees(wrap(pose[4] - start[4])))
    if cmd.startswith("turn"):
        # report only: artifact 012's stricter turn gate (xy drift < 15 px)
        row["gate_012"] = bool(success and row["block_moved_px"] < 15.0)
    row["n_crumb_hops"] = sum(h["target"] == "crumb" for h in hops)
    row["n_no_path"] = sum(bool(h.get("no_path")) for h in hops)
    row["hops"] = hops
    return row


def summarize(rows, dry):
    out = {"n": len(rows)}
    by = {}
    for cmd in COMMANDS:
        sub = [r for r in rows if r["command"] == cmd]
        if dry:
            by[cmd] = f"{sum(r['route'] for r in sub)}/{len(sub)}"
        else:
            by[cmd] = f"{sum(r['success'] for r in sub)}/{len(sub)}"
    if dry:
        out["with_route"] = int(sum(r["route"] for r in rows))
        out["no_goal_node"] = int(sum(r["n_goal_nodes"] == 0 for r in rows))
        out["route_by_command"] = by
    else:
        out["successes"] = int(sum(r["success"] for r in rows))
        out["success_rate"] = out["successes"] / len(rows) if rows else None
        turns = [r for r in rows if r["command"].startswith("turn")]
        out["turn_successes"] = f"{sum(r['success'] for r in turns)}/{len(turns)}"
        out["turn_gate_012"] = f"{sum(r.get('gate_012', False) for r in turns)}/{len(turns)}"
        out["success_by_command"] = by
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planner", choices=("flat", "crumbs"), default="crumbs")
    p.add_argument("--budget", type=int, default=300, help="env steps per episode")
    p.add_argument("--n", type=int, default=10, help="live reset seeds 0..n-1, every command on each")
    p.add_argument("--dry", action="store_true", help="goal nodes and routes only, no model, no CEM")
    return p.parse_args()


def main():
    args = parse_args()
    import eval_official_cem as official  # sets STABLEWM_HOME and headless env vars

    np.random.seed(TORCH_SEED)
    ds = official.open_dataset()
    mean, std, _stats = official.action_stats(ds)
    lengths = np.asarray(ds.lengths)
    offsets = np.asarray(ds.offsets)
    states = np.asarray(ds.get_col_data("state"), dtype=np.float64)
    train_eps, _test_eps = split_episodes(len(lengths))
    t0 = time.time()
    graph = DemoGraph(states, offsets, lengths, train_eps)
    print(f"graph nodes={graph.n_nodes} jumps={graph.n_jump} build_s={time.time() - t0:.1f}", flush=True)
    turns = in_place_turns(graph)
    print("IN_PLACE", json.dumps(turns), flush=True)

    env = official.make_env()
    starts = []
    for seed in range(args.n):
        env.reset(seed=seed)
        starts.append((seed, np.asarray(env.unwrapped._get_obs(), dtype=np.float64)))
    pairs = [(seed, start, cmd) for seed, start in starts for cmd in COMMANDS]
    dropped = [(seed, cmd) for seed, start, cmd in pairs if not valid(cmd, start)]
    pairs = [(seed, start, cmd) for seed, start, cmd in pairs if valid(cmd, start)]
    print(f"pairs={len(pairs)} dropped={len(dropped)} (push target outside the arena box)", flush=True)

    model = transform = crumb_env = None
    if not args.dry:
        import torch

        torch.manual_seed(TORCH_SEED)
        torch.cuda.manual_seed_all(TORCH_SEED)
        model = official.load_model()
        transform = official.make_transform()
        crumb_env = official.make_env()

    rows = []
    t_start = time.time()
    try:
        for i, (seed, start, cmd) in enumerate(pairs):
            goal_nodes = np.nonzero(in_region(cmd, start, graph.state))[0]
            meta = {"seed": seed, "command": cmd, "words": COMMANDS[cmd], "start_state": start.tolist()}
            meta["n_goal_nodes"] = int(len(goal_nodes))
            if len(goal_nodes) == 0:
                row = {"route": False, "success": False, "no_goal_node": True}
                row.update(meta)
                rows.append(row)
                print(f"ep {i} seed={seed} {cmd} no goal node", flush=True)
                continue
            plan = plan_to_nodes(graph, goal_nodes)
            goal_node, info = goal_image_node(graph, plan, goal_nodes, start)
            meta.update(info)
            meta["goal_node_state"] = graph.state[goal_node].tolist()
            if args.dry:
                row = dict(meta)
                print(
                    f"ep {i} seed={seed} {cmd} goal_nodes={len(goal_nodes)} route={info['route']} "
                    f"entry_gap={info['entry_gap']:.1f} steps={info.get('demo_steps', float('nan')):.0f} "
                    f"jumps={info.get('jumps', -1)}",
                    flush=True,
                )
            else:
                env.reset(seed=seed)
                row = run_episode(
                    env, crumb_env, model, transform, mean, std, cmd, start, graph, plan, goal_node,
                    args.budget, args.planner,
                )
                row.update(meta)
                print(
                    f"ep {i} seed={seed} {cmd} success={row['success']} steps={row['steps']} "
                    f"moved={row['block_moved_px']:.0f}px turned={row['block_turned_deg']:.0f}deg "
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
        "in_place_turns": turns,
        "args": vars(args),
        "wall_s": time.time() - t_start,
        "dropped_pairs": dropped,
        "commands": COMMANDS,
        "constants": {
            "tol_xy": TOL_XY,
            "tol_ang_deg": float(np.degrees(TOL_ANG)),
            "turn_drift": TURN_DRIFT,
            "push_px": PUSH_PX,
            "target": TARGET.tolist(),
            "arena": ARENA,
            "screen": "y down; positive block angle is clockwise on screen",
        },
        "graph": {"nodes": graph.n_nodes, "forward_edges": graph.n_forward, "jump_edges": graph.n_jump},
        "episodes": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = "words_dry" if args.dry else f"words_{args.planner}_b{args.budget}"
    out = OUT / f"{name}_n{len(rows)}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
