#!/usr/bin/env python3
"""Language end to end: free sentence -> parser -> demo-graph breadcrumbs -> env.

Check 3 used seven fixed commands. Here a command carries its own numbers
(turn by any signed angle, push any direction by any distance), and comes from
a sentence through scripts/command_parser.py.

Each held-out sentence gets one live start (reset seed = id % 10) and is run
twice with the check-2/3 crumbs planner:

  gold    the command the sentence means (label in the jsonl)
  parsed  the command the LLM parser read from the sentence

Both are scored on the GOLD region: did the block end up where the person
asked? So the parsed-vs-gold gap is what the language layer costs. When the
parse equals the gold command exactly, the parsed run is the gold run (same
input, same planner), so it is not re-run. An unusable parse counts as a miss.

  python scripts/language_breadcrumbs.py research/language/paraphrases.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import word_breadcrumbs as words
from command_parser import gold_command, parse_json
from demo_breadcrumbs import HOP_STEPS, OUT, TORCH_SEED, DemoGraph, encode, render_state, split_episodes, wrap
from word_breadcrumbs import ARENA, DIRS, TARGET, TOL_ANG, TOL_XY, TURN_DRIFT, angle_diff, goal_image_node, plan_to_nodes

N_STARTS = 10


def push_point(cmd, start):
    return np.asarray(start[2:4], dtype=np.float64) + cmd["px"] * np.asarray(DIRS[cmd["dir"]])


def valid(cmd, start) -> bool:
    if cmd["action"] != "push":
        return True
    p = push_point(cmd, start)
    return bool(np.all((p >= ARENA[0]) & (p <= ARENA[1])))


def in_region(cmd, start, states) -> np.ndarray:
    """Block-only test of a numeric command on (N, >=5) states. Same tolerances as check 3."""
    s = np.atleast_2d(np.asarray(states, dtype=np.float64))
    xy, th = s[:, 2:4], s[:, 4]
    start = np.asarray(start, dtype=np.float64)
    if cmd["action"] == "target":
        return (np.linalg.norm(xy - TARGET[:2], axis=1) < TOL_XY) & (angle_diff(th, TARGET[2]) < TOL_ANG)
    if cmd["action"] == "turn":
        return (np.linalg.norm(xy - start[2:4], axis=1) < TURN_DRIFT) & (
            angle_diff(th, start[4] + np.radians(cmd["deg"])) < TOL_ANG
        )
    return np.linalg.norm(xy - push_point(cmd, start), axis=1) < TOL_XY


def run_episode(env, crumb_env, model, transform, mean, std, cmd, gold, start, graph, plan, goal_node, budget):
    """Check-3 crumbs loop, chasing `cmd` and scored on `gold`."""
    import eval_official_cem as official

    env.unwrapped._set_goal_state(np.asarray(graph.state[goal_node], dtype=np.float64))
    z_goal = encode(model, transform, render_state(crumb_env, graph.state[goal_node]))

    executed = 0
    success = False
    n_hops = n_crumb = 0
    while executed < budget and not success:
        z_target = z_goal
        obs = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
        node, _gap, cost = graph.entry(obs, plan)
        if np.isfinite(cost):
            crumb, final, *_ = graph.walk(node, plan, HOP_STEPS)
            n_crumb += not final
            if crumb != goal_node:
                z_target = encode(model, transform, render_state(crumb_env, graph.state[crumb]))

        pixels = official.to_pixels(transform, official.hwc(np.asarray(env.render()))).unsqueeze(0).cuda()

        def cost_fn(cand, pixels=pixels, z_t=z_target):
            z_h = official.rollout_last(model, pixels, cand)
            return ((z_h - z_t) ** 2).sum(-1)

        plan_norm = official.cem(cost_fn).detach().cpu().numpy()
        actions = official.denormalize_plan(plan_norm, mean, std)
        for action in actions[: budget - executed]:
            env.step(action)
            executed += 1
            if in_region(gold, start, env.unwrapped._get_obs())[0]:
                success = True
                break
        n_hops += 1

    pose = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    return {
        "success": success,
        "steps": executed,
        "n_hops": n_hops,
        "n_crumb_hops": n_crumb,
        "end_block": pose[2:5].tolist(),
        "block_moved_px": float(np.linalg.norm(pose[2:4] - start[2:4])),
        "block_turned_deg": float(np.degrees(wrap(pose[4] - start[4]))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sentences", type=Path)
    p.add_argument("--parsed", type=Path, help="parse_*.json from command_parser --eval (default: matching name)")
    p.add_argument("--budget", type=int, default=300)
    args = p.parse_args()
    parsed_path = args.parsed or OUT / f"parse_{args.sentences.stem}_n{sum(1 for l in args.sentences.read_text().splitlines() if l.strip())}.json"
    parsed = {r["id"]: r["llm"] for r in json.loads(parsed_path.read_text())["rows"]}

    import torch

    import eval_official_cem as official

    np.random.seed(TORCH_SEED)
    ds = official.open_dataset()
    mean, std, _stats = official.action_stats(ds)
    states = np.asarray(ds.get_col_data("state"), dtype=np.float64)
    lengths, offsets = np.asarray(ds.lengths), np.asarray(ds.offsets)
    graph = DemoGraph(states, offsets, lengths, split_episodes(len(lengths))[0])
    model = official.load_model()
    transform = official.make_transform()
    env, crumb_env = official.make_env(), official.make_env()

    rows, dropped = [], []
    t_start = time.time()
    try:
        for line in args.sentences.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            gold = gold_command(r)
            seed = int(r["id"]) % N_STARTS
            env.reset(seed=seed)
            start = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
            if gold is None or not valid(gold, start):
                dropped.append(r["id"])
                continue
            row = {"id": r["id"], "text": r["text"], "style": r.get("style"), "seed": seed, "gold": gold}
            row["parsed"] = parsed.get(r["id"])
            for name, cmd in (("gold", gold), ("parsed", row["parsed"])):
                if name == "parsed" and cmd == gold:
                    row["parsed_run"] = dict(row["gold_run"], reused_gold=True)
                    continue
                goal_nodes = None if cmd is None else np.nonzero(in_region(cmd, start, graph.state))[0]
                if cmd is None or len(goal_nodes) == 0:
                    row[f"{name}_run"] = {"success": False, "skipped": "no parse" if cmd is None else "no goal node"}
                    continue
                torch.manual_seed(TORCH_SEED + int(r["id"]))
                env.reset(seed=seed)
                plan = plan_to_nodes(graph, goal_nodes)
                goal_node, _info = goal_image_node(graph, plan, goal_nodes, start)
                row[f"{name}_run"] = run_episode(
                    env, crumb_env, model, transform, mean, std, cmd, gold, start, graph, plan, goal_node, args.budget
                )
            rows.append(row)
            print(
                f"id {r['id']} {gold['action']} gold={row['gold_run']['success']} parsed={row['parsed_run']['success']} "
                f"same_cmd={row['parsed'] == gold} | {r['text'][:60]}",
                flush=True,
            )
    finally:
        env.close()
        crumb_env.close()

    def rate(sub, key):
        return f"{sum(x[key + '_run']['success'] for x in sub)}/{len(sub)}"

    summary = {
        "n": len(rows),
        "dropped": len(dropped),
        "gold_successes": sum(x["gold_run"]["success"] for x in rows),
        "parsed_successes": sum(x["parsed_run"]["success"] for x in rows),
        "same_command": sum(x["parsed"] == x["gold"] for x in rows),
        "gold_by_action": {a: rate([x for x in rows if x["gold"]["action"] == a], "gold") for a in ("target", "turn", "push")},
        "parsed_by_action": {a: rate([x for x in rows if x["gold"]["action"] == a], "parsed") for a in ("target", "turn", "push")},
    }
    summary["gold_rate"] = summary["gold_successes"] / len(rows)
    summary["parsed_rate"] = summary["parsed_successes"] / len(rows)
    payload = {
        "summary": summary,
        "args": {k: str(v) for k, v in vars(args).items()},
        "parsed_from": str(parsed_path),
        "wall_s": time.time() - t_start,
        "dropped_ids": dropped,
        "constants": {"n_starts": N_STARTS, "turn_drift": TURN_DRIFT, "tol_xy": TOL_XY, "tol_ang_deg": float(np.degrees(TOL_ANG))},
        "rows": rows,
    }
    out = OUT / f"language_{args.sentences.stem}_b{args.budget}_n{len(rows)}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
