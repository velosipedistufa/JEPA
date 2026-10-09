#!/usr/bin/env python3
"""Words straight into LeWM latent space: a text-conditioned subgoal head.

Sasha's question: can the words go into the world model instead of through a
parser and a graph? LeWM's planner is reliable only for goals about 25 steps
away, so the words should not name the final goal latent. They should name the
next nearby subgoal latent, again and again.

  head(z_start, z_now, text) -> z_next      (LeWM latents, 192-d; text = frozen Qwen mean-pooled embedding)

Training targets come from the demo graph (train episodes only): for random
(start, command) pairs, the graph route is cut into hops of 25 demo steps and
each hop gives one example (start, now, sentence) -> next crumb. The sentences
come from the template grammar below. The held-out Codex paraphrases are never
used for training.

At test time there is no graph and no parser: every hop the head turns
(z_start, z_now, sentence) into a target latent and the official CEM chases it.

  data     build routes and examples (CPU), render + encode them (GPU)
  train    fit the head, report validation MSE against copying z_now
  eval     end to end on research/language/paraphrases.jsonl (same starts, same
           gold scoring as L2); --shuffle gives every sentence another row's text
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from command_parser import gold_command
from demo_breadcrumbs import HOP_STEPS, OUT, TORCH_SEED, DemoGraph, render_state, split_episodes
from language_breadcrumbs import in_region, valid
from word_breadcrumbs import goal_image_node, plan_to_nodes

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "latent_subgoal"  # git-excluded: examples, latents, weights
QWEN = ROOT / "data" / "hf_models" / "Qwen2.5-1.5B-Instruct"

N_PLANS = 6000
TRAIN_LIVE_SEED0 = 1000  # live starts for training use reset seeds >= this; the test uses 0..9
SENTENCES_PER_PLAN = 3
DATA_SEED = 7
VAL_FRACTION = 0.1
MAX_HOPS = 16

TURN_DEGS = (30, 45, 60, 90, 120, 135, 150, 180)
PUSH_PXS = (30, 40, 50, 60, 80, 100, 120, 150, 160, 200)

# ---------------------------------------------------------------- sentences


def describe(cmd, rng) -> str:
    """One training sentence for a command. Written before and independent of the Codex test set."""
    pick = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    obj = pick(["the T", "it", "the block", "the piece", "the T block", "the shape"])
    if cmd["action"] == "target":
        return pick(
            [
                f"put {obj} on the target",
                f"move {obj} onto the green outline",
                f"place {obj} in the goal pose",
                f"line {obj} up with the green T",
                f"get {obj} into the target spot",
                f"match {obj} to the green outline",
                f"put {obj} where the green shape is",
                f"align {obj} with the goal",
                f"bring {obj} to the target position and angle",
                f"park {obj} on the green T",
            ]
        )
    if cmd["action"] == "turn":
        d = abs(int(round(cmd["deg"])))
        cw = cmd["deg"] > 0
        way = pick(["clockwise", "to the right", "right"] if cw else ["counterclockwise", "anticlockwise", "counter-clockwise", "to the left", "left"])
        amounts = [f"{d} degrees", f"{d} deg", f"by {d} degrees"]
        if d == 90:
            amounts += ["a quarter turn", "", "90 degrees"]
        if d == 180:
            amounts += ["half a turn", "all the way round", "a half turn"]
        if d == 45:
            amounts += ["a bit", "slightly", "a little", "an eighth of a turn"]
        amount = pick(amounts)
        verb = pick(["turn", "rotate", "spin", "twist"])
        return " ".join(
            w
            for w in pick(
                [[verb, obj, amount, way], [verb, obj, way, amount], [verb, amount, way], [verb, obj, way] if amount == "" else [verb, obj, amount, way]]
            )
            if w
        )
    d, px = cmd["dir"], int(round(cmd["px"]))
    way = pick(
        {
            "left": ["left", "to the left", "leftward", "west"],
            "right": ["right", "to the right", "rightward", "east"],
            "up": ["up", "upward", "toward the top", "north"],
            "down": ["down", "downward", "toward the bottom", "south"],
        }[d]
    )
    amounts = [f"{px} pixels", f"{px} px", f"by {px} pixels"]
    if px == 80:
        amounts += [""]
    if px == 40:
        amounts += ["a bit", "slightly", "a little"]
    if px == 100:
        amounts += ["one T length"]
    if px == 200:
        amounts += ["two T lengths"]
    if px == 160:
        amounts += ["far", "a lot"]
    amount = pick(amounts)
    verb = pick(["push", "move", "slide", "shove"] + (["nudge"] if px == 40 else []))
    return " ".join(w for w in pick([[verb, obj, way, amount], [verb, obj, amount, way]]) if w)


def random_command(rng):
    r = rng.random()
    if r < 0.2:
        return {"action": "target"}
    if r < 0.6:
        return {"action": "turn", "deg": float(rng.choice(TURN_DEGS)) * float(rng.choice((-1, 1)))}
    return {"action": "push", "dir": str(rng.choice(("left", "right", "up", "down"))), "px": float(rng.choice(PUSH_PXS))}


# ---------------------------------------------------------------- data

_G = None  # graph shared with forked workers


def route_examples(args):
    """One (start, command) plan -> list of (now_state, next_state) hops along the graph route."""
    start, cmd = args
    goal_nodes = np.nonzero(in_region(cmd, start, _G.state))[0]
    if len(goal_nodes) == 0:
        return None
    plan = plan_to_nodes(_G, goal_nodes)
    node, _gap, cost = _G.entry(start, plan)
    if not np.isfinite(cost):
        return None
    pairs, now = [], np.asarray(start)
    for _ in range(MAX_HOPS):
        nxt, final, *_ = _G.walk(node, plan, HOP_STEPS)
        pairs.append((now, _G.state[nxt]))
        if final or nxt == node:
            pairs.append((_G.state[nxt], _G.state[nxt]))  # at the goal: hold
            break
        now, node = _G.state[nxt], nxt
    return pairs


def build_data(args):
    global _G
    import eval_official_cem as official

    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(DATA_SEED)
    ds = official.open_dataset()
    states = np.asarray(ds.get_col_data("state"), dtype=np.float64)
    lengths, offsets = np.asarray(ds.lengths), np.asarray(ds.offsets)
    _G = DemoGraph(states, offsets, lengths, split_episodes(len(lengths))[0])

    env = official.make_env()
    jobs = []
    while len(jobs) < N_PLANS:
        if rng.random() < 0.5:
            env.reset(seed=TRAIN_LIVE_SEED0 + len(jobs))
            start = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
        else:
            start = _G.state[int(rng.integers(_G.n_nodes))].copy()
        cmd = random_command(rng)
        if valid(cmd, start):
            jobs.append((start, cmd))
    t0 = time.time()
    with mp.get_context("fork").Pool(24) as pool:
        routes = pool.map(route_examples, jobs, chunksize=20)
    print(f"routes {sum(r is not None for r in routes)}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)

    plan_id, start_s, now_s, next_s, cmds = [], [], [], [], []
    for i, ((start, cmd), hops) in enumerate(zip(jobs, routes)):
        if not hops:
            continue
        for now, nxt in hops:
            plan_id.append(i)
            start_s.append(start)
            now_s.append(now)
            next_s.append(nxt)
        cmds.append((i, cmd))
    sentences = {i: [describe(cmd, rng) for _ in range(SENTENCES_PER_PLAN)] for i, cmd in cmds}
    print(f"examples {len(plan_id)} plans {len(cmds)}", flush=True)

    # render + encode every distinct state once
    all_states = np.concatenate([np.asarray(start_s), np.asarray(now_s), np.asarray(next_s)])
    uniq, inv = np.unique(np.round(all_states, 4), axis=0, return_inverse=True)
    z = encode_states(uniq)
    n = len(plan_id)
    np.savez_compressed(
        WORK / "examples.npz",
        plan_id=np.asarray(plan_id),
        z_start=z[inv[:n]],
        z_now=z[inv[n : 2 * n]],
        z_next=z[inv[2 * n :]],
    )
    (WORK / "sentences.json").write_text(json.dumps({"plans": {str(i): s for i, s in sentences.items()}, "commands": {str(i): c for i, c in cmds}}, indent=1))
    print("WROTE", WORK / "examples.npz", flush=True)


def encode_states(states, batch=256):
    import torch

    import eval_official_cem as official

    model, transform, env = official.load_model(), official.make_transform(), official.make_env()
    out = []
    t0 = time.time()
    for b in range(0, len(states), batch):
        frames = [render_state(env, s) for s in states[b : b + batch]]
        pix = torch.cat([official.to_pixels(transform, f) for f in frames]).unsqueeze(1).cuda()
        with torch.inference_mode():
            out.append(model.encode({"pixels": pix})["emb"][:, -1].float().cpu().numpy())
    print(f"encoded {len(states)} states in {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(out)


# ---------------------------------------------------------------- text


class TextEncoder:
    """Frozen Qwen2.5-1.5B, mean-pooled last hidden layer of the raw sentence."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(QWEN)
        self.model = AutoModel.from_pretrained(QWEN, dtype=torch.bfloat16).cuda().eval()

    def __call__(self, texts, batch=128):
        out = []
        for b in range(0, len(texts), batch):
            enc = self.tok(list(texts[b : b + batch]), return_tensors="pt", padding=True).to("cuda")
            with self.torch.inference_mode():
                h = self.model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        return np.concatenate(out)


# ---------------------------------------------------------------- head


def make_head(text_dim, z_dim=192, hidden=512):
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 256), nn.GELU())
            self.net = nn.Sequential(
                nn.Linear(2 * z_dim + 256, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, z_dim)
            )

        def forward(self, z_start, z_now, t):
            import torch

            return z_now + self.net(torch.cat([z_start, z_now, self.text(t)], -1))  # predict the step from z_now

    return Head()


def train(args):
    import torch

    torch.manual_seed(TORCH_SEED)
    d = np.load(WORK / "examples.npz")
    meta = json.loads((WORK / "sentences.json").read_text())
    plans = sorted({int(p) for p in d["plan_id"]})
    rng = np.random.default_rng(DATA_SEED)
    val_plans = set(rng.choice(plans, size=int(len(plans) * VAL_FRACTION), replace=False).tolist())

    texts = [s for p in plans for s in meta["plans"][str(p)]]
    emb = TextEncoder()(texts)
    t_index = {p: SENTENCES_PER_PLAN * k for k, p in enumerate(plans)}

    pid = d["plan_id"]
    is_val = np.array([p in val_plans for p in pid])
    zs, zn, zx = (torch.tensor(d[k]).cuda() for k in ("z_start", "z_now", "z_next"))
    T = torch.tensor(emb).cuda()
    tbase = torch.tensor([t_index[int(p)] for p in pid]).cuda()
    tr, va = torch.tensor(np.nonzero(~is_val)[0]).cuda(), torch.tensor(np.nonzero(is_val)[0]).cuda()

    head = make_head(T.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = 6000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        b = tr[torch.randint(len(tr), (512,), device="cuda")]
        ti = tbase[b] + torch.randint(SENTENCES_PER_PLAN, (len(b),), device="cuda")
        loss = ((head(zs[b], zn[b], T[ti]) - zx[b]) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step} train_mse {loss.item():.3f}", flush=True)

    head.eval()
    with torch.no_grad():
        ti = tbase[va]
        pred = head(zs[va], zn[va], T[ti])
        mse = ((pred - zx[va]) ** 2).sum(-1).mean().item()
        copy = ((zn[va] - zx[va]) ** 2).sum(-1).mean().item()
        perm = va[torch.randperm(len(va), device="cuda")]
        shuf = ((head(zs[va], zn[va], T[tbase[perm]]) - zx[va]) ** 2).sum(-1).mean().item()
    torch.save(head.state_dict(), WORK / "head.pt")
    summary = {
        "examples": int(len(pid)),
        "plans": len(plans),
        "val_examples": int(is_val.sum()),
        "val_mse": mse,
        "val_copy_now_mse": copy,
        "ratio_to_copy": mse / copy,
        "val_shuffled_text_mse": shuf,
    }
    (OUT / "latent_subgoal_train.json").write_text(json.dumps({"summary": summary}, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


# ---------------------------------------------------------------- eval


def evaluate(args):
    import torch

    import eval_official_cem as official

    rows_in = [json.loads(l) for l in args.sentences.read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in rows_in]
    emb = TextEncoder()(texts)
    if args.shuffle:
        emb = emb[np.random.default_rng(DATA_SEED).permutation(len(emb))]
    head = make_head(emb.shape[1]).cuda()
    head.load_state_dict(torch.load(WORK / "head.pt"))
    head.eval()

    ds = official.open_dataset()
    mean, std, _ = official.action_stats(ds)
    model, transform, env = official.load_model(), official.make_transform(), official.make_env()

    def z_of(frame):
        pix = official.to_pixels(transform, official.hwc(np.asarray(frame))).unsqueeze(0).cuda()
        with torch.inference_mode():
            return model.encode({"pixels": pix})["emb"][0, -1].float()

    rows, dropped = [], []
    t0 = time.time()
    for k, r in enumerate(rows_in):
        gold = gold_command(r)
        seed = int(r["id"]) % 10
        env.reset(seed=seed)
        start = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
        if gold is None or not valid(gold, start):
            dropped.append(r["id"])
            continue
        torch.manual_seed(TORCH_SEED + int(r["id"]))
        t = torch.tensor(emb[k]).cuda().unsqueeze(0)
        z_start = z_of(env.render()).unsqueeze(0)
        executed, success, hops = 0, False, 0
        while executed < args.budget and not success:
            pixels = official.to_pixels(transform, official.hwc(np.asarray(env.render()))).unsqueeze(0).cuda()
            with torch.no_grad():
                z_t = head(z_start, z_of(env.render()).unsqueeze(0), t)[0]

            def cost_fn(cand, pixels=pixels, z_t=z_t):
                return ((official.rollout_last(model, pixels, cand) - z_t) ** 2).sum(-1)

            actions = official.denormalize_plan(official.cem(cost_fn).detach().cpu().numpy(), mean, std)
            for a in actions[: args.budget - executed]:
                env.step(a)
                executed += 1
                if in_region(gold, start, env.unwrapped._get_obs())[0]:
                    success = True
                    break
            hops += 1
        pose = np.asarray(env.unwrapped._get_obs())
        rows.append(
            {
                "id": r["id"], "text": r["text"], "style": r.get("style"), "seed": seed, "gold": gold,
                "success": success, "steps": executed, "hops": hops,
                "block_moved_px": float(np.linalg.norm(pose[2:4] - start[2:4])),
            }
        )
        print(f"id {r['id']} {gold['action']} success={success} steps={executed} | {r['text'][:60]}", flush=True)

    by = {a: f"{sum(x['success'] for x in rows if x['gold']['action'] == a)}/{sum(x['gold']['action'] == a for x in rows)}" for a in ("target", "turn", "push")}
    summary = {"n": len(rows), "dropped": len(dropped), "successes": sum(x["success"] for x in rows), "by_action": by, "shuffled_text": args.shuffle}
    summary["rate"] = summary["successes"] / len(rows)
    name = f"latent_subgoal_{args.sentences.stem}{'_shuffled' if args.shuffle else ''}_b{args.budget}_n{len(rows)}.json"
    (OUT / name).write_text(json.dumps({"summary": summary, "wall_s": time.time() - t0, "dropped_ids": dropped, "rows": rows}, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)
    print("WROTE", OUT / name, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    sub.add_parser("train")
    e = sub.add_parser("eval")
    e.add_argument("sentences", type=Path)
    e.add_argument("--budget", type=int, default=300)
    e.add_argument("--shuffle", action="store_true")
    args = p.parse_args()
    {"data": build_data, "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
