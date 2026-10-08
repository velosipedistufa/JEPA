# Handoff: demo breadcrumbs on the lab 5090

Branch `5090-run-demo-breadcrumbs`. Written 2026-10-08 on Faith's laptop, to continue on the lab RTX 5090 PC. Nothing here has run on a GPU yet. `master` is untouched.

If you are Claude on the GPU PC: read this whole file first, then do the checks in order. Faith wants plain explanations with simple pictures first and jargon after. Do not push to `master`. Ask Faith before any push. Never write a GitHub token into a file, a remote URL, or a commit.

## The idea in plain words

LeWM is the robot's imagination. It turns a camera frame into 192 numbers and predicts how those numbers change when the pusher moves. The planner (CEM) tries hundreds of push sequences in that imagination and picks the one that ends closest to a goal frame.

That planner works well when the goal is about 25 steps away (45/50). It fails when the goal is far (the real Push-T task from a random reset: 0/20 at budget 50, `research/artifacts/011`). It also fails when the goal is a state the human demos never show (in-place turns, rendered breadcrumbs at +30/+60/+90: 1/20, `research/artifacts/012`; there the block was mostly not touched at all).

So we give it breadcrumbs taken from real expert demos. Think of a learner driver who can only drive 25 m at a time: instead of drawing fake photos, look through the dashcam recordings of experts, find the moment that looks like where we are now, and follow the next frames of that recording. The recordings are the 2.3M-frame Push-T demo set. Different recordings are linked where they pass through nearly the same state, so a route can switch from one demo to another.

Caselli et al. 2026 (Hi-LeWM, arXiv:2607.12547) showed that future frames from the same demo work as subgoals (73.3% at offset 50). They used the demo the start came from. Our version must find the route in other demos, because a new situation has no demo of its own. That is the difference, and it is what the robot would need too.

Later (check 3) words pick the goal: "put the T on the target", "turn the T clockwise". The graph then finds a route of real demo states to a goal that matches the words. That is Sasha's "language names reachable goals" idea, but with long tasks made reachable.

## What the script does

`scripts/demo_breadcrumbs.py`. It imports Sasha's `scripts/eval_official_cem.py`, so the model loading, the F13 action z-score, and CEM 300/30/top 30 with horizon 5 and action block 5 are the same code as the 45/50 run.

- Graph: one node every 5 demo steps, train episodes only (90%, split seed 0). Forward edges follow a demo. Jump edges link nodes from different demos closer than 15 in a state distance (agent xy and block xy in px, angle as 60 px per radian).
- Each hop: find the node near the current env state with the cheapest route to the goal. Walk about 25 demo steps along the route. Render that demo state, encode it, plan one CEM plan of 25 env steps toward it, and execute it. Then read the graph again from wherever the block is now. When the route reaches the goal, the planner chases the real goal frame.
- Retrieval uses the env state, not the latent. On a real robot that state comes from a tag on the T. Latent retrieval is a later variant.
- `--planner flat` is the same loop with no graph. It is the fair baseline at the same budget.
- `--dry` builds the graph and reports whether routes exist. No model, no CEM, so it is fast.
- Results go to `research/artifacts/5090/*.json`.

Graph logic was tested on synthetic demos on the laptop (route across two demos, 25-step breadcrumb, disconnected goal gives no route). The model and Lance parts have not run anywhere yet.

## Setup on the 5090

Linux is easiest (Sasha's paths and the checkpoint symlink assume it). On Windows, `data/checkpoints/models--quentinll--lewm-pusht` checks out as a text file. Replace it with a copy of `data/hf_pusht/` under that name.

```bash
git clone https://github.com/velosipedistufa/JEPA.git
cd JEPA
git switch 5090-run-demo-breadcrumbs

uv venv --python 3.11 .venv
source .venv/bin/activate
# 5090 is Blackwell: needs a CUDA 12.8 build of torch and an NVIDIA driver >= 570
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install 'stable-worldmodel[train]==0.1.1' 'transformers>=4.50,<5' \
  pygame pymunk shapely opencv-python-headless gymnasium imageio \
  scikit-learn scipy matplotlib huggingface_hub
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If the second install replaced torch with a CPU build, reinstall the cu128 wheel and check again. Keep `transformers` below 5: version 5 renames the ViT weights and `weights.pt` will not load. Do not run `scripts/make_lance_optional.py` (that patch is only for Sasha's old Xeon). `data/hf_pusht/weights.pt` is in the repo (about 69 MB). If it is a small text pointer instead, download it with `hf download quentinll/lewm-pusht --local-dir data/hf_pusht`.

Demo dataset, about 14 GB, the Lance table Sasha used on the 4090:

```bash
hf download galilai-group/lewm-pusht --repo-type dataset \
  --local-dir data/datasets/galilai-group--lewm-pusht
```

The scripts look for any `*.lance` folder under `data/datasets/galilai-group--lewm-pusht/` and prefer `pusht_expert_train`. If the repo id or type differs, check the Hugging Face page. Only the folder location matters.

## Checks, in order

Each check is written before it runs, in the same style as `research/LEDGER.md`: a claim, and the number that kills it. Do not change a constant after seeing a scored result. Thresholds may be tuned only with `--dry`, before any scored run.

| # | Command | Claim | Kill / stop if |
|---|---|---|---|
| 0 | `python scripts/eval_official_cem.py 50` | Setup matches Sasha's 4090: about 45/50 | at most 40/50. Then fix the setup before anything else. |
| 1a | `python scripts/demo_breadcrumbs.py --task dataset --offset 50 --dry --n 50` | Most held-out tasks have a route through other demos | fewer than 35/50 with a route. Then look at `entry_gap` / `goal_gap` and tune `JUMP_RADIUS` in dry mode only |
| 1b | same with `--task live --dry --n 20` | Report only. Live goals are random full states, so coverage may be poor | — |
| 2a | `--task dataset --offset 50 --planner flat --n 50` | Baseline. Hi-LeWM reports flat 52.7% at offset 50 after a sweep | — (it is the number to beat) |
| 2b | `--task dataset --offset 50 --planner crumbs --n 50` | Crumbs beat 2a by at least 10 points | crumbs minus flat is at most 2 points (1 episode) |
| 2c | offset 100, flat then crumbs, `--n 50` | Gap grows with distance | crumbs not above flat |
| 2d | `--task live`, flat then crumbs, `--n 20` | Crumbs reach at least 4/20 where flat stays lower | crumbs at most flat + 1 |

All scored runs use `--budget 300`. Expect each CEM plan to take under a second on a 5090, so one 50-episode run should take minutes. Graph build over about 420k nodes should take seconds to a minute.

Report back to Faith: the `SUMMARY` line of each run, and the JSON files under `research/artifacts/5090/`. Commit them on this branch with a short plain message. Do not push without asking.

## Results on the 5090 (2026-10-08)

The JSON files are in `research/artifacts/5090/`. Scored runs use budget 300.

| # | Result | Verdict |
|---|---|---|
| 0 | official CEM 47/50 (4090: 45/50) | setup holds |
| 1a | 50/50 with a route, median 0.5 jumps | holds |
| 1b | 13/20 with a route; median entry/goal gap 37/39 px | report |
| 2a / 2b | offset 50: flat 24/50, crumbs 34/50 (+20 pts; paired 16 crumbs-only vs 6 flat-only) | holds |
| 2c | offset 100: flat 10/50, crumbs 29/50 (+38 pts; paired 23 vs 4) | holds, gap grows |
| 2d | live: flat 0/20, crumbs 3/20; median block error 162 → 58 px | between claim (4) and kill (1) |
| 3b | words: flat 18/64, crumbs 34/64 (+25 pts; paired 24 vs 8) | holds |
| 3c | turns: crumbs 8/20 (flat 4/20). The 012 gate (drift under 15 px): crumbs 0/20, flat 1/20 | holds on the frozen gate. The turns happen while the block drifts 15–40 px, not strictly in place |

Crumbs per command: target 6/10, turn cw 5/10, turn ccw 3/10, left 4/8, right 4/8, up 7/10, down 5/8.

## Check 3, after check 2 holds (not written yet)

Words pick the goal region instead of a goal image:

- "put the T on the target": block pose near the env goal pose `(256, 256, π/4)`, agent anywhere.
- "turn the T 90° clockwise" / "counter-clockwise": block angle `θ0 ∓ 90°` (check the sign convention in the env first), block xy within about 40 px of the start.
- "push the T left / right / up / down": block xy moved about 80 px, angle free.

Goal nodes are the graph nodes that satisfy the command. Route as in check 2. The last target is the best goal node's rendered frame. Success is scored on the block only. Start with a fixed list of command templates (honest and easy to score). A sentence-embedding model for paraphrases is optional on top. Stretch for the "generative" part: a small network that produces the next breadcrumb latent from (current latent, sentence), trained on the routes the graph found, compared against the graph itself.

The dry graph also answers a side question for the report: do the demos ever turn the T in place? Count nodes whose block moves less than 15 px while turning more than 90° within 50 steps.

### Check 3 as run (written 2026-10-08 on the 5090, before any scored check-3 run)

Script: `scripts/word_breadcrumbs.py`. It reuses the check-2 graph and planner code unchanged.

- Sign convention, checked on the env render: screen y points down, so "up" is −y. A positive block angle turns the T clockwise on screen, so clockwise is θ0 + 90°.
- Seven commands: target, turn cw, turn ccw, and push left/right/up/down.
- Tolerances are the env's own `eval_state` ones: 20 px and 20°. Turns allow 40 px of xy drift. Pushes move the block 80 px with the angle free.
- A push whose target point leaves the box [80, 432] is dropped before running.
- Starts are live resets, seeds 0–9. That gives 64 command/start pairs (6 pushes dropped).
- Both planners chase the same goal image: the rendered demo node that the graph route from the start ends at. Flat chases it for the whole budget. Crumbs adds the check-2 breadcrumbs.
- Budget 300. Success is scored on the block only, at any step.

The dry run (`--dry --n 10`) went first and changed no constant:
- 56/64 pairs have a route.
- In-place turns are in the demos: 8,040 of 427k nodes, in 1,691 of 16,817 episodes.

| # | Command | Claim | Kill / stop if |
|---|---|---|---|
| 3a | `--dry --n 10` | Report only. 56/64 with a route | — |
| 3b | `--planner flat --n 10`, then `--planner crumbs --n 10` | Crumbs beat flat by at least 10 points on the 64 pairs | crumbs at most flat + 1 episode |
| 3c | the turn rows of 3b (20 pairs) | Crumbs reach at least 5/20 turns | crumbs at most 2/20. Also report the stricter 012 gate (drift under 15 px), report only |

## Where things are

- Sasha's index of measured numbers: `reports/HANDOFF.md`, `research_os/lab/facts.md`, `research/LEDGER.md`.
- Official planner: `scripts/eval_official_cem.py`. Live-goal run: `scripts/live_goal_cem.py`. Rendered breadcrumbs: `scripts/staged_rotate_cem.py`.
- After the course: Faith and Didar plan the same method on the lab UR3 (camera over a table, 3D-printed T with a tag, teleop demos, LeWM retrained on those). Keep the code free of sim-only shortcuts where possible, so it can carry over.
