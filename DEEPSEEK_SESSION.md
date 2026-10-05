# DEEPSEEK Session State

## Last Updated
Sun Oct 4 2026, evening MSK (machine TZ = UTC+3; loop logs use UTC)

## >>> RESUME / CRASH-RECOVERY QUICK START <<<

Preferred (auto-restarts on crash, sounds the alarm ONLY on a real crash):
```bash
cd /home/alex/GitRepos/MMAGAIProjectOpen
pgrep -af loop_guard.sh || { source scripts/env.sh; nohup scripts/loop_guard.sh & }
```
Direct (no guard):
```bash
cd /home/alex/GitRepos/MMAGAIProjectOpen
rm -rf automation/__pycache__ scripts/__pycache__     # stale .pyc bug
source scripts/env.sh
python -u automation/loop_v2.py --max-cycles 60 --pause 3 > .cycle/loop_v2_run.out 2>&1 &
```
**Monitor:** `tail -40 .cycle/loop_v2_run.out` · `cat automation/state_local/journal.jsonl` ·
`cat automation/state_local/ledger.json` · `cat automation/experiments.json`

**Never** `pkill -f loop_v2` from an OpenCode shell — the pattern matches the shell
itself. Kill by PID.

## Current Status
- Loop runs under **`scripts/loop_guard.sh`** (background). Guard restarts on
  python crash; stops on clean exit / 4h timeout / operator signal.
- Real budget is synced at startup from OpenRouter `/credits` (no more fake seeds).
- Last verified: cycle 46 propose→dispatch→fee(d)back→patch all working.

## Alarm policy (user requirement)
The alarm (`Alarm Sound Effect.m4a`, played with `ffplay`) sounds **only on a
genuine crash**, never for notifications or normal events:
- `loop_v2.py` → `play_alert()` called *only* in the top-level `except Exception`
  (unhandled crash). It then re-raises → exit 1.
- `loop_guard.sh` → plays alarm *only* for unexpected death (e.g. OOM/SIGKILL);
  rc 1 (loop already alarmed) restarts quietly, rc 0/124/130/143 stop quietly.

## Fixes applied this session (all in `automation/loop_v2.py` + `scripts/dispatch.py`)
1. **Experiment path bug (root cause of all early failures).** Scripts written at
   project root made `sys.path.insert(dirname/../automation)` resolve wrong →
   `ModuleNotFoundError: env_common`. Added `normalize_script_path()`; every
   experiment is forced to `experiments/<name>.py`.
2. **Propose model stall.** `qwen3.8-max-prime` is a reasoning model; with a big
   prompt it spent the whole token budget on `reasoning` and returned empty
   `content` (`finish_reason: length`). Fixes:
   - `call_llm(..., json_mode=True)` sets `response_format={"type":"json_object"}`
     (falls back if the provider rejects it);
   - retries once with 2× max_tokens when truncated before content;
   - propose/patch `max_tokens` raised 8000 → 16000 (cap 32000);
   - request timeout 90s → 180s;
   - raw output dumped to `.cycle/propose_raw_*.txt` when JSON parsing fails;
   - fallback now re-asks the proposer (previously used the judge, also a
     reasoning model, so it failed the same way).
3. **Metrics were thrown away.** `dispatch.run_blocking` now creates a remote run
   dir, sets `LEWM_JOB_DIR`, `cd`s into the project, and returns parsed
   `metrics.json` as `result["result"]`. The judge now sees real metrics/tracebacks.
4. **Budget was fictional.** `main()` now reads real credits from OpenRouter and
   sets `BUDGET_START`/`CURRENT_SPENT` accordingly.

## Models (from `~/.config/openrouter/env`)
| Role | Model |
|---|---|
| propose / patch | `qwen/qwen3.8-max-prime` |
| judge | `deepseek/deepseek-v4.1-flash` |
| deepen | `z-ai/glm-5.2` |

Balance at last check: **$35 total, ~$25.8 remaining**.

## Remote GPU Machine
- `root@n1.msk.cloreai.ru` port **1837** — RTX 4090, auth via `~/.ssh/id_ed25519`
  (no passphrase, no stored password).
- `torch 2.6.0+cu124`, `.venv/bin/python`, `automation/env_common.py`,
  `experiments/` present.
- **Lance dataset IS present** (verified): `data/datasets/galilai-group--lewm-pusht/
  pusht_expert_train.lance`, **14 GB, 2,336,736 rows**. Schema: `episode_idx`,
  `step_idx`, `action[2]`, `pixels(binary)`, `proprio[4]`, `state[7]`.
  Access via `env_common.find_lance_dataset()` / `action_stats_from_dataset()`
  (mean `[-0.0078, 0.0069]`, std `[0.208, 0.206]`). There is **no** `lance_data/`
  dir and **no** `action_stats.pt` — the LLM used to invent those; prompts now say so.

## Progress so far (current loop)
| Cycle | Experiment | Verdict | Note |
|---|---|---|---|
| 1–3 | cem_cycle_diagnostic | failure | path bug |
| 4 | cem_cycle_diagnostic | failure | transient blip |
| 5 | cem_cycle_diagnostic | **success** | path fix validated |
| 7–42 | rotation/action probes | 2 success / 21 failure | metrics not collected |
| 43–45 | (propose stall) | skipped | reasoning-model truncation |
| 46 | cem_rotation_delta_z_probe | failure | metrics now visible; patched |
| 47+ | (guard) | running | real self-correction enabled |

## Known Issues / Watch-list
1. Stale `__pycache__` — delete on restart.
2. `loop_v2.py` stdout is buffered unless `-u` is used; guard always passes `-u`.
3. Generated experiments frequently buggy (e.g. `_main() takes 0 positional
   arguments but 2 were given`) — now the patch model gets the real traceback and
   retries, so this is expected to converge.
4. `experiments/env_common.py` exists (a prior proposal was literally named
   `env_common.py`) — harmless (not on the import path) but noisy.
5. Only ONE loop must run: `pgrep -af loop_v2.py`.
