#!/usr/bin/env python3
"""loop_v2.py — Fully autonomous research loop.

Designed by GLM 5.3 Prime.

Loop: HEALTH → PICK → PROPOSE → VALIDATE → DISPATCH → COLLECT → JUDGE → LOG → DEEPEN → repeat

Design contract:
  - Local Xeon NEVER executes CUDA code
  - All GPU work dispatches to remote 4090 via dispatch.py
  - Budget managed automatically; stops when budget hits floor
  - Remote loss handled with backoff retry
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────

AUTOMATION_DIR = ROOT / "automation"
STATE_DIR = AUTOMATION_DIR / "state_local"
STAGING_DIR = STATE_DIR / "staging"
JOURNAL_PATH = STATE_DIR / "journal.jsonl"
LEDGER_PATH = STATE_DIR / "ledger.json"
REGISTRY_PATH = AUTOMATION_DIR / "experiments.json"
CAPABILITIES_PATH = AUTOMATION_DIR / "state" / "capabilities.json"

# ── Environment ───────────────────────────────────────────────────────────────

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")

# Load OpenRouter key if needed
def _ensure_env():
    env_path = Path.home() / ".config" / "openrouter" / "env"
    if env_path.exists() and "OPENROUTER_API_KEY" not in os.environ:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                os.environ.setdefault(k, v)

_ensure_env()

# ── Budget ────────────────────────────────────────────────────────────────────

BUDGET_START = 9.18
BUDGET_FLOOR = 0.40
CURRENT_SPENT = 0.80  # from 20 prior cycles + 2 GLM calls

# ── LLM Model Config ─────────────────────────────────────────────────────────

MODELS = {
    "propose": {"slug": "qwen/qwen3.8-max-prime", "max_tokens": 16000},
    "judge": {"slug": "deepseek/deepseek-v4.1-flash", "max_tokens": 2000},
    "deepen": {"slug": "z-ai/glm-5.2", "max_tokens": 4000},
    "patch": {"slug": "qwen/qwen3.8-max-prime", "max_tokens": 16000},
}

# Max tokens a reasoning model may burn before it is allowed to emit content.
MAX_TOKEN_CAP = 32000

# ── Helpers ───────────────────────────────────────────────────────────────────


def slug_for(role: str) -> str:
    return os.environ.get(f"OPENROUTER_{role.upper()}_MODEL", MODELS[role]["slug"])


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return key


def call_llm(messages: list, role: str = "propose", max_tokens: int | None = None,
             json_mode: bool = False, _retry: bool = True) -> str:
    """Call an OpenRouter LLM. Returns response text.

    Handles reasoning models (which can put everything into a 'reasoning' field
    and leave 'content' empty). If the model is cut off by the token limit before
    emitting content, retry once with double the budget. `json_mode` forces a
    JSON object response (falls back to a normal call if the model rejects it).
    """
    import urllib.request
    import urllib.error
    config = MODELS[role]
    slug = slug_for(role)
    mt = max_tokens or config["max_tokens"]
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    body = {
        "model": slug,
        "messages": messages,
        "max_tokens": mt,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + api_key(), "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError:
        # Some providers/models reject response_format; retry once without it.
        if json_mode and _retry:
            return call_llm(messages, role, max_tokens, json_mode=False, _retry=False)
        raise
    choice = payload["choices"][0]
    content = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning") or ""
    finish = choice.get("finish_reason")
    usage = payload.get("usage", {})
    cost = usage.get("cost", 0)
    global CURRENT_SPENT
    CURRENT_SPENT += cost
    _update_ledger(slug, role, usage, cost)
    # Reasoning model burned the whole budget in `reasoning` -> no content emitted.
    if _retry and not content.strip() and finish == "length" and mt < MAX_TOKEN_CAP:
        print(f"    (truncated at {mt} tokens, retrying with {min(mt*2, MAX_TOKEN_CAP)})")
        return call_llm(messages, role, min(mt * 2, MAX_TOKEN_CAP), json_mode, _retry=False)
    # If content is empty but reasoning is present, use reasoning as a last resort.
    return content if content.strip() else reasoning


def normalize_script_path(script: str, name: str) -> str:
    """Force experiment scripts under experiments/ so the bootstrap path
    `os.path.dirname(__file__)/../automation` resolves to <root>/automation.

    A root-level script (e.g. foo.py) resolves `../automation` to <parent>/automation,
    which breaks `from env_common import ...`. Every generated experiment must live
    one level below the project root.
    """
    script = (script or "").strip().lstrip("./")
    if not script:
        return f"experiments/{name}.py"
    # Strip a leading "experiments/" then re-add exactly once to avoid nesting.
    base = script
    if base.startswith("experiments/"):
        base = base[len("experiments/"):]
    base = os.path.basename(base) or f"{name}.py"
    if not base.endswith(".py"):
        base += ".py"
    return f"experiments/{base}"


def _update_ledger(slug, role, usage, cost):
    """Append to the local budget ledger."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "slug": slug,
        "role": role,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": cost,
    }
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def budget_remaining() -> float:
    return BUDGET_START - CURRENT_SPENT


# ── Crash / stall audio alert ─────────────────────────────────────────────────

ALARM_SOUND = ROOT / "Alarm Sound Effect.m4a"


def play_alert(reason: str = "") -> None:
    """Play the alarm sound for crash/stall events. Never fatal if audio is down."""
    print(f"\n  [ALERT] {reason or 'alert'} — playing alarm\n", flush=True)
    if not ALARM_SOUND.exists():
        print(f"  [ALERT] sound file missing: {ALARM_SOUND}", flush=True)
        return
    players = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(ALARM_SOUND)],
        ["cvlc", "--play-and-exit", "--intf", "dummy", str(ALARM_SOUND)],
        ["mpv", "--no-video", "--really-quiet", str(ALARM_SOUND)],
        ["paplay", str(ALARM_SOUND)],
    ]
    for cmd in players:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, timeout=25,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                continue
    print("  [ALERT] no working audio player found", flush=True)


def fetch_credits() -> tuple[float, float] | None:
    """Return (total_credits, total_usage) from OpenRouter, or None on failure."""
    import urllib.request
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    try:
        req = urllib.request.Request(
            base + "/credits",
            headers={"Authorization": "Bearer " + api_key()},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            d = json.loads(resp.read().decode())["data"]
        return float(d["total_credits"]), float(d["total_usage"])
    except Exception:
        return None


# ── Remote dispatch (using dispatch.py subprocess — simpler, proven working) ──────

DISPATCH_PY = str(ROOT / "scripts" / "dispatch.py")


def remote_health() -> dict:
    """Check remote GPU via dispatch.py health."""
    import subprocess, json
    try:
        r = subprocess.run(
            [sys.executable, DISPATCH_PY, "health"],
            capture_output=True, text=True, timeout=60,
        )
        return json.loads(r.stdout.strip())
    except (json.JSONDecodeError, Exception):
        return {"ok": False}


def _dispatch_run(script_rel: str, timeout: int = 600) -> dict:
    """Run a script on remote via dispatch.py. Returns parsed JSON result."""
    import subprocess, json
    try:
        r = subprocess.run(
            [sys.executable, DISPATCH_PY, "run",
             "--script", script_rel, "--timeout", str(timeout)],
            capture_output=True, text=True, timeout=timeout + 120,
        )
        # dispatch.py outputs JSON to stdout
        return json.loads(r.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except (json.JSONDecodeError, Exception) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _dispatch_push(local_path: str, remote_rel: str) -> bool:
    """Push a file to remote via dispatch.py. Returns True on success."""
    import subprocess, json
    try:
        r = subprocess.run(
            [sys.executable, DISPATCH_PY, "push",
             "--local", local_path, "--remote", remote_rel],
            capture_output=True, text=True, timeout=120,
        )
        result = json.loads(r.stdout.strip())
        return result.get("ok", False)
    except Exception:
        return False


# ── Experiment Registry ──────────────────────────────────────────────────────


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"experiments": [], "backlog": [], "ledger": {}}


def save_registry(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def pick_next(reg: dict) -> dict | None:
    """Pick the next experiment to run. Priority: retry > seed > backlog > free."""
    for exp in reg.get("experiments", []):
        if exp.get("status") in ("pending", "failed") and exp.get("retries", 0) < exp.get("max_retries", 2):
            return exp
    return None


# ── Safety Validation ──────────────────────────────────────────────────────


FORBIDDEN_PATTERNS = [
    r"requires_grad\s*=\s*True",
    r"\.train\(\)",
    r"backward\(",
    r"rm\s+-rf",
    r"os\.system",
    r"shutil\.rmtree",
]


def validate_code(code: str) -> list[str]:
    """Validate experiment code. Returns list of violations (empty = valid)."""
    violations = []
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            violations.append(f"forbidden pattern: {pat}")
    # py_compile
    try:
        compile(code, "<experiment>", "exec")
    except SyntaxError as e:
        violations.append(f"compile error: {e}")
    # Size check
    if len(code) > 20000:
        violations.append(f"code too large: {len(code)} chars > 20000")
    return violations


# ── LLM Prompts ──────────────────────────────────────────────────────────────


PROPOSE_SYSTEM = """You are an autonomous research scientist embedded in the MMAGAIProject closed loop.

HARDWARE:
- Remote GPU: RTX 4090 (24GB VRAM) at root@n1.msk.cloreai.ru:1837
- torch 2.6.0+cu124, cuda=True
- All experiments run on this GPU via SSH dispatch

ENVIRONMENT SETUP (IMPORTANT — your script MUST include this at the top):
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env_common import experiment_main, DEVICE
```

REQUIRED FUNCTION SIGNATURE:
```python
def run(out_dir: str, args: dict) -> dict:
    # args contains experiment parameters
    # Return a dict with at least 'ok': True/False
```

DECORATOR:
```python
@experiment_main
def run(out_dir, args):
    ...
```

AVAILABLE HELPERS (in env_common):
- `attach_world_model()` — loads frozen LeWM to DEVICE (cached). Returns model.
- `find_lance_dataset()` — returns (lance_dataset, path, format_string) or (None, None, None)
- `action_stats_from_dataset()` — returns dict with mean, std, n or None
- `encode_frame(model, frame: np.ndarray) -> torch.Tensor` — encode to latent z (192,)
- `make_env()` — creates Push-T gym env (render_mode='rgb_array')
- `DEVICE` — str, 'cuda' or 'cpu'

AVAILABLE SCRIPTS (can import from):
- `scripts.eval_official_cem` — official CEM eval (import scripts.eval_official_cem as cem)
- `scripts.oracle_rotate_cem` — in-place rotation CEM

CONSTRAINTS:
- Encoder is FROZEN — no requires_grad_(True), no .train(), no .backward()
- transformers must stay <5 (already enforced on remote)
- All GPU operations happen on the remote 4090 automatically

KEY FINDINGS SO FAR:
- Image-goal CEM: 45/50 (90%) with z-scored actions
- Ridge angle MAE from latent z: 12-15°
- Constant delta-z per command: ratio 0.991 (99% unexplained variance)
- In-place 180° rotation CEM: reach 0.00 at budget 50 and 100
- Probe-cost rotate 90: 0.10 reach (but block physically translated 115-130px)

You output ONLY a JSON object (no markdown, no other text):
```json
{{"name": "<short name>",
 "hypothesis": "<one sentence>",
 "script": "experiments/<name>.py",
 "code": "<complete python code following all rules above>",
 "params": {{}},
 "timeout_s": 600,
 "expected": "<what this will tell us>"}}
```"""  # noqa: E501

JUDGE_SYSTEM = """Grade this experiment result. Output JSON only:
{{"verdict": "success"|"partial"|"failure", "score": 0-10, "insights": ["..."], "next_ideas": ["..."], "suggest_deepen": false}}"""

DEEPEN_SYSTEM = """Synthesize across recent results. Output JSON only:
{{"ideas": [{{"title": "...", "rationale": "...", "sketch": "..."}}]}}"""

PATCH_SYSTEM = """Fix this experiment code given the error traceback. Output ONLY fenced python code (```python ... ```).

The code MUST:
1. Start with: import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE
2. Define a function: def run(out_dir: str, args: dict) -> dict:
3. Use @experiment_main decorator
4. No requires_grad_(True), no backward(), no train()
5. Return a dict with at least 'ok': True/False
6. Access the Lance dataset ONLY via env_common.find_lance_dataset() /
   env_common.action_stats_from_dataset(). There is no 'lance_data/' directory
   and no 'action_stats.pt' file — never hardcode dataset paths.

Fix ONLY the errors shown in the traceback. Preserve the experiment's intent."""


def propose_experiment(reg: dict) -> dict:
    """Call propose LLM to generate a new experiment. Tries primary then fallback."""
    # Build minimal context
    last_result = ""
    if JOURNAL_PATH.exists():
        lines = JOURNAL_PATH.read_text().strip().splitlines()
        if lines:
            last_result = lines[-1][:500]
    
    context = (
        "Available: frozen LeWM on RTX 4090, Lance dataset with action stats, "
        "env_common.py helper library at automation/env_common.py\n"
        "DATASET FACTS (already downloaded on the remote — NEVER hardcode a path):\n"
        "  path   = data/datasets/galilai-group--lewm-pusht/pusht_expert_train.lance\n"
        "  rows   = 2,336,736; columns: episode_idx, step_idx, action[2, float], pixels[binary], proprio[4], state[7]\n"
        "  access = from env_common import find_lance_dataset, action_stats_from_dataset\n"
        "           ds, path, fmt = find_lance_dataset();  stats = action_stats_from_dataset()\n"
        "  There is NO 'lance_data/' directory and NO 'action_stats.pt' file. Never reference them.\n"
        f"Key findings: CEM 45/50, rotation 0/20, delta-z ratio 0.991\n"
        f"Last result: {last_result}\n"
        f"Total cycles completed: {reg.get('ledger', {}).get('cycles_completed', 0)}\n\n"
        "Propose the next experiment. Output JSON ONLY: {name, hypothesis, script, code, params, timeout_s}"
    )
    
    text = call_llm([
        {"role": "system", "content": "You are an autonomous research scientist. Output ONLY a JSON object with keys: name, hypothesis, script, code, params, timeout_s. Keep `code` compact (<120 lines). Code MUST: import sys,os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'automation')); from env_common import experiment_main, DEVICE. Access the dataset ONLY via env_common.find_lance_dataset() / env_common.action_stats_from_dataset() — never hardcode dataset paths (there is no lance_data/ dir). Define: def run(out_dir, args) -> dict. Use @experiment_main. No requires_grad_(True)."},
        {"role": "user", "content": context},
    ], role="propose", json_mode=True)

    parsed = _extract_json(text)
    if parsed.get("name"):
        return parsed

    # Dump the raw output so we can see why parsing failed.
    try:
        (ROOT / ".cycle").mkdir(exist_ok=True)
        dump = ROOT / ".cycle" / f"propose_raw_{int(time.time())}.txt"
        dump.write_text(text or "")
        print(f"  (raw propose output dumped to {dump})")
    except Exception:
        pass

    # Fallback: same proposer model, re-ask with an even more rigid instruction.
    print("  Primary propose failed, retrying with rigid JSON instruction...")
    text2 = call_llm([
        {"role": "system", "content": "Return ONLY a single JSON object. Keys exactly: name, hypothesis, script, code, params, timeout_s. No prose, no markdown."},
        {"role": "user", "content": context[:4000]},
    ], role="propose", json_mode=True)

    parsed2 = _extract_json(text2)
    if not parsed2.get("name"):
        try:
            dump = ROOT / ".cycle" / f"propose_raw_fallback_{int(time.time())}.txt"
            dump.write_text(text2 or "")
            print(f"  (raw fallback output dumped to {dump})")
        except Exception:
            pass
    return parsed2


def judge_result(experiment: dict, result: dict) -> dict:
    """Call judge LLM to evaluate experiment result."""
    prompt = JUDGE_SYSTEM
    context = {
        "experiment": experiment["name"],
        "purpose": experiment.get("purpose", ""),
        "hypothesis": experiment.get("hypothesis", ""),
        "result": {k: v for k, v in result.items() if k != "stdout_tail"},
        "log_tail": (result.get("stdout_tail") or "")[:1500],
    }
    text = call_llm([
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(context)},
    ], role="judge")
    return _extract_json(text)


def deepen_ideas() -> list:
    """Call deepen LLM to generate follow-up ideas."""
    text = call_llm([
        {"role": "system", "content": DEEPEN_SYSTEM},
        {"role": "user", "content": f"Recent journal:\n{JOURNAL_PATH.read_text()[-3000:] if JOURNAL_PATH.exists() else 'None'}"},
    ], role="deepen")
    parsed = _extract_json(text)
    return parsed.get("ideas", [])


def patch_code(experiment: dict, error: str) -> str | None:
    """Call patch LLM to fix experiment code."""
    current_code = experiment.get("code", "")
    text = call_llm([
        {"role": "system", "content": PATCH_SYSTEM},
        {"role": "user", "content": f"Code:\n{current_code}\n\nError:\n{error}\n\nFix it."},
    ], role="patch")
    # Extract fenced code
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return text.strip()


def _extract_json(text: str) -> dict:
    """Extract first JSON object from text (handles markdown fences and trailing text)."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
        clean = re.sub(r"\n?\s*```$", "", clean)
    decoder = json.JSONDecoder()
    i = 0
    while True:
        m = clean.find("{", i)
        if m < 0:
            break
        try:
            v, e = decoder.raw_decode(clean, m)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            i = m + 1
            continue
        i = e
    return {}


# ── Loop State ──────────────────────────────────────────────────────────────


def load_loop_state() -> dict:
    state_path = STATE_DIR / "loop_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"cycle": 0, "last_health": None, "consecutive_failures": 0}


def save_loop_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "loop_state.json").write_text(json.dumps(state, indent=2))


def log_journal(entry: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main Loop ────────────────────────────────────────────────────────────────


def run_cycle(cycle_id: int, state: dict) -> dict:
    """Run one full cycle: PICK → (PROPOSE) → DISPATCH → COLLECT → JUDGE → LOG.
    
    Returns result dict.
    """
    print(f"\n{'='*60}")
    print(f"CYCLE {cycle_id}")
    print(f"{'='*60}")
    
    # 1. HEALTH
    health = remote_health()
    if not health["ok"]:
        print(f"  Remote unhealthy, sleeping 60s...")
        time.sleep(60)
        health = remote_health()
        if not health["ok"]:
            return {"cycle": cycle_id, "status": "skipped", "reason": "remote_unhealthy"}
    print(f"  Remote: {health.get('gpu', {}).get('name', 'ok')}")
    
    # 2. Budget check
    if budget_remaining() < BUDGET_FLOOR:
        print(f"  Budget exhausted: ${budget_remaining():.2f} remaining")
        return {"cycle": cycle_id, "status": "stopped", "reason": "budget_exhausted"}
    
    # 3. PICK or PROPOSE experiment
    reg = load_registry()
    exp = pick_next(reg)
    
    if exp is None:
        # Propose from backlog or free propose
        print("  Proposing new experiment via LLM (deepseek-v4-flash)...")
        proposal = propose_experiment(reg)
        if not proposal.get("name"):
            print("  LLM proposal failed, sleeping 30s")
            time.sleep(30)
            return {"cycle": cycle_id, "status": "skipped", "reason": "proposal_failed"}
        
        # Register the experiment
        exp = {
            "name": proposal["name"],
            "hypothesis": proposal.get("hypothesis", ""),
            "script": normalize_script_path(proposal.get("script", ""), proposal["name"]),
            "code": proposal.get("code", ""),
            "params": proposal.get("params", {}),
            "timeout_s": proposal.get("timeout_s", 1800),
            "status": "pending",
            "created_by": "llm",
            "purpose": proposal.get("expected", proposal.get("purpose", "")),
            "retries": 0,
            "max_retries": 3,
        }
        
        # Write and validate the experiment code
        if exp.get("code"):
            violations = validate_code(exp["code"])
            if violations:
                print(f"  Code invalid: {violations[0]}. Asking LLM to fix...")
                # Let the patch model fix it
                fixed = patch_code(exp, f"Validation error: {violations[0]}")
                if fixed:
                    exp["code"] = fixed
                    violations = validate_code(fixed)
                    if violations:
                        print(f"  Still invalid: {violations[0]}")
                        return {"cycle": cycle_id, "status": "skipped", "reason": f"invalid: {violations[0]}"}
            
            # Write experiment file
            exp_path = ROOT / exp["script"]
            exp_path.parent.mkdir(parents=True, exist_ok=True)
            exp_path.write_text(exp["code"])
        
        reg["experiments"].append(exp)
        save_registry(reg)
    
    # Normalize whichever experiment we ended up with (also covers pre-existing
    # registry entries written before the experiments/ path fix).
    exp["script"] = normalize_script_path(exp.get("script", ""), exp.get("name", "experiment"))
    if exp.get("code"):
        _norm_path = ROOT / exp["script"]
        _norm_path.parent.mkdir(parents=True, exist_ok=True)
        if not _norm_path.exists():
            _norm_path.write_text(exp["code"])

    # 4. DISPATCH — push code and run on remote GPU
    exp["status"] = "running"
    save_registry(reg)
    print(f"  Experiment: {exp['name']} ({exp.get('hypothesis', '')[:80]})")
    
    exp_path = ROOT / exp["script"]
    if not exp_path.exists() and not exp.get("code"):
        print(f"    No script/code — dispatching raw hypothesis")
        result = {"ok": False, "returncode": -1, "result": None, "stdout_tail": "no code", "elapsed_s": None}
    else:
        if exp_path.exists():
            _dispatch_push(str(exp_path), exp["script"])
            print(f"    Script: {exp['script']}")
        else:
            print(f"    (code written inline)")
        
        # Push shared libs
        for core in ["automation/env_common.py", "automation/remote_runner.py"]:
            p = ROOT / core
            if p.exists():
                _dispatch_push(str(p), core)
        
        run_timeout = exp.get("timeout_s", 1800) + 300
        print(f"    Dispatching to 4090 (timeout={run_timeout}s)...")
        result = _dispatch_run(
            exp["script"],
            timeout=run_timeout,
        )
    
    # 5. COLLECT
    if not result.get("stdout_tail"):
        result["stdout_tail"] = (result.get("stdout") or "")[-1500:]
    print(f"  Result: ok={result.get('ok')}, elapsed={result.get('elapsed_s', '?')}s")
    if result.get("result"):
        print(f"    metrics: {json.dumps(result['result'])[:220]}")
    
    # 6. JUDGE
    verdict = None
    insights = []
    if result.get("result"):
        verdict = "success" if result["result"].get("ok") else "failure"
        judge_output = judge_result(exp, result)
        verdict = judge_output.get("verdict", verdict or "unknown")
        insights = judge_output.get("insights", [])
    
    if verdict is None:
        verdict = "success" if result.get("ok") else "failure"
    
    print(f"  Verdict: {verdict}")
    
    # 7. Patch on failure: give traceback to LLM to fix code
    if verdict != "success" and exp.get("retries", 0) < exp.get("max_retries", 2):
        traceback_text = result.get("stderr", "") or result.get("stdout_tail", "") or ""
        if traceback_text and exp.get("code"):
            print(f"  Patching code via qwen-coder...")
            fixed = patch_code(exp, traceback_text)
            if fixed and fixed != exp.get("code"):
                exp["code"] = fixed
                # Rewrite the experiment file
                exp_path = ROOT / exp["script"]
                exp_path.parent.mkdir(parents=True, exist_ok=True)
                exp_path.write_text(fixed)
                print(f"    Code rewritten ({len(fixed)} chars)")
                # Push updated code to remote
                _dispatch_push(str(exp_path), exp["script"])
    
    # 8. Update registry
    exp["status"] = "done" if verdict == "success" else "failed"
    if verdict != "success":
        exp["retries"] = exp.get("retries", 0) + 1
        if exp["retries"] < exp.get("max_retries", 2):
            exp["status"] = "pending"  # Retry next cycle with patched code
    
    exp["last_result"] = result.get("result", {})
    exp["last_verdict"] = verdict
    save_registry(reg)
    
    # 8. Journal
    entry = {
        "cycle": cycle_id,
        "ts": datetime.utcnow().isoformat(),
        "experiment": exp["name"],
        "verdict": verdict,
        "insights": insights[:3],
        "elapsed_s": result.get("elapsed_s"),
        "budget_remaining": round(budget_remaining(), 4),
    }
    log_journal(entry)
    
    # 9. Deepen (occasionally)
    if verdict == "success" and cycle_id % 5 == 0 and budget_remaining() > 2:
        print("  Deepening...")
        ideas = deepen_ideas()
        if ideas:
            reg.setdefault("backlog", []).extend(ideas)
            save_registry(reg)
    
    return {"cycle": cycle_id, "status": "done", "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description="MMAGAI Autonomous Research Loop v2")
    ap.add_argument("--max-cycles", type=int, default=None, help="Max cycles to run")
    ap.add_argument("--once", action="store_true", help="Run one cycle and exit")
    ap.add_argument("--pause", type=int, default=30, help="Sleep between cycles (s)")
    ap.add_argument("--budget-floor", type=float, default=BUDGET_FLOOR)
    args = ap.parse_args()
    
    print("╔══════════════════════════════════════════════════╗")
    print("║   MMAGAI Autonomous Research Loop v2            ║")
    print("║   Designed by GLM 5.3 Prime                     ║")
    print("╚══════════════════════════════════════════════════╝")
    # Sync budget to the REAL OpenRouter balance rather than hard-coded seeds.
    global BUDGET_START, CURRENT_SPENT
    creds = fetch_credits()
    if creds:
        BUDGET_START, CURRENT_SPENT = creds
        print(f"OpenRouter: credits=${BUDGET_START:.2f} usage=${CURRENT_SPENT:.2f} "
              f"remaining=${budget_remaining():.2f}")
    else:
        print(f"Budget (local seed): start=${BUDGET_START:.2f} spent=${CURRENT_SPENT:.2f}")
    print(f"Floor: ${args.budget_floor:.2f}")
    print(f"Remote: root@n1.msk.cloreai.ru:1837")
    
    # Check remote immediately
    health = remote_health()
    if health["ok"]:
        gpu = health.get("gpu", {})
        print(f"Remote GPU: {gpu.get('name', 'RTX 4090')} ({gpu.get('mem_total_mb', '24GB')})")
    else:
        print("WARNING: Remote GPU not reachable. Loop will retry.")
    
    # Ensure local dirs
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    
    loop_state = load_loop_state()
    cycles_run = 0

    try:
        while True:
            if args.max_cycles and cycles_run >= args.max_cycles:
                print(f"\nReached max cycles ({args.max_cycles})")
                break

            if budget_remaining() < args.budget_floor:
                print(f"\nBudget exhausted (${budget_remaining():.2f} < ${args.budget_floor:.2f})")
                break

            loop_state["cycle"] += 1
            result = run_cycle(loop_state["cycle"], loop_state)
            save_loop_state(loop_state)
            cycles_run += 1

            if result.get("status") == "stopped":
                break

            if args.once:
                break

            print(f"\nSleeping {args.pause}s...")
            time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nInterrupted by operator (not a crash).")
    except Exception:
        # Genuine crash -> this is the ONLY place the alarm should sound.
        print("\n!!! LOOP CRASHED — unhandled exception !!!", flush=True)
        traceback.print_exc()
        play_alert("autonomous loop crashed (unhandled exception)")
        raise

    print(f"\nLoop finished. {cycles_run} cycles completed.")
    print(f"Budget remaining: ${budget_remaining():.4f}")


if __name__ == "__main__":
    main()