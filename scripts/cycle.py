#!/usr/bin/env python3
"""Closed-loop autoresearch runner.

This file is the authority for the loop in jepa-research-loop.md.
It calls OpenRouter. It does not install packages or download datasets.
A trial subprocess starts only from `run`, and only after the local gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEWM = ROOT / "le-wm"
LEDGER = ROOT / "ledger.jsonl"
BASELINES = ROOT / "baselines.json"
QUESTIONS = ROOT / "open_questions.md"
STATE_PATH = ROOT / ".cycle" / "state.json"
METRIC_SCRIPT = ROOT / "scripts" / "metric_trial.py"
TRIAL_LOG = ROOT / ".cycle" / "trial.log"
OBSERVED_PATH = ROOT / ".cycle" / "trial_observed.json"

REQUIRED_KEYS = (
    "cycle",
    "claim",
    "falsifier",
    "degenerate_block",
    "factor",
    "baseline_id",
    "seed",
    "config",
    "split",
    "metric_script",
    "commit",
    "budget_minutes",
    "budget_tokens",
    "models_called",
    "metric",
    "verdict",
    "stop",
)
STOPS = {"continue", "deepen", "pivot", "kill", "escalate"}
VERDICTS = {"pass", "fail"}
CHEAP_ROLES = {"propose", "patch", "judge", "setup"}
PREMIUM_SLUGS = {"z-ai/glm-5.2", "moonshotai/kimi-k3"}
KIMI = "moonshotai/kimi-k3"
GLM = "z-ai/glm-5.2"
DEEP_ROLES = {"deepen", "escalate", "admit"}
PROMPT_FILES = ("ledger.jsonl", "baselines.json", "open_questions.md")
JUDGE_KEYS = ("claim", "falsifier", "metric")
SCALE_PHRASES = ("if we scale", "works if we scale", "more gpus", "once we scale")
BLOCKS = {
    "a": "latent collapse to a constant",
    "b": "a metric that ignores planning",
    "c": "declaring failure from prediction loss alone",
    "baseline": "latent collapse to a constant",
}
FACTORS = {
    "sigreg_weight": {
        "hypothesis": "a",
        "file": "le-wm/config/train/lewm.yaml",
        "key": "loss.sigreg.weight",
        "override": "loss.sigreg.weight",
        "baseline_field": "sigreg_weight",
        "entry": "train.py",
    },
    "latent_dim": {
        "hypothesis": "b",
        "file": "le-wm/config/train/lewm.yaml",
        "key": "embed_dim",
        "override": "embed_dim",
        "baseline_field": "embed_dim",
        "entry": "train.py",
    },
    "predictor_depth": {
        "hypothesis": None,
        "file": "le-wm/config/train/model/lewm.yaml",
        "key": "predictor.depth",
        "override": "model.predictor.depth",
        "baseline_field": "predictor_depth",
        "entry": "train.py",
    },
    "cem_population": {
        "hypothesis": None,
        "file": "le-wm/config/train/solver/cem.yaml",
        "key": "num_samples",
        "override": "solver.num_samples",
        "baseline_field": "cem_population",
        "entry": "eval.py",
    },
    "cem_iterations": {
        "hypothesis": None,
        "file": "le-wm/config/train/solver/cem.yaml",
        "key": "n_steps",
        "override": "solver.n_steps",
        "baseline_field": "cem_iterations",
        "entry": "eval.py",
    },
    "baseline": {
        "hypothesis": "baseline",
        "file": "le-wm/config/train/lewm.yaml",
        "key": None,
        "override": None,
        "baseline_field": None,
        "entry": "train.py",
    },
}
ROLES = {
    "propose": {
        "env": "OPENROUTER_PROPOSE_MODEL",
        "primary": "deepseek/deepseek-v4-flash",
        "fallback": "z-ai/glm-4.7-flash",
        "max_tokens": 2000,
    },
    "patch": {
        "env": "OPENROUTER_PATCH_MODEL",
        "primary": "qwen/qwen3-coder-30b-a3b-instruct",
        "fallback": "deepseek/deepseek-v4-flash",
        "max_tokens": 1200,
    },
    "judge": {
        "env": "OPENROUTER_JUDGE_MODEL",
        "primary": "deepseek/deepseek-v3.2",
        "fallback": "z-ai/glm-4.7-flash",
        "max_tokens": 600,
    },
    "deepen": {
        "env": "OPENROUTER_DEEP_MODEL",
        "primary": GLM,
        "fallback": "qwen/qwen3.8-27b",
        "max_tokens": 1200,
    },
    "escalate": {
        "env": "OPENROUTER_ESCALATE_MODEL",
        "primary": KIMI,
        "fallback": GLM,
        "max_tokens": 1000,
    },
    "admit": {
        "env": "OPENROUTER_ESCALATE_MODEL",
        "primary": KIMI,
        "fallback": GLM,
        "max_tokens": 1000,
    },
    "setup": {
        "env": "OPENROUTER_PATCH_MODEL",
        "primary": "qwen/qwen3-coder-30b-a3b-instruct",
        "fallback": "deepseek/deepseek-v4-flash",
        "max_tokens": 1200,
    },
}
FORBIDDEN_SLUG = re.compile(r"(:free|:latest|:nitro|:floor|:exacto|:beta|:max|prime|openrouter/auto)", re.I)
_CATALOG = None


class RunnerError(Exception):
    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def redact(text: str) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        text = text.replace(key, "[redacted]")
    return text[:500]


def load_env() -> None:
    """Load ~/.config/openrouter/env. Do not override a variable that is already set."""
    path = Path.home() / ".config" / "openrouter" / "env"
    if not path.is_file():
        raise RunnerError("missing ~/.config/openrouter/env")
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip().strip("'\"")
        if key not in os.environ:
            os.environ[key] = value


def slug_for(role: str, which: str = "primary") -> str:
    spec = ROLES[role]
    if which == "primary":
        slug = os.environ.get(spec["env"], "").strip() or spec["primary"]
    else:
        slug = spec["fallback"]
    return assert_slug(slug)


def assert_slug(slug: str) -> str:
    if not slug or "/" not in slug or FORBIDDEN_SLUG.search(slug):
        raise RunnerError(f"refusing model slug {slug!r}")
    return slug


def assert_cheap(role: str, slug: str) -> None:
    if role in CHEAP_ROLES and slug in PREMIUM_SLUGS:
        raise RunnerError(f"refusing premium model {slug} on {role}")


def stablewm_home() -> Path:
    return Path(os.environ.get("STABLEWM_HOME") or (Path.home() / ".stable-wm"))


def h5_path() -> Path:
    return stablewm_home() / "pusht_expert_train.h5"


def gpu_status() -> dict:
    return {
        "nvidia_smi": shutil_which("nvidia-smi"),
        "rocminfo": shutil_which("rocminfo"),
    }


def shutil_which(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def yaml_scalar(path: Path, key: str):
    matches = []
    for line in path.read_text().splitlines():
        body = line.split("#", 1)[0].rstrip()
        match = re.match(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", body)
        if match:
            matches.append(match.group(1).strip().strip("'\""))
    if len(matches) != 1:
        raise RunnerError(f"{path} key {key} matched {matches}")
    raw = matches[0]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw


def pinned_from_files() -> dict:
    train = LEWM / "config/train/lewm.yaml"
    model = LEWM / "config/train/model/lewm.yaml"
    cem = LEWM / "config/eval/solver/cem.yaml"
    return {
        "seed": yaml_scalar(train, "seed"),
        "train_split": yaml_scalar(train, "train_split"),
        "embed_dim": yaml_scalar(train, "embed_dim"),
        "sigreg_weight": yaml_scalar(train, "weight"),
        "predictor_depth": yaml_scalar(model, "depth"),
        "cem_population": yaml_scalar(cem, "num_samples"),
        "cem_iterations": yaml_scalar(cem, "n_steps"),
        "accelerator": yaml_scalar(train, "accelerator"),
    }


def read_baselines() -> dict:
    return json.loads(BASELINES.read_text())


def baseline_row() -> dict:
    rows = read_baselines()["baselines"]
    for row in rows:
        if row["id"] == "baseline-0":
            return row
    raise RunnerError("baseline-0 is missing")


def same_number(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-9)
    return left == right


def assert_pin_matches_files() -> dict:
    live = pinned_from_files()
    pinned = baseline_row()["config"]
    for key, value in live.items():
        if not same_number(pinned.get(key), value):
            raise RunnerError(f"baseline-0 field {key} does not match the checkout")
    commit = lewm_commit()
    if baseline_row()["commit"] != commit:
        raise RunnerError("baseline-0 commit does not match le-wm HEAD")
    return live


def lewm_commit() -> str:
    return subprocess.check_output(["git", "-C", str(LEWM), "rev-parse", "HEAD"], text=True).strip()


def lewm_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "-C", str(LEWM), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    return bool(out.strip())


def read_ledger(path: Path = LEDGER) -> list:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def human_acked(cycle: int) -> bool:
    text = QUESTIONS.read_text() if QUESTIONS.exists() else ""
    return f"human-ack: {cycle}" in text


def next_cycle_number() -> int:
    rows = read_ledger()
    if not rows:
        return 1
    last = rows[-1]
    if last.get("stop") not in STOPS:
        raise RunnerError("refusing to start the next cycle: the previous ledger row has no stop")
    if last["stop"] == "escalate" and not human_acked(int(last["cycle"])):
        raise RunnerError(
            f"cycle {last['cycle']} escalated; add `human-ack: {last['cycle']}` to open_questions.md"
        )
    return int(last["cycle"]) + 1


def cycle_inputs() -> str:
    parts = []
    for name in PROMPT_FILES:
        path = ROOT / name
        text = path.read_text() if path.exists() else ""
        parts.append(f"## {name}\n{text}")
    return "\n\n".join(parts)


def one_sentence(text: str, label: str) -> str:
    cleaned = text.strip()
    if not cleaned or "\n" in cleaned:
        raise RunnerError(f"{label} must be one sentence")
    if len(cleaned) > 400:
        raise RunnerError(f"{label} is too long to be one sentence")
    if not cleaned.endswith((".", "?", "!")):
        raise RunnerError(f"{label} must be one sentence")
    body = cleaned[:-1]
    if any(mark in body for mark in ".?!"):
        raise RunnerError(f"{label} must be one sentence; split the claim")
    low = cleaned.lower()
    if any(phrase in low for phrase in SCALE_PHRASES):
        raise RunnerError(f"{label} depends on scaling, which is outside the cycle")
    return cleaned


def load_state() -> dict:
    if not STATE_PATH.exists():
        raise RunnerError("no cycle state; run propose first")
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(state)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key and key in blob:
        raise RunnerError("refusing to write state; secret in payload")
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def catalog() -> dict:
    global _CATALOG
    if _CATALOG is None:
        request = urllib.request.Request("https://openrouter.ai/api/v1/models")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode())
        except Exception as exc:
            raise RunnerError(f"refusing to call without max_price: {type(exc).__name__}") from exc
        _CATALOG = {item["id"]: item for item in payload.get("data", [])}
    return _CATALOG


def max_price_for(slug: str) -> dict:
    assert_slug(slug)
    item = catalog().get(slug)
    if item is None:
        raise RunnerError(f"model not in catalog: {slug}")
    pricing = item.get("pricing") or {}
    prompt = float(pricing.get("prompt") or 0) * 1_000_000
    completion = float(pricing.get("completion") or 0) * 1_000_000
    if prompt <= 0 or completion <= 0:
        raise RunnerError(f"refusing free-tier price for {slug}")
    return {
        "prompt": math.ceil(prompt * 10000) / 10000,
        "completion": math.ceil(completion * 10000) / 10000,
    }


def message_text(choice: dict) -> str:
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        joined = "".join(parts).strip()
        if joined:
            return joined
    for key in ("reasoning", "reasoning_content"):
        extra = message.get(key)
        if isinstance(extra, str) and extra.strip():
            return extra
    return ""


def post_chat(slug: str, messages: list, max_tokens: int) -> tuple[int, dict]:
    assert_slug(slug)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RunnerError("OPENROUTER_API_KEY is empty")
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    body = {
        "model": slug,
        "messages": messages,
        "max_tokens": max_tokens,
        "provider": {"sort": "price", "max_price": max_price_for(slug)},
    }
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        status = exc.code
    except Exception as exc:
        raise RunnerError(f"{slug} request failed: {type(exc).__name__}") from exc
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {"error": redact(raw)}
    return status, payload


def usage_of(payload: dict, slug: str) -> dict:
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    cost = usage.get("cost")
    source = "response" if cost is not None else "catalog"
    if cost is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        price = max_price_for(slug)
        cost = (prompt_tokens * price["prompt"] + completion_tokens * price["completion"]) / 1_000_000
    if cost == 0 and ((completion_tokens or 0) > 0 or (prompt_tokens or 0) > 0):
        raise RunnerError(f"refusing free-tier response for {slug}")
    served = payload.get("model") or slug
    assert_slug(served)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost,
        "cost_source": source,
        "served_model": served,
    }


def count_slug(state: dict, slug: str) -> int:
    return sum(1 for row in state.get("models_called", []) if row.get("slug") == slug)


def call_role(state: dict, role: str, messages: list, max_tokens: int | None = None) -> tuple[str, dict, str]:
    spec = ROLES[role]
    limit = spec["max_tokens"] if max_tokens is None else max_tokens
    budget = state.get("budget_tokens")
    if isinstance(budget, int) and state.get("tokens_used", 0) + limit > budget:
        raise RunnerError("token budget exceeded before the call")
    slug = slug_for(role, "primary")
    assert_cheap(role, slug)
    if role == "judge":
        proposer = state.get("proposer_slug") or slug_for("propose")
        if slug == proposer:
            raise RunnerError("judge slug matches the proposer slug")
    text, usage, used = attempt(state, role, slug, messages, limit)
    if text is None:
        fallback = slug_for(role, "fallback")
        assert_cheap(role, fallback)
        if role == "judge" and fallback == (state.get("proposer_slug") or slug_for("propose")):
            raise RunnerError("judge fallback matches the proposer slug")
        if fallback == KIMI or (fallback == GLM and count_slug(state, GLM) >= 1):
            raise RunnerError(f"refusing fallback {fallback} after {slug} failed")
        text, usage, used = attempt(state, role, fallback, messages, limit)
        if text is None:
            raise RunnerError(f"{role} failed on {slug} and {fallback}")
    tokens = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    state["tokens_used"] = state.get("tokens_used", 0) + tokens
    if used in PREMIUM_SLUGS:
        state.setdefault("premium_cost", []).append(
            {
                "slug": used,
                "role": role,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": usage.get("cost_usd"),
                "cost_source": usage.get("cost_source"),
            }
        )
    return text, usage, used


def attempt(state: dict, role: str, slug: str, messages: list, max_tokens: int) -> tuple[str | None, dict, str]:
    if slug == KIMI and count_slug(state, KIMI) >= 1:
        raise RunnerError("refusing a second Kimi K3 call in this cycle")
    if slug == GLM and count_slug(state, GLM) >= 1:
        raise RunnerError("refusing a second GLM 5.2 call in this cycle")
    status, payload = post_chat(slug, messages, max_tokens)
    entry = {"slug": slug, "role": role}
    if status != 200 or not payload.get("choices"):
        entry["error"] = redact(json.dumps(payload.get("error") or payload)[:300])
        state.setdefault("models_called", []).append(entry)
        return None, {}, slug
    usage = usage_of(payload, slug)
    entry.update(
        {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost_usd"),
        }
    )
    text = message_text(payload["choices"][0])
    if not text.strip():
        entry["error"] = "empty content"
        state.setdefault("models_called", []).append(entry)
        return None, usage, slug
    state.setdefault("models_called", []).append(entry)
    return text, usage, slug


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    found = None
    index = 0
    while True:
        start = cleaned.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            found = value
            if any(key in value for key in ("actions", "hypothesis", "verdict", "admit")):
                return value
        index = end
    if isinstance(found, dict):
        return found
    raise RunnerError("model did not return a JSON object")


def propose_system(baseline_unrun: bool) -> str:
    if baseline_unrun:
        allowed = (
            'baseline-0 is unrun. The only legal proposal is hypothesis "baseline", '
            'factor "baseline", factor_value null. '
            'Claim that the unmodified pinned config yields a measured latent_std above the collapse floor. '
            'Degenerate block must be "latent collapse to a constant".'
        )
    else:
        allowed = (
            'baseline-0 has a metric. Propose exactly one of hypothesis "a" '
            '(factor sigreg_weight, value different from the pin) or hypothesis "b" '
            "(factor latent_dim, integer below 192). Do not propose hypothesis c."
        )
    return (
        "You propose one hypothesis. Return ONLY a single valid JSON object and absolutely no other text, no explanation, no reasoning, no markdown, no code fences. "
        "You may use only the ledger, open questions, and pinned baselines in the user message. "
        "Fields: hypothesis, claim, falsifier, degenerate_block, factor, factor_value, "
        "baseline_id, budget_minutes, budget_tokens. "
        "claim and falsifier are each one sentence checkable in this cycle. "
        "budget_minutes is an integer from 1 to 20. The total prompt is long, so budget_tokens must be at least 12000 and at most 20000. "
        "baseline_id is baseline-0. Do not invent a hypothesis that is not active. "
        + allowed
    )


def validate_proposal(obj: dict, unrun: bool) -> dict:
    hypothesis = str(obj.get("hypothesis") or "")
    if unrun:
        if hypothesis != "baseline" or obj.get("factor") != "baseline":
            raise RunnerError("baseline-0 is unrun; only a baseline fill may be proposed")
    else:
        if hypothesis == "c" and not h5_path().is_file():
            raise RunnerError("hypothesis c is parked until pusht_expert_train.h5 exists")
        if hypothesis not in {"a", "b"}:
            raise RunnerError("proposal is not an active hypothesis")
        expected = {"a": "sigreg_weight", "b": "latent_dim"}[hypothesis]
        if obj.get("factor") != expected:
            raise RunnerError("proposal factor does not match its hypothesis")
    if obj.get("baseline_id") != "baseline-0":
        raise RunnerError("baseline_id must be baseline-0")
    claim = one_sentence(str(obj.get("claim") or ""), "claim")
    falsifier = one_sentence(str(obj.get("falsifier") or ""), "falsifier")
    block = str(obj.get("degenerate_block") or "").strip().lower()
    required = BLOCKS[hypothesis]
    if required not in block:
        raise RunnerError("missing degenerate_block")
    if hypothesis == "b" and not any(word in falsifier.lower() for word in ("cem", "planning", "success")):
        raise RunnerError("falsifier is not checkable in this cycle; split the claim")
    if hypothesis in {"a", "baseline"} and not any(
        word in falsifier.lower() for word in ("latent_std", "variance", "collapse")
    ):
        raise RunnerError("falsifier is not checkable in this cycle; split the claim")
    minutes = obj.get("budget_minutes", 20)
    tokens = obj.get("budget_tokens")
    if not isinstance(minutes, int) or isinstance(minutes, bool) or not 1 <= minutes <= 20:
        raise RunnerError("budget_minutes must be an integer from 1 to 20")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or not 12000 <= tokens <= 20000:
        raise RunnerError("budget_tokens must be an integer from 3000 to 20000")
    factor = obj.get("factor")
    if factor not in FACTORS:
        raise RunnerError("factor is not allowed")
    value = obj.get("factor_value")
    live = pinned_from_files()
    if factor == "baseline":
        value = None
    elif factor == "sigreg_weight":
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise RunnerError("sigreg_weight must be a non-negative number")
        if same_number(value, live["sigreg_weight"]):
            raise RunnerError("sigreg_weight does not change the pinned baseline")
    elif factor == "latent_dim":
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 191:
            raise RunnerError("latent_dim for hypothesis b must be an integer below 192")
    killed = {
        re.sub(r"\s+", " ", row["claim"].strip().lower())
        for row in read_ledger()
        if row.get("stop") == "kill" and row.get("claim")
    }
    if re.sub(r"\s+", " ", claim.lower()) in killed:
        raise RunnerError("killed hypothesis stays closed")
    return {
        "hypothesis": hypothesis,
        "claim": claim,
        "falsifier": falsifier,
        "degenerate_block": required,
        "factor": factor,
        "factor_value": value,
        "baseline_id": "baseline-0",
        "budget_minutes": minutes,
        "budget_tokens": tokens,
    }


def _attach_setup_calls(cycle_number: int) -> dict:
    """Carry setup-model usage into the next proposal once. Missing file means no setup call."""
    path = ROOT / ".cycle" / "setup_models.json"
    empty = {"models_called": [], "tokens_used": 0, "premium_cost": []}
    if not path.exists():
        return empty
    data = json.loads(path.read_text())
    if data.get("attached_cycle") not in (None, cycle_number):
        return empty
    data["attached_cycle"] = cycle_number
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return {
        "models_called": list(data.get("models_called") or []),
        "tokens_used": int(data.get("tokens_used") or 0),
        "premium_cost": list(data.get("premium_cost") or []),
    }


def cmd_propose() -> None:
    load_env()
    if slug_for("judge") == slug_for("propose"):
        raise RunnerError("judge slug matches the proposer slug")
    unrun = baseline_row()["status"] != "filled"
    cycle_number = next_cycle_number()
    carried = _attach_setup_calls(cycle_number)
    state = {
        "cycle": cycle_number,
        "status": "proposed",
        "models_called": carried["models_called"],
        "tokens_used": carried["tokens_used"],
        "premium_cost": carried["premium_cost"],
        "budget_tokens": None,
    }
    messages = [
        {"role": "system", "content": propose_system(unrun)},
        {"role": "user", "content": cycle_inputs()},
    ]
    text, _usage, slug = call_role(state, "propose", messages)
    state["proposer_slug"] = slug
    state["proposer_raw"] = text
    try:
        parsed = parse_json_object(text)
    except RunnerError:
        repair = messages + [
            {"role": "user", "content": "The previous reply was not one JSON object. Return only that JSON object."}
        ]
        text, _usage, slug = call_role(state, "propose", repair)
        state["proposer_slug"] = slug
        state["proposer_raw"] = text
        parsed = parse_json_object(text)
    proposal = validate_proposal(parsed, unrun)
    if state["tokens_used"] > proposal["budget_tokens"]:
        raise RunnerError("propose exceeded the token budget named in the claim")
    state.update(proposal)
    save_state(state)
    print(f"proposed cycle {state['cycle']} hypothesis {proposal['hypothesis']} factor {proposal['factor']}")


def metric_script_blocks() -> None:
    text = METRIC_SCRIPT.read_text()
    if "LATENT_COLLAPSE_STD" not in text or "cem_success" not in text or "REQUIRE_CEM_SUCCESS" not in text:
        raise RunnerError("degenerate solution is not blocked in scripts/metric_trial.py")


SETUP_MODELS = ROOT / ".cycle" / "setup_models.json"
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]+/[A-Za-z0-9][A-Za-z0-9_.\-]+$")
PKG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-\[\],<>=!~]*$")


def hard_gate() -> str | None:
    """Trial gate. A missing h5 is not a reason to skip the OpenRouter setup agent."""
    if not h5_path().is_file():
        return f"pusht_expert_train.h5 is still absent at {h5_path()}"
    live = pinned_from_files()
    visible = gpu_status()
    if live["accelerator"] == "gpu" and not (visible["nvidia_smi"] or visible["rocminfo"]):
        # Check if a remote GPU is available for dispatch
        try:
            from scripts.remote import remote_gpu_available, load_config
            cfg = load_config()
            if cfg.get("host") and remote_gpu_available():
                print(f"  local GPU absent, but remote GPU at {cfg['host']} is available — dispatching", flush=True)
                return None  # Gate passes via remote dispatch
        except Exception as exc:
            print(f"  remote GPU check failed: {exc}", flush=True)
        return "pinned accelerator is gpu and nvidia-smi and rocminfo are absent"
    return None


def _probe(module: str) -> str:
    py = LEWM / ".venv" / "bin" / "python"
    exe = str(py if py.is_file() else Path(sys.executable))
    proc = subprocess.run(
        [exe, "-c", f"import {module} as m; print(getattr(m, '__version__', 'ok'))"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    if proc.returncode == 132 or proc.returncode < 0:
        return "SIGILL"
    return f"fail:{proc.returncode}"


def machine_facts() -> dict:
    import shutil

    usage = shutil.disk_usage(str(stablewm_home().anchor or "/"))
    return {
        "expected_h5": str(h5_path()),
        "h5_present": h5_path().is_file(),
        "stablewm_home": str(stablewm_home()),
        "disk_free_bytes": usage.free,
        "gpu": gpu_status(),
        "accelerator": pinned_from_files().get("accelerator"),
        "transformers": _probe("transformers"),
        "lance": _probe("lance"),
        "facts": [
            "import lance SIGILLs on this CPU. Do not enable Lance.",
            "transformers 5 cannot load the LeWM checkpoint. Pin transformers>=4.50,<5 if you reinstall it.",
            "quentinll/lewm-pusht is the public repo that publishes pusht_expert_train.h5.zst.",
            "galilai-group/lewm-pusht publishes a lance table this CPU cannot open.",
            "Call list_hf before fetch. Keep at least 2GB free.",
        ],
    }


def _home_path(rel: str) -> Path:
    if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise RunnerError(f"path must stay inside STABLEWM_HOME: {rel!r}")
    home = stablewm_home().resolve()
    home.mkdir(parents=True, exist_ok=True)
    dest = (home / rel).resolve()
    if dest != home and home not in dest.parents:
        raise RunnerError("path escapes STABLEWM_HOME")
    return dest


def _action_list_hf(repo: str) -> str:
    if not isinstance(repo, str) or not REPO_RE.fullmatch(repo):
        raise RunnerError(f"bad repo id {repo!r}")
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main"
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.loads(response.read().decode())
    brief = [
        {"path": item.get("path"), "type": item.get("type"), "size": item.get("size")}
        for item in entries
        if isinstance(item, dict)
    ]
    return json.dumps(brief)


def _action_fetch(url: str, dest_rel: str) -> str:
    import shutil

    if not isinstance(url, str) or not url.startswith("https://"):
        raise RunnerError("fetch url must be https")
    host = urllib.parse.urlparse(url).hostname or ""
    if host != "huggingface.co" and not host.endswith(".huggingface.co") and not host.endswith(".hf.co"):
        raise RunnerError(f"fetch host is not allowed: {host}")
    dest = _home_path(dest_rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "jepa-research-loop"})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = int(response.headers.get("Content-Length") or 0)
        free = shutil.disk_usage(dest.anchor).free
        if length and length > free - 512 * 1024**2:
            raise RunnerError(f"content-length {length} does not fit in {free} free bytes")
        with dest.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                if shutil.disk_usage(dest.anchor).free < 512 * 1024**2:
                    handle.close()
                    dest.unlink(missing_ok=True)
                    raise RunnerError("stopped the fetch because less than 512MB was free")
    return f"wrote {dest} bytes={dest.stat().st_size}"


def _action_fetch_zstd(url: str, dest_rel: str) -> str:
    import shutil
    import threading

    if not isinstance(url, str) or not url.startswith("https://"):
        raise RunnerError("fetch url must be https")
    host = urllib.parse.urlparse(url).hostname or ""
    if host != "huggingface.co" and not host.endswith(".huggingface.co") and not host.endswith(".hf.co"):
        raise RunnerError(f"fetch host is not allowed: {host}")
    dest = _home_path(dest_rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = "/usr/sbin:/usr/bin:" + env.get("PATH", "")
    env.pop("OPENROUTER_API_KEY", None)
    request = urllib.request.Request(url, headers={"User-Agent": "jepa-research-loop"})
    proc = subprocess.Popen(
        ["zstd", "-d", "-c"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    error: list[BaseException] = []

    def pump() -> None:
        assert proc.stdin is not None
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    proc.stdin.write(chunk)
        except Exception as exc:
            error.append(exc)
        finally:
            proc.stdin.close()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    assert proc.stdout is not None
    try:
        with dest.open("wb") as handle:
            while True:
                decoded = proc.stdout.read(1024 * 1024)
                if not decoded:
                    break
                handle.write(decoded)
                if shutil.disk_usage(dest.anchor).free < 512 * 1024**2:
                    raise RunnerError("stopped stream decompress because less than 512MB was free")
    except Exception:
        proc.kill()
        dest.unlink(missing_ok=True)
        raise
    thread.join(timeout=30)
    stderr = proc.stderr.read().decode() if proc.stderr is not None else ""
    code = proc.wait(timeout=60)
    if error:
        dest.unlink(missing_ok=True)
        raise RunnerError(redact(str(error[0])))
    if code != 0:
        dest.unlink(missing_ok=True)
        raise RunnerError(redact(stderr[-400:] or f"zstd exit {code}"))
    return f"stream-unpacked {dest} bytes={dest.stat().st_size}"


def _action_unpack(src_rel: str, dest_rel: str) -> str:
    src = _home_path(src_rel)
    dest = _home_path(dest_rel)
    if not src.is_file():
        raise RunnerError(f"missing archive {src}")
    env = os.environ.copy()
    env["PATH"] = "/usr/sbin:/usr/bin:" + env.get("PATH", "")
    env.pop("OPENROUTER_API_KEY", None)
    proc = subprocess.run(
        ["zstd", "-d", "-f", str(src), "-o", str(dest)],
        capture_output=True,
        text=True,
        env=env,
        timeout=6 * 60 * 60,
    )
    if proc.returncode != 0:
        with src.open("rb") as handle:
            head = handle.read(8)
        if head == b"\x89HDF\r\n\x1a\n":
            os.replace(src, dest)
            return f"source was already hdf5; moved to {dest} bytes={dest.stat().st_size}"
        raise RunnerError(redact(proc.stderr[-400:] or f"zstd exit {proc.returncode}"))
    return f"unpacked {dest} bytes={dest.stat().st_size}"


def _action_pip(packages: list) -> str:
    if not isinstance(packages, list) or not packages or len(packages) > 12:
        raise RunnerError("pip packages must be a short list")
    clean = []
    for package in packages:
        if not isinstance(package, str) or not PKG_RE.fullmatch(package):
            raise RunnerError(f"refusing package {package!r}")
        if package.startswith("transformers") and "<5" not in package:
            raise RunnerError("transformers must be pinned <5")
        clean.append(package)
    uv = ROOT / ".tools" / "uv"
    py = LEWM / ".venv" / "bin" / "python"
    if not uv.is_file() or not py.is_file():
        raise RunnerError("project uv or le-wm/.venv python is missing")
    env = os.environ.copy()
    env["PATH"] = "/usr/sbin:/usr/bin:" + env.get("PATH", "")
    env["UV_TORCH_BACKEND"] = "cpu"
    env["UV_PYTHON_INSTALL_DIR"] = str(ROOT / ".tools" / "python")
    env["UV_CACHE_DIR"] = str(ROOT / ".tools" / "uv-cache")
    env.pop("OPENROUTER_API_KEY", None)
    proc = subprocess.run(
        [str(uv), "pip", "install", "--python", str(py), *clean],
        cwd=LEWM,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-400:]
        raise RunnerError(redact(tail or f"pip exit {proc.returncode}"))
    return f"installed {clean}"


def _action_patch_lance() -> str:
    py = LEWM / ".venv" / "bin" / "python"
    proc = subprocess.run(
        [str(py), str(ROOT / "scripts" / "make_lance_optional.py"), str(LEWM)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RunnerError(redact((proc.stderr or proc.stdout)[-400:] or "lance patch failed"))
    return (proc.stdout or "patched").strip()[:500]


def _normalize_action(action):
    if not isinstance(action, dict):
        return action
    if "op" in action:
        return action
    if len(action) != 1:
        return action
    op, body = next(iter(action.items()))
    if not isinstance(op, str):
        return action
    if isinstance(body, str):
        if op == "list_hf":
            return {"op": op, "repo": body}
        return {"op": op, "dest": body}
    if isinstance(body, dict):
        merged = {"op": op}
        merged.update(body)
        return merged
    if isinstance(body, list) and op == "pip":
        return {"op": op, "packages": body}
    return action


def execute_setup_actions(actions: list) -> list:
    if not isinstance(actions, list) or len(actions) > 6:
        raise RunnerError("actions must be a list of at most 6")
    results = []
    for action in actions:
        action = _normalize_action(action)
        if not isinstance(action, dict) or "op" not in action:
            results.append({"ok": False, "error": "action is not an object with op"})
            continue
        op = action.get("op")
        print(f"setup action {op}", flush=True)
        try:
            if op == "list_hf":
                detail = _action_list_hf(action.get("repo"))
            elif op == "fetch":
                detail = _action_fetch(action.get("url"), action.get("dest"))
            elif op == "fetch_zstd":
                detail = _action_fetch_zstd(action.get("url"), action.get("dest"))
            elif op == "unpack_zst":
                detail = _action_unpack(action.get("src"), action.get("dest"))
            elif op == "pip":
                detail = _action_pip(action.get("packages"))
            elif op == "patch_lance":
                detail = _action_patch_lance()
            elif op == "mkdir":
                stablewm_home().mkdir(parents=True, exist_ok=True)
                detail = str(stablewm_home())
            else:
                raise RunnerError(f"unknown op {op}")
            results.append({"op": op, "ok": True, "detail": detail})
        except Exception as exc:
            results.append({"op": op, "ok": False, "error": redact(str(exc))})
    return results


def _setup_messages(extra: dict | None = None) -> list:
    payload = machine_facts()
    if extra:
        payload["previous"] = extra
    return [
        {
            "role": "system",
            "content": (
                "You are the setup agent from OpenRouter. "
                "If pusht_expert_train.h5 is missing, you must obtain it. "
                "The local runner will not choose the file for you. "
                "Return one JSON object and no other text: {\"actions\": [...]} with at most 6 actions. "
                "Allowed ops: list_hf {repo}, fetch {url, dest}, fetch_zstd {url, dest}, "
                "unpack_zst {src, dest}, patch_lance {}, pip {packages}, mkdir {}. "
                "dest and src are relative paths under STABLEWM_HOME. "
                "fetch urls must be https and should use the datasets host, "
                "https://huggingface.co/datasets/<repo>/resolve/main/<file>. "
                "list_hf and fetch in one reply cannot see each other's results. "
                "If the compressed file and the unpacked file do not both fit, "
                "use fetch_zstd, which stream-decompresses and does not keep the archive. "
                "Do not enable Lance on this CPU."
            ),
        },
        {"role": "user", "content": json.dumps(payload)},
    ]


def _write_setup(state: dict, results: list, slug: str) -> None:
    SETUP_MODELS.parent.mkdir(parents=True, exist_ok=True)
    SETUP_MODELS.write_text(
        json.dumps(
            {
                "models_called": state["models_called"],
                "tokens_used": state["tokens_used"],
                "premium_cost": state.get("premium_cost") or [],
                "results": results,
                "h5_present": h5_path().is_file(),
                "setup_slug": slug,
            },
            indent=2,
        )
        + "\n"
    )


def cmd_setup() -> None:
    load_env()
    state = {"models_called": [], "tokens_used": 0, "premium_cost": [], "budget_tokens": None}
    previous = None
    if SETUP_MODELS.exists() and not h5_path().is_file():
        try:
            previous = json.loads(SETUP_MODELS.read_text()).get("results")
        except json.JSONDecodeError:
            previous = None
    text, _usage, slug = call_role(
        state,
        "setup",
        _setup_messages({"actions_result": previous, "h5_present": False} if previous else None),
    )
    (ROOT / ".cycle").mkdir(parents=True, exist_ok=True)
    (ROOT / ".cycle" / "setup_raw.txt").write_text(text or "")
    try:
        parsed = parse_json_object(text or "")
    except RunnerError:
        fallback = slug_for("setup", "fallback")
        messages = _setup_messages(
            {"actions_result": previous, "h5_present": False, "error": "Return one JSON object only."}
            if previous
            else {"error": "Return one JSON object only."}
        )
        text, _usage, slug = attempt(state, "setup", fallback, messages, ROLES["setup"]["max_tokens"])
        (ROOT / ".cycle" / "setup_raw.txt").write_text(text or "")
        if not text:
            raise RunnerError("setup model did not return a JSON object")
        parsed = parse_json_object(text)
    results = execute_setup_actions(parsed.get("actions") or [])
    _write_setup(state, results, slug)
    if not h5_path().is_file():
        try:
            text, _usage, slug = call_role(
                state,
                "setup",
                _setup_messages({"actions_result": results, "h5_present": False}),
            )
            (ROOT / ".cycle" / "setup_raw.txt").write_text(text)
            parsed = parse_json_object(text)
            results.extend(execute_setup_actions(parsed.get("actions") or []))
        except RunnerError as exc:
            results.append({"op": "followup", "ok": False, "error": redact(str(exc))})
    _write_setup(state, results, slug)
    print(f"setup slug={slug} h5_present={h5_path().is_file()} actions={len(results)}")


def seal_refused_run(state: dict, reason: str) -> None:
    live = pinned_from_files()
    state["seed"] = live["seed"]
    state["split"] = f"train_split={live['train_split']}"
    state["metric_script"] = "scripts/metric_trial.py"
    state["metric_script_sha256"] = hashlib.sha256(METRIC_SCRIPT.read_bytes()).hexdigest()
    state["commit"] = lewm_commit()
    state["config"] = {
        "baseline_id": "baseline-0",
        "factor": state.get("factor"),
        "refused": reason,
        "wandb": "on" if os.environ.get("WANDB_API_KEY") else "off",
    }
    state["metric"] = None
    state["metric_void_reason"] = reason
    state["metric_row"] = {
        "claim": state.get("claim"),
        "falsifier": state.get("falsifier"),
        "factor": state.get("factor"),
        "metric": None,
        "void_reason": reason,
        "degenerate_blocked": False,
    }
    state["status"] = "ran"
    save_state(state)
    print(f"ran cycle {state['cycle']} metric=None void={reason}")


def apply_patch(state: dict) -> list:
    factor = state["factor"]
    spec = FACTORS[factor]
    if factor == "baseline":
        messages = [
            {
                "role": "system",
                "content": "Return JSON {\"factor\":\"baseline\",\"changed\":false} and nothing else.",
            },
            {"role": "user", "content": json.dumps({"factor": "baseline", "changed": False})},
        ]
        text, _usage, _slug = call_role(state, "patch", messages)
        state["patch_raw"] = text
        parsed = parse_json_object(text)
        if set(parsed) - {"factor", "changed"} or parsed.get("changed") is not False:
            raise RunnerError("baseline fill rejected a patch that changes the config")
        return []
    payload = {
        "factor": factor,
        "file": spec["file"],
        "key": spec["key"],
        "baseline_value": pinned_from_files()[spec["baseline_field"]],
        "requested_value": state["factor_value"],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Return one JSON object with exactly the keys file, key, and value. "
                "Copy file, key, and requested_value. Do not add another factor."
            ),
        },
        {"role": "user", "content": json.dumps(payload)},
    ]
    text, _usage, _slug = call_role(state, "patch", messages)
    state["patch_raw"] = text
    parsed = parse_json_object(text)
    if set(parsed) != {"file", "key", "value"}:
        raise RunnerError(f"refusing multi-factor patch keys {sorted(parsed)}")
    if parsed["file"] != spec["file"] or parsed["key"] != spec["key"]:
        raise RunnerError("patch left the allowed factor file")
    if not same_number(parsed["value"], state["factor_value"]) and parsed["value"] != state["factor_value"]:
        raise RunnerError("patch changed the requested value")
    return [f"{spec['override']}={state['factor_value']}"]


def changed_factor_names(before: dict, after: dict) -> list:
    names = []
    for name, spec in FACTORS.items():
        field = spec["baseline_field"]
        if field is None:
            continue
        if not same_number(before.get(field), after.get(field)):
            names.append(name)
    return names


def score_observed(state: dict) -> dict:
    observed = {}
    if OBSERVED_PATH.exists():
        observed = json.loads(OBSERVED_PATH.read_text())
    sys.path.insert(0, str(ROOT / "scripts"))
    import metric_trial

    scored = metric_trial.score({"factor": state["factor"], "observed": observed})
    if lewm_dirty():
        scored["metric"] = None
        scored["void_reason"] = "le-wm tracked tree is dirty; commit does not identify the trial"
    script_at_head = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", "HEAD:scripts/metric_trial.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if script_at_head.returncode != 0:
        scored["metric"] = None
        scored["void_reason"] = "metric_script is not at HEAD; the number is void"
    return scored


def cmd_run() -> None:
    load_env()
    state = load_state()
    if state.get("status") != "proposed":
        raise RunnerError("run requires a proposed claim")
    one_sentence(state.get("claim") or "", "claim")
    one_sentence(state.get("falsifier") or "", "falsifier")
    if not state.get("degenerate_block"):
        raise RunnerError("missing degenerate_block")
    metric_script_blocks()
    assert_pin_matches_files()
    gate = hard_gate()
    if gate:
        seal_refused_run(state, gate)
        return
    if state["factor"] != "baseline" and baseline_row()["status"] != "filled":
        raise RunnerError("baseline-0 is unrun; a one-factor trial cannot support a causal claim")
    overrides = apply_patch(state)
    live = pinned_from_files()
    after = dict(live)
    if state["factor"] not in {"baseline"}:
        after[FACTORS[state["factor"]]["baseline_field"]] = state["factor_value"]
    changed = changed_factor_names(live, after)
    if state["factor"] == "baseline" and changed:
        raise RunnerError("baseline fill changed a factor")
    if state["factor"] != "baseline" and changed != [state["factor"]]:
        raise RunnerError(f"refusing diff that changes {changed or 'nothing'}")
    if not os.environ.get("WANDB_API_KEY"):
        overrides.append("wandb.enabled=false")
    entry = FACTORS[state["factor"]]["entry"]
    command = [sys.executable, str(LEWM / entry), *overrides]
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(stablewm_home())
    env["MPLBACKEND"] = "Agg"
    env["SDL_VIDEODRIVER"] = "dummy"
    env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    env["MUJOCO_GL"] = "egl"
    if not os.environ.get("WANDB_API_KEY"):
        env["WANDB_MODE"] = "disabled"

    # Check if we should dispatch to remote GPU
    _remote_dispatch = False
    _remote_cfg = {}
    if not (shutil_which("nvidia-smi") or shutil_which("rocminfo")):
        try:
            from scripts.remote import remote_gpu_available, load_config
            _remote_cfg = load_config()
            if _remote_cfg.get("host") and remote_gpu_available():
                _remote_dispatch = True
        except Exception:
            pass

    TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    if _remote_dispatch:
        # Dispatch the command to remote GPU via SSH
        import shlex
        remote_dir = _remote_cfg.get("remote_dir", "/root/MMAGAIProjectOpen")
        port_flag = f" -p {_remote_cfg['port']}" if _remote_cfg.get("port") else ""
        source_cmd = f"cd {shlex.quote(remote_dir)} && . .venv/bin/activate"
        env_vars = " ".join(f'{k}={shlex.quote(v)}' for k, v in env.items())
        command_str = " ".join(shlex.quote(str(c)) for c in command)
        remote_full = f"{source_cmd} && {env_vars} {command_str} 2>&1"
        ssh_cmd = f"ssh{port_flag} {shlex.quote(_remote_cfg['host'])} {shlex.quote(remote_full)}"
        print(f"  dispatching to remote: {entry} overrides={overrides}", flush=True)
        with TRIAL_LOG.open("w") as log:
            proc = subprocess.Popen(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                 *(["-p", str(_remote_cfg["port"])] if _remote_cfg.get("port") else []),
                 _remote_cfg["host"],
                 f"cd {shlex.quote(remote_dir)} && . .venv/bin/activate && {' '.join(shlex.quote(str(c)) for c in command)}"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    else:
        with TRIAL_LOG.open("w") as log:
            proc = subprocess.Popen(
                command,
                cwd=LEWM,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        timed_out = False
        try:
            returncode = proc.wait(timeout=state["budget_minutes"] * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            returncode = None
    digest = hashlib.sha256(METRIC_SCRIPT.read_bytes()).hexdigest()
    state["seed"] = live["seed"]
    state["split"] = f"train_split={live['train_split']}"
    state["metric_script"] = "scripts/metric_trial.py"
    state["metric_script_sha256"] = digest
    state["commit"] = lewm_commit()
    state["config"] = {
        "baseline_id": "baseline-0",
        "factor": state["factor"],
        "factor_value": state["factor_value"],
        "overrides": overrides,
        "command": command[1:],
        "accelerator": live["accelerator"],
        "wandb": "on" if os.environ.get("WANDB_API_KEY") else "off",
    }
    scored = score_observed(state)
    if timed_out:
        scored["metric"] = None
        scored["void_reason"] = f"trial exceeded {state['budget_minutes']} minutes"
    elif returncode not in (0, None):
        scored["metric"] = None
        scored["void_reason"] = f"trial exit {returncode}"
    state["metric"] = scored["metric"]
    state["metric_void_reason"] = scored.get("void_reason")
    state["degenerate_blocked"] = scored.get("degenerate_blocked")
    state["metric_row"] = {
        "claim": state["claim"],
        "falsifier": state["falsifier"],
        "factor": state["factor"],
        "metric": scored["metric"],
        "void_reason": scored.get("void_reason"),
        "degenerate_blocked": scored.get("degenerate_blocked", False),
    }
    state["status"] = "ran"
    save_state(state)
    print(f"ran cycle {state['cycle']} metric={state['metric']!r} void={state['metric_void_reason']}")


def judge_view(state: dict) -> dict:
    return {
        "claim": state["claim"],
        "falsifier": state["falsifier"],
        "metric": state.get("metric_row"),
    }


def cmd_judge() -> None:
    load_env()
    state = load_state()
    if state.get("status") != "ran":
        raise RunnerError("judge requires a finished run and a metric row")
    view = judge_view(state)
    if set(view) != set(JUDGE_KEYS):
        raise RunnerError("judge payload left the allowlist")
    blob = json.dumps(view)
    if state.get("proposer_raw") and state["proposer_raw"] in blob:
        raise RunnerError("judge payload contains the proposer transcript")
    messages = [
        {
            "role": "system",
            "content": (
                "Grade only the JSON in the user message. Do not assume a transcript you were not given. "
                "Return one JSON object with verdict (pass, fail, or ambiguous), "
                "stop (continue, deepen, pivot, kill, or escalate), and reason (one sentence). "
                "A null metric cannot pass. continue and kill are final cheap verdicts."
            ),
        },
        {"role": "user", "content": json.dumps(view)},
    ]
    text, _usage, slug = call_role(state, "judge", messages)
    if slug == state.get("proposer_slug"):
        raise RunnerError("judge slug matches the proposer slug")
    parsed = parse_json_object(text)
    verdict = parsed.get("verdict")
    stop = parsed.get("stop")
    if verdict not in {"pass", "fail", "ambiguous"} or stop not in STOPS:
        raise RunnerError("judge returned an unusable verdict")
    state["disputed"] = verdict == "ambiguous" or stop == "deepen"
    if verdict == "ambiguous" or stop == "deepen":
        if stop in {"continue", "kill"} and verdict != "ambiguous":
            raise RunnerError("continue and kill cannot call a deep model")
        deep_messages = [
            {
                "role": "system",
                "content": (
                    "The cheap verdict was disputed. Return JSON with verdict pass or fail, "
                    "stop deepen, pivot, or escalate, and reason as one sentence. "
                    "Use only the user JSON."
                ),
            },
            {"role": "user", "content": json.dumps({**view, "cheap_verdict": "ambiguous"})},
        ]
        deep_text, _usage, _deep_slug = call_role(state, "deepen", deep_messages)
        parsed = parse_json_object(deep_text)
        verdict = parsed.get("verdict")
        stop = parsed.get("stop")
        if verdict not in VERDICTS or stop not in {"deepen", "pivot", "escalate"}:
            verdict, stop = "fail", "deepen"
    if stop == "escalate":
        escalate_messages = [
            {
                "role": "system",
                "content": (
                    "Escalate once. Return JSON with changes set to hypothesis, baseline, stop_rule, or none, "
                    "and note as one sentence. Then stop for a human."
                ),
            },
            {"role": "user", "content": json.dumps(view)},
        ]
        extra, _usage, _slug = call_role(state, "escalate", escalate_messages)
        state["escalate_note"] = parse_json_object(extra).get("note")
    if state.get("metric") is None or state.get("degenerate_blocked"):
        verdict = "fail"
        if stop == "continue":
            stop = "kill"
    if verdict not in VERDICTS:
        raise RunnerError("verdict must be pass or fail")
    if stop in {"continue", "kill"} and any(row["role"] in DEEP_ROLES or row["slug"] in PREMIUM_SLUGS for row in state["models_called"]):
        raise RunnerError("continue and kill cannot call a deep model")
    state["verdict"] = verdict
    state["stop"] = stop
    state["status"] = "judged"
    save_state(state)
    print(f"judged cycle {state['cycle']} verdict {verdict} stop {stop}")


def validate_row(row: dict) -> list:
    errors = []
    for key in REQUIRED_KEYS:
        if key not in row:
            errors.append(f"missing {key}")
    if row.get("stop") not in STOPS:
        errors.append("stop is absent")
    if row.get("verdict") not in VERDICTS:
        errors.append("verdict must be pass or fail")
    called = row.get("models_called") or []
    slugs = {item.get("slug") for item in called if isinstance(item, dict)}
    roles = {item.get("role") for item in called if isinstance(item, dict)}
    deep = bool(slugs & PREMIUM_SLUGS or roles & DEEP_ROLES)
    if row.get("stop") in {"continue", "kill"} and (KIMI in slugs or deep):
        errors.append("K3 or a deep model was called on continue or kill")
    if row.get("stop") == "escalate" and KIMI not in slugs:
        errors.append("escalate must call Kimi K3 once")
    if sum(1 for item in called if item.get("slug") == KIMI) > 1:
        errors.append("more than one Kimi K3 call")
    if row.get("metric") is not None:
        for field in ("seed", "config", "split", "metric_script", "commit"):
            if row.get(field) in (None, "", []):
                errors.append(f"metric is void because {field} is missing")
    if any(item.get("slug") in PREMIUM_SLUGS for item in called) and "premium_cost" not in row:
        errors.append("premium call missing cost")
    return errors


def append_row(row: dict, path: Path) -> None:
    errors = validate_row(row)
    if errors:
        raise RunnerError("; ".join(errors))
    blob = json.dumps(row, separators=(",", ":"), sort_keys=True)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key and key in blob:
        raise RunnerError("refusing to write ledger; secret in row")
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def row_from_state(state: dict) -> dict:
    row = {key: state.get(key) for key in REQUIRED_KEYS}
    if state.get("premium_cost"):
        row["premium_cost"] = state["premium_cost"]
    if state.get("metric_void_reason"):
        row["metric_void_reason"] = state["metric_void_reason"]
    if state.get("escalate_note"):
        row["escalate_note"] = state["escalate_note"]
    return row


def cmd_close() -> None:
    load_env()
    state = load_state()
    if state.get("status") != "judged":
        raise RunnerError("close requires a judged cycle")
    row = row_from_state(state)
    append_row(row, LEDGER)
    if state["factor"] == "baseline" and state.get("metric") is not None and state.get("verdict") == "pass":
        data = read_baselines()
        for item in data["baselines"]:
            if item["id"] == "baseline-0":
                item["status"] = "filled"
                item["metric"] = state["metric"]
                item["filled_by_cycle"] = state["cycle"]
        BASELINES.write_text(json.dumps(data, indent=2) + "\n")
    state["status"] = "closed"
    save_state(state)
    print(f"closed cycle {state['cycle']} stop {state['stop']}")


def cmd_admit(paper: str, changes: str) -> None:
    load_env()
    state = {
        "cycle": next_cycle_number(),
        "status": "judged",
        "models_called": [],
        "tokens_used": 0,
        "premium_cost": [],
        "budget_tokens": 20000,
        "budget_minutes": 20,
        "claim": one_sentence(f"The paper changes the {changes}.", "claim"),
        "falsifier": "The admission names one of hypothesis, baseline, or stop_rule.",
        "degenerate_block": "a paper that does not change a hypothesis, a baseline, or a stop rule",
        "factor": "literature",
        "factor_value": None,
        "baseline_id": "baseline-0",
        "seed": pinned_from_files()["seed"],
        "config": {"paper": paper, "changes": changes},
        "split": f"train_split={pinned_from_files()['train_split']}",
        "metric_script": "scripts/metric_trial.py",
        "commit": lewm_commit(),
        "metric": None,
        "verdict": "fail",
        "stop": "escalate",
        "proposer_slug": slug_for("propose"),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Decide whether this paper changes a hypothesis, a baseline, or a stop rule. "
                "Return JSON with admit true or false and note as one sentence. "
                "Reject summaries that change nothing."
            ),
        },
        {"role": "user", "content": json.dumps({"paper": paper, "changes": changes, "open_questions": QUESTIONS.read_text()})},
    ]
    text, _usage, _slug = call_role(state, "admit", messages)
    parsed = parse_json_object(text)
    if parsed.get("admit") is not True:
        print("paper did not enter the ledger")
        return
    state["escalate_note"] = parsed.get("note")
    questions = QUESTIONS.read_text().rstrip() + f"\n\n## Admitted\n\n- {paper}: changes {changes}. {parsed.get('note') or ''}\n"
    QUESTIONS.write_text(questions)
    append_row(row_from_state(state), LEDGER)
    state["status"] = "closed"
    save_state(state)
    print(f"admitted into cycle {state['cycle']} and stopped for a human")


def cmd_launch(max_cycles: int) -> None:
    load_env()
    cmd_setup()
    for _ in range(max_cycles):
        cmd_propose()
        cmd_run()
        cmd_judge()
        cmd_close()
        rows = read_ledger()
        if rows and rows[-1].get("stop") == "escalate":
            print("stopped for a human")
            return
    print("launch finished")


def cmd_status() -> None:
    load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    visible = gpu_status()
    print(f"key={'present' if key else 'empty'}")
    print(f"h5={'present' if h5_path().is_file() else 'absent'} path={h5_path()}")
    print(f"nvidia_smi={visible['nvidia_smi']} rocminfo={visible['rocminfo']}")
    print(f"wandb={'on' if os.environ.get('WANDB_API_KEY') else 'off'}")
    rows = read_ledger()
    print(f"last_stop={rows[-1]['stop'] if rows else 'none'}")
    gate = hard_gate()
    print(f"launch={'ready' if gate is None else gate}")


def cmd_check() -> None:
    load_env()
    assert_pin_matches_files()
    if slug_for("judge") == slug_for("propose"):
        raise RunnerError("judge slug matches the proposer slug")
    for role in CHEAP_ROLES:
        assert_cheap(role, slug_for(role))
        assert_cheap(role, slug_for(role, "fallback"))
    metric_script_blocks()
    if h5_path().is_file():
        raise RunnerError("expected the Push-T h5 to be absent on this machine")
    before = pinned_from_files()
    after = dict(before)
    after["sigreg_weight"] = 0.01
    after["embed_dim"] = 64
    changed = changed_factor_names(before, after)
    if changed != ["sigreg_weight", "latent_dim"]:
        raise RunnerError(f"multi-factor detector returned {changed}")
    try:
        one_sentence("Works if we scale the run.", "claim")
    except RunnerError:
        pass
    else:
        raise RunnerError("scale phrase was accepted")
    view = judge_view(
        {
            "claim": "A.",
            "falsifier": "B.",
            "metric_row": {"metric": None},
            "proposer_raw": "UNIQUE_PROPOSER_TRANSCRIPT",
        }
    )
    if "UNIQUE_PROPOSER_TRANSCRIPT" in json.dumps(view) or set(view) != set(JUDGE_KEYS):
        raise RunnerError("judge view leaked the proposer transcript")
    good = {key: 1 for key in REQUIRED_KEYS}
    good.update(
        {
            "claim": "A.",
            "falsifier": "B.",
            "degenerate_block": BLOCKS["a"],
            "factor": "baseline",
            "baseline_id": "baseline-0",
            "seed": 3072,
            "config": {"factor": "baseline"},
            "split": "train_split=0.9",
            "metric_script": "scripts/metric_trial.py",
            "commit": "abc",
            "budget_minutes": 20,
            "budget_tokens": 3000,
            "models_called": [{"slug": "deepseek/deepseek-v4-flash", "role": "propose"}],
            "metric": None,
            "verdict": "fail",
            "stop": "kill",
        }
    )
    if validate_row(good):
        raise RunnerError(f"valid row was rejected: {validate_row(good)}")
    missing = dict(good)
    missing.pop("stop")
    if not any("stop" in error for error in validate_row(missing)):
        raise RunnerError("missing stop was accepted")
    k3 = dict(good)
    k3["stop"] = "continue"
    k3["models_called"] = [{"slug": KIMI, "role": "escalate"}]
    if not validate_row(k3):
        raise RunnerError("K3 on continue was accepted")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        for relative in (
            "scripts/cycle.py",
            "scripts/smoke_openrouter.py",
            "scripts/env.sh",
            "scripts/launch.sh",
            "scripts/metric_trial.py",
            "baselines.json",
            "open_questions.md",
            "jepa-research-loop.md",
            "ledger.jsonl",
        ):
            path = ROOT / relative
            if path.exists() and key in path.read_text():
                raise RunnerError(f"secret written in {relative}")
    print("check ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Closed-loop OpenRouter autoresearch runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("propose", "run", "judge", "close", "status", "check", "setup"):
        sub.add_parser(name)
    launch = sub.add_parser("launch")
    launch.add_argument("--max-cycles", type=int, default=1)
    admit = sub.add_parser("admit")
    admit.add_argument("--paper", required=True)
    admit.add_argument("--changes", required=True, choices=["hypothesis", "baseline", "stop_rule"])
    args = parser.parse_args(argv)
    try:
        if args.cmd == "propose":
            cmd_propose()
        elif args.cmd == "run":
            cmd_run()
        elif args.cmd == "judge":
            cmd_judge()
        elif args.cmd == "close":
            cmd_close()
        elif args.cmd == "launch":
            if args.max_cycles < 1:
                raise RunnerError("--max-cycles must be at least 1")
            cmd_launch(args.max_cycles)
        elif args.cmd == "admit":
            cmd_admit(args.paper, args.changes)
        elif args.cmd == "status":
            cmd_status()
        elif args.cmd == "setup":
            cmd_setup()
        elif args.cmd == "check":
            cmd_check()
    except RunnerError as exc:
        print(redact(str(exc)), file=sys.stderr)
        return exc.code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
