#!/usr/bin/env python3
"""Sentence -> robot command, for the language part of HANDOFF_5090.md.

A command is a dict:
  {"action": "target"}
  {"action": "turn", "deg": signed degrees, positive = clockwise on screen}
  {"action": "push", "dir": "left" | "right" | "up" | "down", "px": pixels}

Two parsers:
  llm      Qwen2.5-1.5B-Instruct (local, greedy), few-shot prompt below, JSON out.
  keyword  rules built only from the seven check-3 template sentences. Baseline.

The prompt and few-shot examples were written before the held-out paraphrases
existed (research/language/paraphrases.jsonl, written by Codex), and are frozen.

  python scripts/command_parser.py "give it a quarter spin to the right"
  python scripts/command_parser.py --eval research/language/paraphrases.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "hf_models" / "Qwen2.5-1.5B-Instruct"
OUT = ROOT / "research" / "artifacts" / "5090"

DIRS = ("left", "right", "up", "down")
ALIASES = {"west": "left", "east": "right", "north": "up", "south": "down", "top": "up", "bottom": "down"}
DEG_TOL = 15.0  # a turn parse is right if the sign matches and |deg - gold| <= this
PX_TOL = 0.25  # a push parse is right if the dir matches and |px - gold| <= max(20, this * gold)

SYSTEM = """You turn one instruction for a robot into a JSON command.

Setting: a top-down 2D table, 512 px square. A grey T-shaped block (about 100 px long) lies on it, \
a round pusher moves it, and a green T outline marks the target pose.

Output exactly one JSON object and nothing else, in one of these forms:
{"action": "target"}
{"action": "turn", "rotation": "clockwise|counterclockwise", "deg": D}
{"action": "push", "dir": "left|right|up|down", "px": P}

Rules:
- target: put the T on the green target outline (also "goal", "outline", "where it belongs").
- turn: rotate the T in place. rotation is as seen on screen; anticlockwise = counterclockwise. D is a positive \
number of degrees. "to the right" for a rotation means clockwise, "to the left" means counterclockwise.
- push: slide the T. up = toward the top of the screen. "away from you"/"north" = up, "toward you"/"south" = down, \
"west" = left, "east" = right.
- Amounts: no amount given -> turn 90, push 80. "a little", "slightly", "a bit", "nudge" -> turn 45, push 40. \
"a lot", "far" -> push 160. Quarter turn = 90, half turn = 180, eighth of a turn = 45. One T length = 100 px."""

FEW_SHOT = [
    ("Rotate the block 60 degrees anticlockwise.", {"action": "turn", "rotation": "counterclockwise", "deg": 60}),
    ("turn it 30 degrees clockwise", {"action": "turn", "rotation": "clockwise", "deg": 30}),
    ("could you slide it over to the right a little", {"action": "push", "dir": "right", "px": 40}),
    ("Line the T up with the green outline.", {"action": "target"}),
    ("spin it half way round", {"action": "turn", "rotation": "clockwise", "deg": 180}),
    ("move the piece two T lengths toward the top", {"action": "push", "dir": "up", "px": 200}),
    ("i want it pointing a quarter turn more to the right", {"action": "turn", "rotation": "clockwise", "deg": 90}),
    ("shove it down 120px", {"action": "push", "dir": "down", "px": 120}),
    ("move it 50 pixels east", {"action": "push", "dir": "right", "px": 50}),
    ("put the letter back where it belongs", {"action": "target"}),
]


def parse_json(text: str):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        return normalize(json.loads(m.group(0)))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def normalize(cmd):
    """Validate and canonicalise a command dict. Returns None if it is not a command."""
    if not isinstance(cmd, dict):
        return None
    action = cmd.get("action")
    if action == "target":
        return {"action": "target"}
    if action == "turn" and isinstance(cmd.get("deg"), (int, float)):
        rot = cmd.get("rotation")
        if rot is None:  # gold labels carry a signed deg
            return {"action": "turn", "deg": float(cmd["deg"])}
        if rot in ("clockwise", "counterclockwise"):
            return {"action": "turn", "deg": abs(float(cmd["deg"])) * (1.0 if rot == "clockwise" else -1.0)}
        return None
    if action == "push" and isinstance(cmd.get("dir"), str):
        cmd = dict(cmd, dir=ALIASES.get(cmd["dir"].lower(), cmd["dir"].lower()))
    if action == "push" and cmd.get("dir") in DIRS and isinstance(cmd.get("px"), (int, float)):
        return {"action": "push", "dir": cmd["dir"], "px": float(cmd["px"])}
    return None


def gold_command(row):
    return normalize({k: row.get(k) for k in ("action", "deg", "dir", "px")})


def correct(pred, gold) -> bool:
    if pred is None or gold is None or pred["action"] != gold["action"]:
        return False
    if gold["action"] == "turn":
        return pred["deg"] * gold["deg"] > 0 and abs(pred["deg"] - gold["deg"]) <= DEG_TOL
    if gold["action"] == "push":
        return pred["dir"] == gold["dir"] and abs(pred["px"] - gold["px"]) <= max(20.0, PX_TOL * gold["px"])
    return True


class KeywordParser:
    """Rules from the seven check-3 templates only: target / clockwise / counter-clockwise / left right up down."""

    def __call__(self, text: str):
        t = text.lower()
        num = re.search(r"(\d+(?:\.\d+)?)", t)
        if "counter" in t:
            return {"action": "turn", "deg": -float(num.group(1)) if num else -90.0}
        if "clockwise" in t:
            return {"action": "turn", "deg": float(num.group(1)) if num else 90.0}
        if "target" in t:
            return {"action": "target"}
        for d in DIRS:
            if re.search(rf"\b{d}\b", t):
                return {"action": "push", "dir": d, "px": float(num.group(1)) if num else 80.0}
        return None


class LLMParser:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()

    def messages(self, text: str):
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in FEW_SHOT:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
        return msgs + [{"role": "user", "content": text}]

    def raw(self, text: str) -> str:
        prompt = self.tok.apply_chat_template(self.messages(text), tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=48, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)

    def __call__(self, text: str):
        return parse_json(self.raw(text))


def evaluate(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    llm, kw = LLMParser(), KeywordParser()
    out, t0 = [], time.time()
    for r in rows:
        gold = gold_command(r)
        raw = llm.raw(r["text"])
        pred = parse_json(raw)
        kpred = kw(r["text"])
        out.append(
            {
                "id": r["id"],
                "text": r["text"],
                "style": r.get("style"),
                "gold": gold,
                "llm": pred,
                "llm_raw": raw,
                "llm_ok": correct(pred, gold),
                "keyword": kpred,
                "keyword_ok": correct(kpred, gold),
            }
        )
    secs = time.time() - t0

    def rate(sub, key):
        return f"{sum(x[key] for x in sub)}/{len(sub)}"

    summary = {
        "n": len(out),
        "llm_correct": sum(x["llm_ok"] for x in out),
        "llm_accuracy": sum(x["llm_ok"] for x in out) / len(out),
        "keyword_correct": sum(x["keyword_ok"] for x in out),
        "keyword_accuracy": sum(x["keyword_ok"] for x in out) / len(out),
        "invalid_gold": sum(x["gold"] is None for x in out),
        "llm_by_style": {s: rate([x for x in out if x["style"] == s], "llm_ok") for s in sorted({x["style"] for x in out})},
        "llm_by_action": {
            a: rate([x for x in out if x["gold"] and x["gold"]["action"] == a], "llm_ok") for a in ("target", "turn", "push")
        },
        "ms_per_sentence": 1000 * secs / len(out),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"parse_{path.stem}_n{len(out)}.json"
    payload = {"summary": summary, "model": MODEL_DIR.name, "deg_tol": DEG_TOL, "px_tol": PX_TOL, "rows": out}
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)
    print("WROTE", dest, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?")
    p.add_argument("--eval", type=Path, help="jsonl of {text, action, deg, dir, px, style}")
    args = p.parse_args()
    if args.eval:
        evaluate(args.eval)
    elif args.text:
        print(json.dumps(LLMParser()(args.text)))
    else:
        sys.exit("give a sentence or --eval FILE")


if __name__ == "__main__":
    main()
