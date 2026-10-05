#!/usr/bin/env python3
"""Call proposer, patch, and judge with max_tokens=16. Do not call GLM 5.2 or Kimi K3."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cycle


def main() -> int:
    try:
        cycle.load_env()
    except cycle.RunnerError as exc:
        print(cycle.redact(str(exc)), file=sys.stderr)
        return exc.code
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("OPENROUTER_API_KEY is empty", file=sys.stderr)
        return 2
    messages = [{"role": "user", "content": "Reply with the single word ok."}]
    state = {"models_called": [], "tokens_used": 0, "premium_cost": [], "proposer_slug": cycle.slug_for("propose")}
    for role in ("propose", "patch", "judge"):
        slug = cycle.slug_for(role)
        if slug in cycle.PREMIUM_SLUGS or cycle.slug_for(role, "fallback") in {cycle.KIMI, cycle.GLM}:
            print(f"refusing premium slug on {role}", file=sys.stderr)
            return 2
        try:
            _text, usage, used = cycle.call_role(state, role, messages, max_tokens=16)
        except cycle.RunnerError as exc:
            print(f"{role} status=error detail={cycle.redact(str(exc))}")
            return 1
        print(
            f"{role} slug={used} status=ok "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
