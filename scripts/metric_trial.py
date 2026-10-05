#!/usr/bin/env python3
"""Score one trial row. This script does not invent measurements."""

from __future__ import annotations

import json
import sys

# The runner refuses a trial if these names are removed.
LATENT_COLLAPSE_STD = 1e-3
REQUIRE_CEM_SUCCESS = True

PLANNING_FACTORS = {"latent_dim", "cem_population", "cem_iterations"}


def score(row: dict) -> dict:
    """Return a metric row. A missing observation voids the number."""
    factor = row.get("factor")
    observed = row.get("observed") or {}
    if not isinstance(observed, dict):
        return {"metric": None, "void_reason": "observed is not an object", "degenerate_blocked": False}

    if factor in {"sigreg_weight", "baseline"}:
        if "latent_std" not in observed:
            return {
                "metric": None,
                "void_reason": "latent_std missing; collapse check cannot run",
                "degenerate_blocked": True,
            }
        std = observed["latent_std"]
        if not isinstance(std, (int, float)) or isinstance(std, bool):
            return {"metric": None, "void_reason": "latent_std is not a number", "degenerate_blocked": True}
        if float(std) < LATENT_COLLAPSE_STD:
            return {
                "metric": None,
                "void_reason": "degenerate latent collapse to a constant",
                "degenerate_blocked": True,
            }
        if factor == "baseline":
            return {"metric": float(std), "void_reason": None, "degenerate_blocked": False}
        if "value" not in observed:
            return {"metric": None, "void_reason": "no measured value beside latent_std", "degenerate_blocked": False}
        return {"metric": observed["value"], "void_reason": None, "degenerate_blocked": False}

    if factor in PLANNING_FACTORS:
        if not REQUIRE_CEM_SUCCESS or "cem_success" not in observed:
            return {
                "metric": None,
                "void_reason": "metric ignores planning; cem_success missing",
                "degenerate_blocked": True,
            }
        success = observed["cem_success"]
        if not isinstance(success, (int, float)) or isinstance(success, bool):
            return {"metric": None, "void_reason": "cem_success is not a number", "degenerate_blocked": True}
        return {"metric": success, "void_reason": None, "degenerate_blocked": False}

    return {"metric": None, "void_reason": f"no scorer for factor {factor}", "degenerate_blocked": False}


def main() -> None:
    row = json.load(sys.stdin)
    json.dump(score(row), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
