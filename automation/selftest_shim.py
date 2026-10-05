#!/usr/bin/env python3
"""selftest_shim.py — wraps env_common's discover_capabilities as an experiment.

Runs on the remote 4090. Discovers GPU, dataset, model, and writes capabilities.json.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation"))

from env_common import discover_capabilities, experiment_main
import json


def run(out_dir: str, args: dict) -> dict:
    caps = discover_capabilities()
    # Write to out_dir as well for the runner
    with open(os.path.join(out_dir, "capabilities.json"), "w") as f:
        json.dump(caps, f, indent=2)
    return caps


if __name__ == "__main__":
    experiment_main(run)