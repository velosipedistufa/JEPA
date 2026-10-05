#!/usr/bin/env python3
"""remote_runner.py — runs experiment jobs on the 4090.

Receives a job JSON path, sets up the environment, runs the target script,
captures stdout/stderr, and writes status.json + metrics.json.

Designed by GLM 5.3 Prime.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = os.environ.get("MMAGAI_ROOT", "/root/MMAGAIProjectOpen")

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLEWM_HOME", os.path.join(PROJECT_ROOT, "data"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def write_status(run_dir: str, state: str, **extra) -> None:
    status = {"state": state, "ts": time.time(), **extra}
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(run_dir, "status.json"), "w") as f:
        json.dump(status, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="Path to job.json")
    args = ap.parse_args()

    job = json.loads(Path(args.job).read_text())
    run_dir = job.get("run_dir", os.path.dirname(args.job))
    script = job["script"]
    argv = job.get("args", [])
    timeout_s = job.get("timeout_s", 1800)

    write_status(run_dir, "starting")

    # Resolve script path
    script_path = os.path.join(PROJECT_ROOT, script)
    if not os.path.exists(script_path):
        write_status(run_dir, "error", error=f"Script not found: {script_path}")
        sys.exit(1)

    # Build command
    cmd = [sys.executable, "-u", script_path] + [str(a) for a in argv]
    env = os.environ.copy()
    env["LEWM_JOB_DIR"] = run_dir
    env["LEWM_JOB_ARGS"] = json.dumps(job.get("params", {}))

    write_status(run_dir, "running")
    start = time.time()

    with open(os.path.join(run_dir, "stdout.log"), "w") as out_f:
        with open(os.path.join(run_dir, "stderr.log"), "w") as err_f:
            proc = subprocess.Popen(
                cmd,
                stdout=out_f,
                stderr=err_f,
                env=env,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                rc = -9
                write_status(run_dir, "timeout", returncode=rc,
                             elapsed_s=round(time.time() - start, 1))
                sys.exit(2)

    elapsed = time.time() - start
    state = "done" if rc == 0 else "error"
    write_status(run_dir, state, returncode=rc, elapsed_s=round(elapsed, 1))

    # Collect metrics.json if written by experiment
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            metrics = json.loads(Path(metrics_path).read_text())
            write_status(run_dir, state, returncode=rc,
                         elapsed_s=round(elapsed, 1), **metrics)
        except Exception:
            pass

    sys.exit(0 if rc == 0 else 3)


if __name__ == "__main__":
    main()