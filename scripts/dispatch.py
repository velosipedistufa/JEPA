"""SSH dispatch layer for the MMAGAIProject autonomous loop.

Design by GLM 5.3 Prime. All GPU work dispatches to the RTX 4090.
Local Xeon NEVER executes CUDA code.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import shlex

REMOTE_HOST = os.environ.get("MMA_REMOTE", "root@n1.msk.cloreai.ru")
REMOTE_PORT = int(os.environ.get("MMA_REMOTE_PORT", "1837"))
REMOTE_DIR = os.environ.get("MMA_REMOTE_DIR", "/root/MMAGAIProjectOpen")
RUNS_DIR = f"{REMOTE_DIR}/runs"
AUTOMATION_DIR = f"{REMOTE_DIR}/automation"
POLL_S = 20

_SSH_BASE = ["ssh", "-p", str(REMOTE_PORT),
             "-o", "ConnectTimeout=20",
             "-o", "ServerAliveInterval=15",
             "-o", "ServerAliveCountMax=3",
             "-o", "StrictHostKeyChecking=accept-new"]


def _ssh(cmd: str, timeout: int = 120, input_text: str | None = None) -> tuple[int, str, str]:
    import time as _time
    for attempt in range(3):
        try:
            p = subprocess.run(
                _SSH_BASE + [REMOTE_HOST, cmd],
                capture_output=True, text=True, timeout=timeout,
                input=input_text,
            )
            return p.returncode, p.stdout.strip(), p.stderr.strip()
        except subprocess.TimeoutExpired:
            if attempt < 2:
                _time.sleep(5)
                continue
            raise
        except Exception:
            if attempt < 2:
                _time.sleep(10)
                continue
            raise


def _scp(src: str, dst: str, timeout: int = 300) -> tuple[int, str]:
    p = subprocess.run(
        ["scp", "-P", str(REMOTE_PORT), "-o", "ConnectTimeout=15",
         "-o", "StrictHostKeyChecking=accept-new", src, f"{REMOTE_HOST}:{dst}"],
        capture_output=True, text=True, timeout=timeout,
    )
    return p.returncode, p.stderr.strip()


def health() -> dict:
    """Verify SSH + GPU. Returns dict with ok=True only if the 4090 answers."""
    rc, out, err = _ssh(
        f"nvidia-smi --query-gpu=name,memory.used,memory.total "
        f"--format=csv,noheader 2>/dev/null && "
        f"test -f {shlex.quote(REMOTE_DIR)}/scripts/eval_official_cem.py "
        f"&& echo HEALTH_OK",
        timeout=60
    )
    gpu = {}
    if out and "HEALTH_OK" in out:
        lines = out.splitlines()
        if lines:
            parts = [x.strip() for x in lines[0].split(",")]
            if len(parts) == 3:
                gpu = {"name": parts[0], "mem_used_mb": parts[1], "mem_total_mb": parts[2]}
    ok = "HEALTH_OK" in out and bool(gpu)
    return {"ok": ok, "gpu": gpu, "raw": out, "err": err}


def gpu_free_mb() -> int:
    """Return estimated free GPU memory in MB."""
    rc, out, _ = _ssh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null")
    try:
        return int(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def push_file(local_path: str, remote_rel: str) -> dict:
    """Upload a file to REMOTE_DIR/remote_rel. Returns {ok, err, remote}."""
    dst = f"{REMOTE_DIR}/{remote_rel}"
    _ssh(f"mkdir -p $(dirname {shlex.quote(dst)})", timeout=30)
    rc, err = _scp(local_path, dst)
    return {"ok": rc == 0, "err": err, "remote": dst}


def push_dir(local_dir: str, remote_rel: str) -> dict:
    """Rsync a local directory to REMOTE_DIR/remote_rel."""
    dst = f"{REMOTE_HOST}:{REMOTE_DIR}/{remote_rel}"
    p = subprocess.run(
        ["rsync", "-a", "--partial", "--progress",
         "-e", f"ssh -p {REMOTE_PORT} -o StrictHostKeyChecking=accept-new",
         local_dir + "/", dst],
        capture_output=True, text=True, timeout=300,
    )
    return {"ok": p.returncode == 0, "err": p.stderr[-200:]}


def pull_file(remote_rel: str, local_path: str) -> dict:
    """Download REMOTE_DIR/remote_rel to local_path."""
    src = f"{REMOTE_HOST}:{REMOTE_DIR}/{remote_rel}"
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    rc, err = _scp(src, local_path)
    return {"ok": rc == 0, "err": err}


def run_blocking(script_rel: str, args: list[str] | None = None,
                 timeout: int = 1800, cwd: str | None = None) -> dict:
    """Run a project script on the remote GPU, blocking until completion.

    Creates a per-run directory, sets LEWM_JOB_DIR so `experiment_main` writes
    metrics.json there, then reads it back. Returns structured dict with
    ok, returncode, stdout, stderr, result (parsed metrics.json), elapsed_s.
    """
    venv_python = f"{REMOTE_DIR}/.venv/bin/python"
    script_path = f"{REMOTE_DIR}/{script_rel}"
    cmd_parts = [venv_python, "-u", script_path]
    if args:
        cmd_parts.extend(shlex.quote(a) for a in args)

    run_id = f"run_{int(time.time())}"
    run_dir = f"{RUNS_DIR}/{run_id}"
    rc, _, _ = _ssh(f"mkdir -p {shlex.quote(run_dir)}", timeout=30)

    cwd_cmd = f"cd {shlex.quote(cwd or REMOTE_DIR)} &&"
    env_prefix = f"LEWM_JOB_DIR={shlex.quote(run_dir)} LEWM_JOB_ARGS='{{}}'"
    full_cmd = f"{cwd_cmd} {env_prefix} {' '.join(cmd_parts)}"

    start = time.time()
    rc, out, err = _ssh(full_cmd, timeout=timeout + 30)
    elapsed = time.time() - start

    # Collect metrics.json written by the experiment harness.
    result = None
    _, metrics_raw, _ = _ssh(
        f"cat {shlex.quote(run_dir)}/metrics.json 2>/dev/null", timeout=30)
    if metrics_raw:
        try:
            result = json.loads(metrics_raw)
        except json.JSONDecodeError:
            pass

    return {
        "ok": rc == 0,
        "returncode": rc,
        "stdout": out,
        "stderr": err,
        "result": result,
        "run_id": run_id,
        "elapsed_s": round(elapsed, 1),
    }


def run_script_with_log(script_rel: str, args: list[str] | None = None,
                        timeout: int = 1800, run_id: str | None = None) -> dict:
    """Run a script and capture output to a run directory. Returns result dict."""
    if run_id is None:
        run_id = f"run_{int(time.time())}"
    run_dir = f"{RUNS_DIR}/{run_id}"
    
    # Create run dir
    _ssh(f"mkdir -p {shlex.quote(run_dir)}", timeout=30)
    
    # Write job spec
    job = {
        "run_id": run_id,
        "run_dir": run_dir,
        "script": script_rel,
        "args": args or [],
        "timeout_s": timeout,
    }
    job_json = json.dumps(job)
    
    # Push job spec and run
    rc, out, err = _ssh(
        f"cat > {shlex.quote(run_dir)}/job.json << 'JOBEOF'\n{job_json}\nJOBEOF\n"
        f"cd {shlex.quote(REMOTE_DIR)} && "
        f"nohup .venv/bin/python -u {shlex.quote(script_rel)} {' '.join(shlex.quote(a) for a in (args or []))} "
        f"> {shlex.quote(run_dir)}/stdout.log 2> {shlex.quote(run_dir)}/stderr.log & echo LAUNCHED",
        timeout=30,
    )
    
    if "LAUNCHED" not in out:
        return {"ok": False, "error": f"launch failed: {err}", "run_id": run_id}
    
    # Wait for completion
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(POLL_S)
        rc, out, _ = _ssh(
            f"ps aux | grep -v grep | grep {shlex.quote(script_rel)} >/dev/null "
            f"&& echo RUNNING || echo DONE",
            timeout=30,
        )
        if "DONE" in out:
            break
    
    # Collect results
    rc_stdout, stdout, _ = _ssh(f"cat {shlex.quote(run_dir)}/stdout.log 2>/dev/null", timeout=30)
    rc_stderr, stderr, _ = _ssh(f"cat {shlex.quote(run_dir)}/stderr.log 2>/dev/null || echo '(no stderr)'", timeout=30)
    
    # Try to parse result from last line of stdout
    result = None
    if stdout:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("RESULT_JSON"):
                try:
                    result = json.loads(line[len("RESULT_JSON"):])
                except json.JSONDecodeError:
                    pass
                break
    
    return {
        "ok": result is not None if result else True,
        "run_id": run_id,
        "stdout": stdout,
        "stderr": stderr,
        "result": result,
    }


def setup_automation_dir() -> dict:
    """Create automation/ directory on remote and push helper scripts."""
    _ssh(f"mkdir -p {shlex.quote(AUTOMATION_DIR)} {shlex.quote(RUNS_DIR)}", timeout=30)
    return {"ok": True}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remote GPU dispatch")
    parser.add_argument("cmd", choices=["health", "run", "push", "pull", "setup"])
    parser.add_argument("--script", help="script relative path")
    parser.add_argument("--args", nargs="*", help="script arguments")
    parser.add_argument("--local", help="local file path")
    parser.add_argument("--remote", help="remote relative path")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    
    if args.cmd == "health":
        print(json.dumps(health(), indent=2))
    elif args.cmd == "setup":
        print(json.dumps(setup_automation_dir(), indent=2))
    elif args.cmd == "push":
        if not args.local or not args.remote:
            print("--local and --remote required")
            sys.exit(1)
        print(json.dumps(push_file(args.local, args.remote), indent=2))
    elif args.cmd == "run":
        if not args.script:
            print("--script required")
            sys.exit(1)
        result = run_blocking(args.script, args.args, timeout=args.timeout)
        print(json.dumps(result, indent=2))
    elif args.cmd == "pull":
        if not args.remote or not args.local:
            print("--remote and --local required")
            sys.exit(1)
        print(json.dumps(pull_file(args.remote, args.local), indent=2))