"""Remote GPU dispatch for the cycle runner.

Handles SSH connections to the remote GPU machine (n1.msk.cloreai.ru)
and runs experiments there when the local machine has no GPU.

Usage:
    from scripts.remote import remote_gpu_available, remote_run, remote_bootstrap
"""
from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".cycle" / "remote.json"
DEFAULT_CONFIG = {
    "host": "root@n1.msk.cloreai.ru",
    "port": "1837",
    "remote_dir": "/root/MMAGAIProjectOpen",
    "ssh_key": None,  # Uses default SSH agent
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(CONFIG_PATH.read_text())
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return merged


def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def ssh_cmd(config: dict | None = None) -> list[str]:
    cfg = config or load_config()
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    if cfg.get("port"):
        cmd += ["-p", str(cfg["port"])]
    cmd.append(cfg["host"])
    return cmd


def remote_gpu_available(config: dict | None = None) -> bool:
    """SSH to remote and check if nvidia-smi works."""
    try:
        result = subprocess.run(
            [*ssh_cmd(config), "nvidia-smi 2>/dev/null && echo GPU_OK || echo NO_GPU"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return "GPU_OK" in result.stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"  remote check failed: {exc}", file=sys.stderr)
        return False


def remote_check_h5(config: dict | None = None) -> bool:
    """Check if pusht_expert_train.h5 exists on remote."""
    try:
        cfg = config or load_config()
        remote_h5 = Path(cfg["remote_dir"]) / "data" / "pusht_expert_train.h5"
        result = subprocess.run(
            [*ssh_cmd(config), f"test -f {shlex.quote(str(remote_h5))} && echo H5_OK || echo NO_H5"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return "H5_OK" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def remote_env_setup(config: dict | None = None) -> str | None:
    """Check that remote has .venv with torch+cuda. Return None if OK, error string if not."""
    try:
        cfg = config or load_config()
        cmd = [
            *ssh_cmd(config),
            f"cd {shlex.quote(cfg['remote_dir'])} && "
            "test -f .venv/bin/python && "
            ".venv/bin/python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return ".venv missing or torch not installed"
        if "True" not in result.stdout:
            return "torch.cuda.is_available() is False on remote"
        return None
    except subprocess.TimeoutExpired:
        return "SSH timeout checking remote env"
    except OSError as exc:
        return f"SSH failed: {exc}"


def remote_run(
    command: str | list[str],
    config: dict | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess:
    """Run a command on the remote machine via SSH."""
    cfg = config or load_config()
    if isinstance(command, list):
        command = " ".join(shlex.quote(str(c)) for c in command)
    remote_cmd = f"cd {shlex.quote(cfg['remote_dir'])} && {command}"
    full_cmd = [*ssh_cmd(config), remote_cmd]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)


def remote_run_script(
    script_rel: str,
    args: list[str] | None = None,
    config: dict | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess:
    """Run a script from the project on the remote machine."""
    cfg = config or load_config()
    parts = [f".venv/bin/python -u {shlex.quote('scripts/' + script_rel)}"]
    if args:
        parts.extend(shlex.quote(str(a)) for a in args)
    return remote_run(" ".join(parts), config, timeout)


def remote_rsync_push(
    local_path: str | Path,
    remote_rel: str,
    config: dict | None = None,
) -> bool:
    """Rsync a local path to the remote machine."""
    cfg = config or load_config()
    remote_dest = f"{cfg['host']}:{cfg['remote_dir']}/{remote_rel}"
    port_flag = ["-e", f"ssh -p {cfg['port']}"] if cfg.get("port") else []
    cmd = [
        "rsync", "-a", "--partial", "--progress",
        *port_flag,
        str(local_path),
        remote_dest,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def remote_rsync_pull(
    remote_rel: str,
    local_path: str | Path,
    config: dict | None = None,
) -> bool:
    """Rsync from the remote machine to local."""
    cfg = config or load_config()
    remote_src = f"{cfg['host']}:{cfg['remote_dir']}/{remote_rel}"
    port_flag = ["-e", f"ssh -p {cfg['port']}"] if cfg.get("port") else []
    cmd = ["rsync", "-a", "--partial", *port_flag, remote_src, str(local_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def remote_bootstrap(config: dict | None = None) -> list[str]:
    """Bootstrap the remote environment step by step. Returns list of steps run."""
    cfg = config or load_config()
    steps = []
    log = []

    # Step 1: Check connectivity
    log.append("Checking SSH connectivity...")
    try:
        r = subprocess.run(
            [*ssh_cmd(config), "echo OK"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return ["SSH FAILED - check host/port and key"]
        log.append("  SSH OK")
    except Exception as exc:
        return [f"SSH FAILED: {exc}"]

    # Step 2: Create .venv with uv
    log.append("Creating .venv with uv...")
    cmds = [
        f"cd {shlex.quote(cfg['remote_dir'])}",
        "export UV_PYTHON_INSTALL_DIR=\"$PWD/.tools/python\"",
        "export UV_CACHE_DIR=\"$PWD/.tools/uv-cache\"",
        # Fresh venv
        "rm -rf .venv 2>/dev/null; true",
        "./.tools/uv venv --python 3.10 .venv",
    ]
    r = subprocess.run(
        [*ssh_cmd(config), " && ".join(cmds)],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        log.append(f"  VENV FAILED: {r.stderr[-200:]}")
        return log
    log.append("  venv created")

    # Step 3: Install CUDA torch
    log.append("Installing CUDA torch...")
    cmds = [
        f"cd {shlex.quote(cfg['remote_dir'])}",
        "export UV_PYTHON_INSTALL_DIR=\"$PWD/.tools/python\"",
        "export UV_CACHE_DIR=\"$PWD/.tools/uv-cache\"",
        "./.tools/uv pip install --python .venv/bin/python "
        "torch torchvision --index-url https://download.pytorch.org/whl/cu124",
    ]
    r = subprocess.run(
        [*ssh_cmd(config), " && ".join(cmds)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        log.append(f"  TORCH FAILED: {r.stderr[-200:]}")
        return log
    log.append("  CUDA torch installed")

    # Step 4: Verify CUDA
    log.append("Verifying CUDA...")
    r = subprocess.run(
        [*ssh_cmd(config),
         f"cd {shlex.quote(cfg['remote_dir'])} && "
         ".venv/bin/python -c 'import torch; print(f\"CUDA={torch.cuda.is_available()}, count={torch.cuda.device_count()}\")'"],
        capture_output=True, text=True, timeout=30,
    )
    log.append(f"  {r.stdout.strip()}")

    # Step 5: Install project deps
    log.append("Installing project deps...")
    cmds = [
        f"cd {shlex.quote(cfg['remote_dir'])}",
        "export UV_PYTHON_INSTALL_DIR=\"$PWD/.tools/python\"",
        "export UV_CACHE_DIR=\"$PWD/.tools/uv-cache\"",
        "./.tools/uv pip install --python .venv/bin/python "
        "'stable-worldmodel[train]==0.1.1' "
        "pygame pymunk shapely opencv-python-headless gymnasium imageio scikit-learn matplotlib "
        "'transformers>=4.50,<5'",
    ]
    r = subprocess.run(
        [*ssh_cmd(config), " && ".join(cmds)],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        log.append(f"  DEPS FAILED: {r.stderr[-200:]}")
        return log
    log.append("  deps installed")

    # Step 6: Verify full env
    log.append("Verifying full env...")
    r = subprocess.run(
        [*ssh_cmd(config),
         f"cd {shlex.quote(cfg['remote_dir'])} && "
         ".venv/bin/python -c 'import torch, stable_worldmodel; print(f\"torch={torch.__version__} cuda={torch.cuda.is_available()}\")'"],
        capture_output=True, text=True, timeout=30,
    )
    log.append(f"  {r.stdout.strip()}")

    return log


def save_default_config() -> None:
    """Write the default remote config file."""
    save_config(DEFAULT_CONFIG)
    print(f"Saved remote config to {CONFIG_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remote GPU dispatch for cycle runner")
    parser.add_argument("cmd", choices=["check", "bootstrap", "config"])
    args = parser.parse_args()

    if args.cmd == "check":
        cfg = load_config()
        print(f"Config: host={cfg['host']} port={cfg['port']} dir={cfg['remote_dir']}")
        print(f"GPU available: {remote_gpu_available()}")
        print(f"H5 present: {remote_check_h5()}")
        print(f"Env: {remote_env_setup() or 'OK'}")

    elif args.cmd == "bootstrap":
        print("Bootstrapping remote environment...")
        for step in remote_bootstrap():
            print(step)

    elif args.cmd == "config":
        save_default_config()