import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE
import env_common

def _get_param(name, default=None):
    return default

def _get_optional(name, default=None):
    return getattr(env_common, name, default)

@experiment_main
def run(out_dir: str, args: dict) -> dict:
    try:
        load_env = _get_optional("load_env")
        if load_env is None:
            for candidate in ("load_environment", "get_env", "make_env", "create_env"):
                load_env = _get_optional(candidate)
                if callable(load_env):
                    break

        if not callable(load_env):
            return {"ok": False, "error": "env_common does not provide an environment loader"}

        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}