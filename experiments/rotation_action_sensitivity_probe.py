import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE
import numpy as np
import importlib

def _get_helper(name):
    import env_common
    return getattr(env_common, name, None)

@experiment_main
def run(out_dir: str, args: dict) -> dict:
    try:
        load_model = _get_helper('load_model')
        load_dataset = _get_helper('load_dataset')
        get_action_stats = _get_helper('get_action_stats')
        encode_obs = _get_helper('encode_obs')
        rollout_latent = _get_helper('rollout_latent')

        helpers = {
            "load_model": load_model,
            "load_dataset": load_dataset,
            "get_action_stats": get_action_stats,
            "encode_obs": encode_obs,
            "rollout_latent": rollout_latent,
        }

        missing = [name for name, helper in helpers.items() if helper is None]
        if missing:
            return {"ok": False, "error": f"env_common missing helpers: {missing}"}

        return {
            "ok": True,
            "device": str(DEVICE),
            "helpers": sorted(helpers.keys()),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}