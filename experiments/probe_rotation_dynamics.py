import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE

import json
import math


def _coerce_dict(args):
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return {}


def _coerce_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value, default):
    try:
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _impl(out_dir: str, args: dict) -> dict:
    cfg = _coerce_dict(args)

    if not out_dir:
        out_dir = "outputs"

    steps = _coerce_int(cfg.get("steps", cfg.get("n_steps", 16)), 16)
    steps = max(1, min(steps, 10000))

    angle = _coerce_float(cfg.get("angle", cfg.get("theta", 0.1)), 0.1)

    result = {
        "ok": True,
        "device": str(DEVICE),
        "steps": steps,
        "angle": angle,
    }

    try:
        two_pi = 2.0 * math.pi
        rotation = 0.0
        first_cos = None
        last_cos = None

        for _ in range(steps):
            rotation = (rotation + angle) % two_pi
            value = math.cos(rotation)

            if first_cos is None:
                first_cos = value
            last_cos = value

        result.update(
            {
                "final_rotation": rotation,
                "first_cos": first_cos,
                "last_cos": last_cos,
            }
        )

        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

    except Exception as exc:
        result["ok"] = False
        result["error"] = repr(exc)

    return result


@experiment_main
def run(out_dir: str, args: dict) -> dict:
    return _impl(out_dir, args)


def main(params=None) -> dict:
    cfg = _coerce_dict(params)
    out_dir = cfg.get("out_dir", cfg.get("output_dir", "outputs"))
    args = cfg.get("args", cfg)
    return _impl(out_dir, args)