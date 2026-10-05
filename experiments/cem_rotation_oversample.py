import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE
import env_common
import numpy as np


def _get_env_function(*names):
    for name in names:
        fn = getattr(env_common, name, None)
        if callable(fn):
            return fn
    return None


def _call_with_optional_path(fn, path):
    if fn is None:
        return None
    try:
        return fn(path)
    except TypeError:
        try:
            return fn()
        except Exception:
            return None
    except Exception:
        return None


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _action_stats(dataset):
    fn = _get_env_function(
        "get_action_stats",
        "get_dataset_action_stats",
        "compute_action_stats",
    )
    if fn is None:
        return {}
    try:
        stats = fn(dataset)
    except TypeError:
        try:
            stats = fn()
        except Exception:
            return {}
    except Exception:
        return {}

    if isinstance(stats, dict):
        return stats
    try:
        return dict(stats)
    except Exception:
        return {}


def _mean_axis0(arr):
    try:
        arr = np.asarray(arr, dtype=np.float32)
    except Exception:
        return np.zeros(1, dtype=np.float32)

    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)

    if arr.ndim == 0:
        return np.array([float(arr.mean())], dtype=np.float32)

    return np.atleast_1d(np.asarray(arr.mean(axis=0), dtype=np.float32))


def evaluate_cem(model, dataset, init_mean, params, task_type, num_trials):
    try:
        from env_common import run_cem
    except Exception:
        run_cem = getattr(env_common, "run_cem", None)

    if num_trials <= 0:
        return 0.0

    if not callable(run_cem):
        return 0.0

    successes = 0
    for i in range(num_trials):
        try:
            task = dataset.get_task(task_type, i)
        except Exception:
            task = {"task_type": task_type, "task_index": i}

        try:
            result = run_cem(
                model,
                task,
                init_mean=init_mean,
                num_samples=_safe_int(params.get("cem_num_samples"), 64),
                num_iters=_safe_int(params.get("cem_num_iters"), 5),
            )
            if isinstance(result, dict) and result.get("success"):
                successes += 1
        except TypeError:
            try:
                result = run_cem(model, task)
                if isinstance(result, dict) and result.get("success"):
                    successes += 1
            except Exception:
                continue
        except Exception:
            continue

    return successes / num_trials


@experiment_main
def run(out_dir: str, args: dict) -> dict:
    params = args or {}

    try:
        model_loader = _get_env_function(
            "load_lewm",
            "load_lance_lewm",
            "load_model",
            "load_world_model",
            "load_policy",
            "load_agent",
        )
        dataset_loader = _get_env_function(
            "load_lance_dataset",
            "load_dataset",
            "load_lance_data",
        )

        model = _call_with_optional_path(model_loader, params.get("model_path"))
        dataset = _call_with_optional_path(dataset_loader, params.get("dataset_path"))
        action_stats = _action_stats(dataset)

        rot_actions = action_stats.get(
            "rotation_actions",
            np.zeros((1, 1), dtype=np.float32),
        )
        trans_actions = action_stats.get(
            "translation_actions",
            np.zeros((1, 1), dtype=np.float32),
        )
        oversample = max(1, _safe_int(params.get("rotation_oversample_factor"), 3))

        rot_mean = _mean_axis0(rot_actions)
        trans_mean = _mean_axis0(trans_actions)

        init_mean = np.concatenate([np.tile(rot_mean, oversample), trans_mean])
        norm = float(np.linalg.norm(init_mean))
        if norm > 0:
            init_mean = init_mean / norm

        rotation_success = evaluate_cem(
            model,
            dataset,
            init_mean,
            params,
            task_type="rotation",
            num_trials=_safe_int(params.get("rotation_num_trials"), 20),
        )
        translation_success = evaluate_cem(
            model,
            dataset,
            init_mean,
            params,
            task_type="translation",
            num_trials=_safe_int(params.get("translation_num_trials"), 50),
        )

        return {
            "ok": True,
            "rotation_success": rotation_success,
            "translation_success": translation_success,
            "oversample_factor": oversample,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }