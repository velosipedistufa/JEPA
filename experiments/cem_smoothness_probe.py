import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE
import torch
import math
import warnings

warnings.filterwarnings("ignore")


def _load_dataset(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _safe_float(value, default=0.0):
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            value = value.detach().mean().item()
        else:
            value = float(value)

        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _extract_actions(stats):
    if isinstance(stats, dict):
        for key in ("actions", "action_sequence", "action", "acts", "action_sequences"):
            if key in stats and stats[key] is not None:
                return stats[key]

        for key in ("stats", "data"):
            if key in stats and isinstance(stats[key], dict):
                nested = _extract_actions(stats[key])
                if nested is not None:
                    return nested

        return None

    return stats


@experiment_main
def run(out_dir: str, args: dict) -> dict:
    if not isinstance(args, dict):
        args = {}

    dataset_path = args.get("dataset_path", "lance_data/action_stats.pt")
    dataset_path = str(dataset_path or "lance_data/action_stats.pt")

    if not os.path.exists(dataset_path):
        return {
            "ok": False,
            "error": f"Dataset not found at {dataset_path}",
            "verdict": "failed",
        }

    try:
        stats = _load_dataset(dataset_path)
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to load dataset: {e}",
            "verdict": "failed",
        }

    actions = _extract_actions(stats)

    if actions is None:
        return {
            "ok": False,
            "error": "No action tensor found in dataset",
            "verdict": "failed",
        }

    if isinstance(actions, (list, tuple)):
        if len(actions) == 0:
            return {
                "ok": False,
                "error": "Action sequence is empty",
                "verdict": "failed",
            }

        try:
            actions = torch.as_tensor(actions)
        except Exception:
            try:
                actions = torch.stack(actions)
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Failed to convert action list to tensor: {e}",
                    "verdict": "failed",
                }

    if not torch.is_tensor(actions):
        try:
            actions = torch.as_tensor(actions)
        except Exception:
            try:
                actions = torch.tensor(actions)
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Actions could not be converted to tensor: {e}",
                    "verdict": "failed",
                }

    try:
        actions = actions.detach().float().to(DEVICE)
    except Exception:
        try:
            actions = actions.float().to(DEVICE)
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to prepare action tensor: {e}",
                "verdict": "failed",
            }

    try:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        actions = torch.where(torch.isfinite(actions), actions, torch.zeros_like(actions))

    if actions.dim() == 0:
        actions = actions.view(1, 1)
    elif actions.dim() == 1:
        actions = actions.unsqueeze(-1)
    elif actions.dim() > 2:
        if actions.numel() == 0:
            actions = actions.reshape(actions.shape[0], 0)
        else:
            actions = actions.reshape(actions.shape[0], -1)

    T = int(actions.shape[0])

    if T < 3 or actions.numel() == 0 or actions.shape[-1] == 0:
        return {
            "ok": True,
            "T": T,
            "mean_velocity_norm": 0.0,
            "mean_acceleration_norm": 0.0,
            "mean_jerk_norm": 0.0,
            "velocity_autocorr_lag1": 0.0,
            "device": str(DEVICE),
            "verdict": "insufficient_data",
        }

    diffs = actions[1:] - actions[:-1]
    accel = diffs[1:] - diffs[:-1]
    jerk = accel[1:] - accel[:-1]

    if diffs.dim() > 1:
        vel_norms = torch.norm(diffs, dim=1)
    else:
        vel_norms = torch.abs(diffs)

    if accel.dim() > 1:
        acc_norms = torch.norm(accel, dim=1)
    else:
        acc_norms = torch.abs(accel)

    if jerk.dim() > 1:
        jerk_norms = torch.norm(jerk, dim=1)
    else:
        jerk_norms = torch.abs(jerk)

    smoothness_vel = _safe_float(vel_norms, 0.0)
    smoothness_acc = _safe_float(acc_norms, 0.0)
    smoothness_jerk = _safe_float(jerk_norms, 0.0)

    if diffs.shape[0] >= 2 and diffs.numel() > 0:
        autocorr_lag1 = _safe_float(torch.mean(diffs[:-1] * diffs[1:]), 0.0)
    else:
        autocorr_lag1 = 0.0

    return {
        "ok": True,
        "T": T,
        "mean_velocity_norm": smoothness_vel,
        "mean_acceleration_norm": smoothness_acc,
        "mean_jerk_norm": smoothness_jerk,
        "velocity_autocorr_lag1": autocorr_lag1,
        "device": str(DEVICE),
        "verdict": "success",
    }


if __name__ == "__main__":
    pass