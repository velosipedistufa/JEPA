import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE
import torch
import numpy as np


def _coerce_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _as_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_finite_float(value, default):
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def run(out_dir: str, args: dict) -> dict:
    try:
        args = _coerce_dict(args)

        raw_params = args.get("params")
        if isinstance(raw_params, dict):
            params = raw_params
        elif hasattr(raw_params, "__dict__"):
            params = vars(raw_params)
        else:
            params = args

        n_samples = max(1, _as_int(params.get("n_samples", 64), 64))
        n_iters = max(0, _as_int(params.get("n_iters", 3), 3))

        elite_frac = _as_finite_float(params.get("elite_frac", 0.2), 0.2)
        elite_frac = min(1.0, max(0.0, elite_frac))

        rot_scale = _as_finite_float(params.get("rot_scale", 0.1), 0.1)
        delta_z_ratio = _as_finite_float(params.get("delta_z_ratio", 0.991), 0.991)
        action_dim = max(1, _as_int(params.get("action_dim", 7), 7))
        seed = _as_int(params.get("seed", 0), 0)

        safe_params = {
            "n_samples": n_samples,
            "n_iters": n_iters,
            "elite_frac": elite_frac,
            "rot_scale": rot_scale,
            "delta_z_ratio": delta_z_ratio,
            "action_dim": action_dim,
            "seed": seed,
        }
    except Exception as exc:
        return {"ok": False, "error": f"invalid parameters: {exc}"}

    try:
        device = DEVICE
        try:
            device = torch.device(device if device is not None else "cpu")
            torch.zeros(1, device=device)
        except Exception:
            device = torch.device("cpu")

        safe_seed = int(seed) % (2**32)
        torch.manual_seed(safe_seed)
        try:
            np.random.seed(safe_seed)
        except Exception:
            np.random.seed(0)

        mean = torch.zeros(action_dim, device=device)
        std = torch.ones(action_dim, device=device)

        if action_dim > 3:
            std[3:min(6, action_dim)] = rot_scale

        best_cost = float("inf")
        best_action = None
        history = []

        for _ in range(n_iters):
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                n_samples, action_dim, device=device
            )

            if action_dim > 5:
                samples[:, 5] = samples[:, 4] * delta_z_ratio

            pos_end = min(3, action_dim)
            rot_end = min(6, action_dim)

            cost = torch.sum(samples[:, :pos_end] ** 2, dim=1)
            if rot_end > 3:
                cost = cost + rot_scale * torch.sum(samples[:, 3:rot_end] ** 2, dim=1)

            if hasattr(torch, "isfinite"):
                cost = torch.where(
                    torch.isfinite(cost),
                    cost,
                    torch.full_like(cost, float("inf")),
                )

            vals, idx = torch.sort(cost)
            if vals.numel() == 0:
                continue

            n_elite = max(1, min(n_samples, int(n_samples * elite_frac)))
            elite = samples[idx[:n_elite]]

            mean = elite.mean(dim=0)

            if elite.shape[0] > 1:
                std = ((elite - mean) ** 2).mean(dim=0).sqrt() + 1e-4
            else:
                std = torch.full_like(mean, 1e-4)

            current_best = float(vals[0].item())
            if not np.isfinite(current_best):
                current_best = float("inf")

            if current_best < best_cost:
                best_cost = current_best
                best_action = samples[idx[0]].detach().cpu().tolist()

            history.append(current_best if np.isfinite(current_best) else 0.0)

        if best_action is None:
            best_action = mean.detach().cpu().tolist()

        if not np.isfinite(best_cost):
            pos_end = min(3, action_dim)
            rot_end = min(6, action_dim)

            fallback_cost = torch.sum(mean[:pos_end] ** 2)
            if rot_end > 3:
                fallback_cost = fallback_cost + rot_scale * torch.sum(mean[3:rot_end] ** 2)

            fallback_value = float(fallback_cost.item())
            best_cost = fallback_value if np.isfinite(fallback_value) else 0.0

        best_action = [float(x) if np.isfinite(x) else 0.0 for x in best_action]

        return {
            "ok": True,
            "best_cost": float(best_cost),
            "best_action": best_action,
            "cost_history": history,
            "device": str(device),
            "params": safe_params,
            "out_dir": str(out_dir),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@experiment_main
def _main(out_dir: str = ".", args: dict = None, **kwargs) -> dict:
    if isinstance(out_dir, dict):
        if isinstance(args, str):
            out_dir, args = args, out_dir
        else:
            args = out_dir
            out_dir = "."
    elif hasattr(out_dir, "__dict__") and not isinstance(out_dir, str):
        args = out_dir
        out_dir = "."

    if args is None:
        args = kwargs if kwargs else {}

    if not isinstance(args, dict):
        if hasattr(args, "__dict__"):
            args = vars(args)
        else:
            args = {}

    if out_dir is None:
        out_dir = "."

    return run(str(out_dir), args)


if __name__ == "__main__":
    _main()