```python
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE
import json
import time


def _finite_float(value, default=0.0):
    try:
        v = float(value)
    except Exception:
        return default
    if v != v or v == float("inf") or v == float("-inf"):
        return default
    return v


def _tensor_stats(x):
    import torch

    x = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    if x.numel() == 1:
        val = _finite_float(x.item(), 0.0)
        return {"mean": val, "std": 0.0, "min": val, "max": val}

    mean_t = x.mean()
    std_t = ((x - mean_t).pow(2).mean()).sqrt()
    return {
        "mean": _finite_float(mean_t, 0.0),
        "std": _finite_float(std_t, 0.0),
        "min": _finite_float(x.min(), 0.0),
        "max": _finite_float(x.max(), 0.0),
    }


def _action_consistency(actions):
    import torch

    a = torch.as_tensor(actions, dtype=torch.float32)
    if a.numel() == 0:
        return {"pairwise_l2": 0.0, "successive_l2": 0.0}

    if a.dim() == 0:
        a = a.reshape(1, 1)
    elif a.dim() == 1:
        a = a.unsqueeze(0)

    a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
    if a.shape[0] <= 1:
        return {"pairwise_l2": 0.0, "successive_l2": 0.0}

    diffs = a[1:] - a[:-1]
    successive_l2 = _finite_float(diffs.norm(dim=-1).mean(), 0.0) if diffs.numel() else 0.0

    flat = a.reshape(a.shape[0], -1)
    if flat.shape[1] == 0:
        pairwise_l2 = 0.0
    else:
        dist = (flat.unsqueeze(0) - flat.unsqueeze(1)).norm(dim=-1)
        mask = ~torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
        pairwise_l2 = _finite_float(dist[mask].mean(), 0.0) if mask.any() else 0.0

    return {"pairwise_l2": pairwise_l2, "successive_l2": successive_l2}


def _fake_cost(actions, rot_scale, delta_z_ratio):
    import torch

    a = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)
    if a.dim() == 0:
        a = a.reshape(1, 1)
    elif a.dim() == 1:
        a = a.unsqueeze(0)

    if a.numel() == 0 or a.shape[-1] == 0:
        return 0.0

    a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
    d = a.shape[-1]

    rot_scale = _finite_float(rot_scale, 1e-6)
    rot_scale = max(rot_scale, 1e-6)
    delta_z_ratio = _finite_float(delta_z_ratio, 1.0)

    if d >= 6:
        rot = a[..., 3:6]
    else:
        rot = torch.zeros(a.shape[:-1] + (0,), dtype=torch.float32, device=DEVICE)

    if rot.numel() == 0:
        penalty = 0.0
    else:
        penalty_t = 0.01 * (rot / rot_scale).pow(2).sum(-1).mean()
        penalty = _finite_float(penalty_t, 0.0)

    if d >= 3:
        z = a[..., 2]
    else:
        z = torch.zeros(a.shape[:-1], dtype=torch.float32, device=DEVICE)

    if z.numel() == 0:
        dz = 0.0
    else:
        dz_t = ((z * delta_z_ratio) - z).abs().mean()
        dz = _finite_float(dz_t, 0.0)

    base_slice = a[..., :min(3, d)]
    if base_slice.numel() == 0:
        base = 0.0
    else:
        base_t = (base_slice - 0.1).pow(2).mean()
        base = _finite_float(base_t, 0.0)

    total = base + penalty + dz
    return _finite_float(total, 0.0)


@experiment_main
def run(out_dir: str, args: dict) -> dict:
    try:
        import torch

        if not isinstance(args, dict):
            args = {}

        def _float(key, default):
            return _finite_float(args.get(key, default), default)

        def _int(key, default):
            try:
                return int(args.get(key, default))
            except Exception:
                return default

        rot_scale = _float("rot_scale", 0.1)
        delta_z_ratio = _float("delta_z_ratio", 0.991)
        iters = max(1, _int("iters", 3))
        pop = max(1, _int("population", 64))
        elite = max(1, _int("elite", 8))
        action_dim = max(1, _int("action_dim", 7))
        elite = min(elite, pop)

        torch.manual_seed(0)

        out = {
            "device": str(DEVICE),
            "rot_scale": rot_scale,
            "delta_z_ratio": delta_z_ratio,
        }
        variants = {
            "baseline": {"rot_scale": 0.0, "delta_z_ratio": 1.0},
            "treatment": {"rot_scale": rot_scale, "delta_z_ratio": delta_z_ratio},
        }
        results = {}

        for name, cfg in variants.items():
            mean