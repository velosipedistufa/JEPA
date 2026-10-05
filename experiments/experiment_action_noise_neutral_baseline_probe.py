import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE

@experiment_main
def run(out_dir: str, args: dict) -> dict:
    import torch

    args = args or {}

    def _get(key, default, cast):
        val = args.get(key, default)
        if val is None:
            return default
        try:
            return cast(val)
        except (TypeError, ValueError):
            return default

    seed = _get("seed", 0, int)
    torch.manual_seed(seed)

    n_steps = _get("n_steps", 16, int)
    batch = _get("batch", 8, int)
    noise_scale = _get("noise_scale", 0.05, float)
    horizon = _get("horizon", 4, int)
    action_dim = _get("action_dim", 7, int)

    if n_steps <= 0 or batch <= 0 or action_dim <= 0:
        return {
            "ok": True,
            "device": DEVICE,
            "successive_l2": 0.0,
            "pairwise_l2": 0.0,
            "costs_first": 0.0,
            "costs_final": 0.0,
            "cost_reduction": 0.0,
            "n_steps": n_steps,
            "noise_scale": noise_scale,
            "horizon": horizon,
            "action_dim": action_dim,
            "verdict": "baseline",
            "insights": ["No steps executed."],
        }

    x = torch.zeros(batch, action_dim, device=DEVICE)
    traj = []
    costs = []

    for _ in range(n_steps):
        eps = torch.randn_like(x) * noise_scale
        x = x + eps

        cost = (
            x.pow(2).sum(dim=-1).mean()
            + 0.01 * eps.pow(2).sum(dim=-1).mean()
        )

        traj.append(x.clone())
        costs.append(float(cost.item()))

    traj_t = torch.stack(traj, dim=1)

    if n_steps > 1:
        succ = float(
            (traj_t[:, 1:] - traj_t[:, :-1])
            .pow(2)
            .sum(-1)
            .sqrt()
            .mean()
            .item()
        )
    else:
        succ = 0.0

    pair_sum = 0.0
    pair_count = 0

    for i in range(traj_t.shape[0]):
        for j in range(i + 1, traj_t.shape[0]):
            pair_sum += float(
                (traj_t[i] - traj_t[j])
                .pow(2)
                .sum(-1)
                .sqrt()
                .mean()
                .item()
            )
            pair_count += 1

    pairwise_l2 = float(pair_sum / max(1, pair_count))
    cost_reduction = float(costs[0] - costs[-1]) / max(1e-12, abs(costs[0]))

    return {
        "ok": True,
        "device": DEVICE,
        "successive_l2": succ,
        "pairwise_l2": pairwise_l2,
        "costs_first": costs[0],
        "costs_final": costs[-1],
        "cost_reduction": cost_reduction,
        "n_steps": n_steps,
        "noise_scale": noise_scale,
        "horizon": horizon,
        "action_dim": action_dim,
        "verdict": "baseline",
        "insights": [
            f"Neutral noise baseline successive_l2={succ:.4f}, pairwise_l2={pairwise_l2:.4f}.",
            f"Cost reduction={cost_reduction:.6f} from {costs[0]:.6f} to {costs[-1]:.6f}.",
            "Use as reference against rotation/delta-z perturbations.",
        ],
    }