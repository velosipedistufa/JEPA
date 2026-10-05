import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'automation'))
from env_common import experiment_main, DEVICE

@experiment_main
def run(out_dir, args):
    import json, time, torch
    t0 = time.time()
    params = args.get('params', {}) if isinstance(args, dict) else {}
    pop = int(params.get('population', 64))
    iters = int(params.get('iterations', 3))
    elite_fraction = float(params.get('elite_fraction', 0.25))
    elite_k = max(2, int(pop * elite_fraction))
    noise_scale = float(params.get('noise_scale', 0.35))
    seed = int(params.get('seed', 0))
    time_limit = float(params.get('time_limit', 6.0))
    torch.manual_seed(seed)
    device = torch.device(DEVICE if torch.cuda.is_available() or str(DEVICE) == 'cpu' else 'cpu')
    stats_paths = [params.get('stats_path', ''), 'automation/action_stats.json', 'data/action_stats.json', 'action_stats.json']
    stats = {}
    for path in stats_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as fh:
                    stats = json.load(fh)
                break
            except Exception:
                pass
    dim = int(params.get('action_dim', len(stats.get('mean', [0.0] * 7))))
    mean = torch.tensor(stats.get('mean', [0.0] * dim), dtype=torch.float32, device=device)
    std = torch.tensor(stats.get('std', [1.0] * dim), dtype=torch.float32, device=device).clamp_min(1e-3)
    if mean.numel() != dim:
        mean = torch.zeros(dim, device=device)
    if std.numel() != dim:
        std = torch.ones(dim, device=device)
    def surrogate_score(actions):
        z = (actions - mean) / std
        nll = z.pow(2).mean(dim=-1)
        rot_penalty = actions[:, 3:6].abs().mean(dim=-1) if actions.shape[-1] >= 6 else torch.zeros(actions.shape[0], device=device)
        dz_prior = torch.exp(-(actions[:, 2].abs() - 1.0).pow(2)) if actions.shape[-1] >= 3 else torch.ones(actions.shape[0], device=device)
        return -nll - 2.0 * rot_penalty + 0.25 * dz_prior
    objective = surrogate_score
    try:
        from env_common import score_actions
        def objective(actions):
            return torch.as_tensor(score_actions(actions), dtype=torch.float32, device=device)
    except Exception:
        pass
    m = mean.clone()
    s = std * noise_scale
    best_score = float('-inf')
    best_action = m.clone()
    history = []
    for i in range(iters):
        if time.time() - t0 > time_limit:
            break
        samples = (m + torch.randn(pop, dim, device=device) * s).clamp(-5.0, 5.0)
        scores = objective(samples)
        if scores.numel() == 0:
            break
        vals, idx = scores.sort(descending=True)
        if float(vals[0]) > best_score:
            best_score = float(vals[0])
            best_action = samples[idx[0]].detach().clone()
        elite = samples[idx[:elite_k]]
        m = elite.mean(dim=0)
        s = elite.std(dim=0).clamp_min(1e-3)
        history.append({'iter': i, 'best': float(vals[0]), 'mean_score': float(scores.mean())})
    elapsed = time.time() - t0
    verdict = 'success' if best_score != float('-inf') and elapsed < 7.5 else 'failure'
    return {'verdict': verdict, 'score': best_score, 'stats_found': bool(stats), 'action_dim': dim, 'history': history, 'elapsed_s': elapsed}