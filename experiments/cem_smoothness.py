import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'automation'))
from env_common import experiment_main, DEVICE
import torch, numpy as np, lance, json

def load_human_actions(ds, n_episodes=50):
    eps = torch.unique(torch.tensor(ds.to_table(columns=['episode_idx'])['episode_idx'].to_numpy()))[:n_episodes]
    seqs = []
    for ep in eps.tolist():
        mask = torch.tensor(ds.to_table(columns=['episode_idx'])['episode_idx'].to_numpy()) == ep
        idx = torch.nonzero(mask).squeeze(1)
        if len(idx) < 5: continue
        acts = torch.stack([torch.tensor(ds.take([int(i)], columns=['action']).to_pylist()[0]['action']) for i in idx])
        seqs.append(acts)
    return seqs

def jerk(actions):
    # actions: T x 2
    if len(actions) < 4: return None
    a = actions.numpy()
    vel = np.diff(a, axis=0)
    acc = np.diff(vel, axis=0)
    jer = np.diff(acc, axis=0)
    return np.mean(np.sum(jer ** 2, axis=1))

def run(out_dir, args):
    from env_common import find_lance_dataset, action_stats_from_dataset
    ds, path, fmt = find_lance_dataset()
    stats = action_stats_from_dataset()
    human_seqs = load_human_actions(ds, n_episodes=args.get('n_episodes', 50))
    human_jerks = [jerk(s) for s in human_seqs if jerk(s) is not None]
    human_jerk_mean = float(np.mean(human_jerks))
    human_jerk_std = float(np.std(human_jerks))

    # Simulate CEM proposals with same mean/std as dataset
    mu = stats['mean'].to(DEVICE)
    std = stats['std'].to(DEVICE)
    T = args.get('horizon', 16)
    cem_jerks = []
    for _ in range(args.get('n_samples', 500)):
        seq = mu + std * torch.randn(T, 2, device=DEVICE)
        j = jerk(seq.cpu())
        if j is not None: cem_jerks.append(j)
    cem_jerk_mean = float(np.mean(cem_jerks))
    cem_jerk_std = float(np.std(cem_jerks))

    ratio = cem_jerk_mean / (human_jerk_mean + 1e-9)
    verdict = 'success' if ratio > 1.2 else 'neutral'
    metrics = {
        'human_jerk_mean': human_jerk_mean,
        'human_jerk_std': human_jerk_std,
        'cem_jerk_mean': cem_jerk_mean,
        'cem_jerk_std': cem_jerk_std,
        'jerk_ratio_cem_to_human': ratio,
        'n_human_episodes': len(human_jerks),
        'n_cem_samples': len(cem_jerks),
        'horizon': T
    }
    return {'verdict': verdict, 'metrics': metrics}

if __name__ == '__main__':
    experiment_main(run)
