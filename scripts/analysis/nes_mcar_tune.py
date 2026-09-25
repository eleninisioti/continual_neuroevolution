"""Read the NES MountainCar tuning sweep (queue_iclr_nes_mcar_tune.sh).

One row an arm: how the centroid does on STATIONARY MountainCar at each
(sigma, learning rate), against the reported NES / OpenES / GA / PPO runs. The
continual phases are 200 generations long, so the column that decides is
`@200` -- what the centroid scores at the end of a phase-length window -- not
the 600-generation final. "Solved" is the paper's -150 threshold on the
reported curve (`centroid_fitness`, 10 fresh episodes); "held" is the share of
generations after the first solve that stay solved, which is the instability
the reported setting shows (finds the goal, loses it).

    .venv/bin/python scripts/analysis/nes_mcar_tune.py [--root ...] [--out ...]

Writes <out>/summary.md and <out>/curves.png.
"""
import argparse, glob, json, os, re
import numpy as np

THRESH = -150.0
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = {  # reported stationary runs, trials 1-5 share the sweep's seeds
    'nes (reported: s0.1 lr0.05)': 'projects/iclr_2027/runs_centroid/gymnax/noncontinual/nes/MountainCar_v0',
    'openes (s0.1 lr0.1, Adam)': 'projects/iclr_2027/runs_centroid/gymnax/noncontinual/es/MountainCar_v0',
    'ga': 'projects/iclr_2027/runs_centroid/gymnax/noncontinual/ga/MountainCar_v0',
}


def curve(trial):
    m = json.load(open(os.path.join(trial, 'training_metrics.json')))
    g = np.array([r['generation'] for r in m], float)
    key = 'centroid_fitness' if 'centroid_fitness' in m[0] else 'mean_reward'
    c = np.array([np.nan if r.get(key) is None else r[key] for r in m], float)
    w = np.array([np.nan if r.get('weight_rms') is None else r['weight_rms'] for r in m], float)
    p = np.array([np.nan if r.get('mean_fitness') is None else r['mean_fitness'] for r in m], float)
    curve.extra = (w, p)
    return g, c


def stats(g, c):
    w, p = curve.extra
    wrms = np.nanmean(w[g >= g.max() - 49]) if np.isfinite(w).any() else np.nan
    popmean = np.nanmean(p[g >= g.max() - 49]) if np.isfinite(p).any() else np.nan
    at200 = np.nanmean(c[(g >= 180) & (g < 200)])
    final = np.nanmean(c[g >= g.max() - 49])
    first = np.flatnonzero(c > THRESH)
    if len(first):
        f0 = int(first[0]); held = float(np.mean(c[f0:] > THRESH)); first_gen = float(g[f0])
    else:
        held = 0.0; first_gen = np.nan
    return dict(at200=at200, final=final, best=np.nanmax(c), first=first_gen, held=held, wrms=wrms, popmean=popmean,
                solved200=at200 > THRESH, solved600=final > THRESH)


def read_arm(root, n=None):
    trials = sorted(glob.glob(os.path.join(root, 'trial_*')), key=lambda s: int(s.rsplit('_', 1)[1]))
    if n: trials = [t for t in trials if int(t.rsplit('_', 1)[1]) <= n]
    rows, curves = [], []
    for t in trials:
        if not os.path.exists(os.path.join(t, 'training_metrics.json')):
            continue
        g, c = curve(t); rows.append(stats(g, c)); curves.append((g, c))
    return rows, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='projects/iclr_2027/runs_hparam/nes_mcar_tune/gymnax/noncontinual')
    ap.add_argument('--out', default='projects/iclr_2027/runs_hparam/nes_mcar_tune')
    ap.add_argument('--num_trials', type=int, default=5)
    a = ap.parse_args()
    os.chdir(REPO)
    arms = {}
    for d in sorted(glob.glob(os.path.join(a.root, 'nes_sigma*_lr*'))):
        arms[os.path.basename(d)] = os.path.join(d, 'MountainCar_v0')
    # the population-split waves: same budget (pop x evals = 1536), other trees
    for tree, tag in [('nes_mcar_tune_pop1536ev1', '_pop1536ev1'), ('nes_mcar_tune_pop256ev6', '_pop256ev6')]:
        for d in sorted(glob.glob(os.path.join(os.path.dirname(a.root.rstrip('/')).replace('nes_mcar_tune', tree) if False else
                                               os.path.join('projects/iclr_2027/runs_hparam', tree, 'gymnax/noncontinual'), 'nes_sigma*_lr*'))):
            arms[os.path.basename(d) + tag] = os.path.join(d, 'MountainCar_v0')
    table, allcurves = [], {}
    for name, root in list(REF.items()) + list(arms.items()):
        rows, curves = read_arm(root, a.num_trials)
        if not rows: continue
        allcurves[name] = curves
        agg = {k: np.nanmean([r[k] for r in rows]) for k in ['at200', 'final', 'best', 'held', 'wrms', 'popmean']}
        agg['first'] = np.nanmedian([r['first'] for r in rows])
        agg['solved200'] = sum(r['solved200'] for r in rows); agg['solved600'] = sum(r['solved600'] for r in rows)
        agg['n'] = len(rows); agg['name'] = name
        m = re.match(r'nes_sigma([\d.]+)_lr([\d.]+)', name)
        agg['sigma'], agg['lr'] = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
        table.append(agg)
    # sort the sweep arms by @200 (the phase-length window), references first
    refs = [r for r in table if r['name'] in REF]; sweep = sorted([r for r in table if r['name'] not in REF], key=lambda r: -r['at200'])
    lines = ['# NES on stationary MountainCar: width x step sweep', '',
             f'Centroid (`centroid_fitness`, 10 fresh episodes). `@200` = mean over generations 180-199 (one continual phase is 200 generations); '
             f'`final` = mean over the last 50 of 600; `first` = median generation of the first score above {THRESH:g}; '
             f'`held` = mean share of generations after the first solve that stay above {THRESH:g}; solved = trials above {THRESH:g} at that window. Sorted by `@200`.', '',
             '`popmean` = population-mean fitness over the last 50 generations (how much of the population solves); `wrms` = weight RMS of the centroid there (with argmax policies the relative perturbation is sigma*sqrt(d)/|theta|, so a small wrms at a fixed sigma means a wide, mostly failing population).', '',
             '| Arm | sigma | lr | lr/sigma | @200 | final | best | first gen | held | solved @200 | solved @600 | popmean | wrms | n |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in refs + sweep:
        ratio = '' if np.isnan(r['sigma']) else f"{r['lr']/r['sigma']:.2f}"
        sg = '' if np.isnan(r['sigma']) else f"{r['sigma']:g}"; lr = '' if np.isnan(r['lr']) else f"{r['lr']:g}"
        lines.append(f"| {r['name']} | {sg} | {lr} | {ratio} | {r['at200']:.0f} | {r['final']:.0f} | {r['best']:.0f} | {r['first']:.0f} | {r['held']:.2f} | {r['solved200']}/{r['n']} | {r['solved600']}/{r['n']} | {r['popmean']:.0f} | {r['wrms']:.2f} | {r['n']} |")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    # curves: one panel an arm, every seed
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = [r['name'] for r in refs + sweep]
    ncol = 5; nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        for g, c in allcurves[name]:
            ax.plot(g, c, lw=0.8, alpha=0.8)
        ax.axhline(THRESH, color='k', lw=0.5, ls=':'); ax.axvline(200, color='k', lw=0.5, ls=':')
        ax.set_title(name, fontsize=8); ax.set_ylim(-510, -80)
    for ax in axes.flat[len(names):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'curves.png'), dpi=120)
    print('wrote', os.path.join(a.out, 'summary.md'), os.path.join(a.out, 'curves.png'))


if __name__ == '__main__':
    main()
