"""Aggregate significance across settings for the ICLR paper (2026-09-21).

The per-setting tests of the figures (Mann-Whitney U, Holm) say where a method
wins; the counts in the text ("ES is ringed on five of ten settings") are not
themselves tested. This script tests them across settings:

1. Figure 2 (stability_plasticity data): the probability of improvement
   (rliable) of ES over every other method on LA - F, the return a task keeps
   at the end of the run, rescaled per setting as in the figure. Averaged over
   settings, with a 95% stratified bootstrap interval (trials resampled within
   each setting). ES beats a method when the interval's lower end is above 0.5.
   Once over all ten settings, once without the two where ES is a frozen
   specialist (its LA - F there is not retention).
2. Figure 4 (basin_width_main data): ES and the GA against PPO within each
   environment, pooling that environment's panels: a permutation test of the
   log action change, labels permuted within each panel (stratified), so the
   unit is the trial and no panel is treated as independent of its siblings.
3. Figure 3 (generalist_scores_centroid data): the probability of improvement
   of ES over every other method on the worse-direction rescaled score of the
   previous task (retention) and of the task just trained (learning), the two
   performance profiles of the figure, over its thirteen panels.

Writes paper/visuals/final/significance.md.

    .venv/bin/python scripts/analysis/aggregate_significance.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_stability_plasticity as psp     # noqa: E402
import plot_continual_lineplots as pcl      # noqa: E402
import plot_generalist_scores as pgs        # noqa: E402
import plot_basin_width_methods as pbw      # noqa: E402

OUT = psp.FINAL / 'significance'
N_BOOT = 5000
N_PERM = 4000
RL = ('ppo', 'trac', 'redo', 'cchain', 'pbt')
ORDER = ('ga', 'ppo', 'trac', 'redo', 'cchain', 'pbt')
NAME = {'es': 'ES', 'ga': 'GA', 'ppo': 'PPO', 'trac': 'TRAC-PPO', 'redo': 'ReDo-PPO',
        'cchain': 'C-CHAIN', 'pbt': 'PBT-PPO'}


def canon(m):
    """The data's arm names -> the paper's methods (every ES / PBT arm)."""
    if m in ('ga', *RL):
        return m
    if m.startswith('pbt'):
        return 'pbt'
    return 'es'


def poi(x, y):
    return ((x[:, None] > y[None]) + 0.5 * (x[:, None] == y[None])).mean()


def agg_poi(pairs, rng):
    """Mean over settings of P(a > b), and its stratified bootstrap interval."""
    point = float(np.mean([poi(a, b) for a, b in pairs]))
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = np.mean([poi(rng.choice(a, a.size), rng.choice(b, b.size)) for a, b in pairs])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def figure2(rng):
    meta = json.loads(psp.DATA.read_text())
    settings = []
    for fig, tree, cell, title in pcl.PANELS:
        if fig != 'main':
            continue
        key = pcl._key(tree, cell)
        env = cell.split('_sigma')[0]
        seeds = psp.as_arrays(meta['panels'][key])
        scaled, _ = psp.normalise({env: seeds}, 'F')
        s = {canon(m): v['s'] for m, v in scaled[env].items() if v['s'].size}
        frozen = {canon(m) for m in psp.frozen_methods(key)}
        settings.append((title, s, frozen))
    rows = []
    for subset, keep in (('all ten', lambda f: True), ('ES not frozen', lambda f: 'es' not in f)):
        use = [(t, s) for t, s, f in settings if keep(f)]
        for m in ORDER:
            pairs = [(s['es'], s[m]) for _, s in use if 'es' in s and m in s]
            p, lo, hi = agg_poi(pairs, rng)
            rows.append((subset, len(pairs), m, p, lo, hi))
    return rows, [(t, sorted(f)) for t, _, f in settings]


def stratified_perm(panels, rng):
    """panels: [(log delta of method, log delta of PPO)]. Statistic: mean over
    panels of (PPO mean - method mean), > 0 = method wider. One-sided p."""
    def stat(ps):
        return np.mean([b.mean() - a.mean() for a, b in ps])
    obs = stat(panels)
    count = 0
    for _ in range(N_PERM):
        perm = []
        for a, b in panels:
            z = rng.permutation(np.concatenate([a, b]))
            perm.append((z[:a.size], z[a.size:]))
        count += stat(perm) >= obs
    return float(obs), (count + 1) / (N_PERM + 1)


def figure4_return(rng):
    """The same test on the return-space width (plot_basin_width_return): log
    width, method minus PPO, > 0 = wider."""
    import plot_basin_width_return as pbr
    data, _ = pbr.widths('solved')
    by_env = {}
    for label, d in data.items():
        env = label.split(',')[0].split(' ')[0]
        by_env.setdefault(env, []).append(d)
    rows = []
    for m in ('es', 'ga', 'trac', 'redo', 'cchain', 'pbt'):
        for env, panels in [*by_env.items(), ('all', [d for p in by_env.values() for d in p])]:
            # widths are larger = wider, so pass PPO first: stat = method - PPO
            pairs = [(np.log(d['ppo']), np.log(d[m])) for d in panels if m in d and 'ppo' in d]
            if not pairs:
                continue
            obs, p = stratified_perm(pairs, rng)
            rows.append((m, env, len(pairs), float(np.exp(obs)), p))
    return rows


def figure4(rng):
    data = pbw.widths(0.1)
    absolute = set(json.loads(pbw.DATA.read_text()).get('absolute', []))
    by_env = {}
    for label, d in data.items():
        if label in absolute:
            continue
        env = label.split(',')[0].split(' ')[0]
        by_env.setdefault(env, []).append(d)
    rows = []
    for m in ('es', 'ga', 'trac', 'redo', 'cchain', 'pbt'):
        for env, panels in [*by_env.items(), ('all', [d for p in by_env.values() for d in p])]:
            pairs = [(np.log(d[m]), np.log(d['ppo'])) for d in panels
                     if m in d and 'ppo' in d and (d[m] > 0).all() and (d['ppo'] > 0).all()]
            if not pairs:
                continue
            obs, p = stratified_perm(pairs, rng)
            rows.append((m, env, len(pairs), float(np.exp(obs)), p))
    return rows


def figure3(rng):
    meta = json.loads(pgs.DATA.read_text())
    runs = {}
    for _, cells in pgs.GRID:
        for spec in cells:
            if not spec:
                continue
            tree, cell, _ = spec
            panel = meta['panels'].get(f'{tree}|{cell}')
            if not panel or not panel['arms']:
                continue
            split = {(m, d): pgs.trial_scores(v['trials'], cell, d)
                     for m, v in panel['arms'].items() for d in (0, 1)}
            floor = pgs.FLOOR[cell.split('_sigma')[0]]
            best_dir = max(s[:, 0].mean() for s in split.values() if s is not None)
            for m in panel['arms']:
                both = np.stack([split[m, d] for d in (0, 1) if split[m, d] is not None])
                runs.setdefault(canon(m), {})[f'{tree}|{cell}'] = (
                    (both - floor) / (best_dir - floor)).min(axis=0)
    rows = []
    for col, what in ((1, 'previous task'), (0, 'task just trained')):
        for m in ORDER:
            common = [k for k in runs['es'] if k in runs.get(m, {})]
            pairs = [(np.clip(runs['es'][k][:, col], 0, 1), np.clip(runs[m][k][:, col], 0, 1))
                     for k in common]
            p, lo, hi = agg_poi(pairs, rng)
            rows.append((what, len(pairs), m, p, lo, hi))
    return rows


def main() -> int:
    rng = np.random.default_rng(0)
    f2, frozen = figure2(rng)
    f3 = figure3(rng)
    f4 = figure4(rng)
    f4r = figure4_return(rng)
    md = ['# Aggregate significance', '', __doc__.split('\n\n', 1)[1].split('Writes')[0].strip(), '',
          '## Figure 2: P(ES > method) on LA - F, over settings', '',
          'Frozen methods per setting: ' + '; '.join(f'{t}: {", ".join(f) or "none"}'
                                                   for t, f in frozen), '',
          '| Settings | n | Method | P(ES > method) | 95% CI | ES better |', '|---|---|---|---|---|---|']
    md += [f'| {s} | {n} | {NAME[m]} | {p:.2f} | [{lo:.2f}, {hi:.2f}] | {"yes" if lo > 0.5 else "no"} |'
           for s, n, m, p, lo, hi in f2]
    md += ['', '## Figure 3: P(ES > method) on the rescaled worse-direction score', '',
           '| Score | panels | Method | P(ES > method) | 95% CI | ES better |', '|---|---|---|---|---|---|']
    md += [f'| {s} | {n} | {NAME[m]} | {p:.2f} | [{lo:.2f}, {hi:.2f}] | '
           f'{"yes" if lo > 0.5 else "worse" if hi < 0.5 else "no"} |' for s, n, m, p, lo, hi in f3]
    md += ['', '## Figure 4: width relative to PPO within each environment (stratified permutation)', '',
           '| Method | Environment | panels | geometric-mean ratio | one-sided p |', '|---|---|---|---|---|']
    md += [f'| {NAME[m]} | {e} | {n} | {r:.2f} | {p:.2g} |' for m, e, n, r, p in f4]
    md += ['', '## Figure 4, return-space width (plot_basin_width_return, rule solved): '
           'geometric-mean width ratio within each environment (stratified permutation)', '',
           '| Method | Environment | panels | geometric-mean ratio | one-sided p |', '|---|---|---|---|---|']
    md += [f'| {NAME[m]} | {e} | {n} | {r:.2f} | {p:.2g} |' for m, e, n, r, p in f4r]
    OUT.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    sys.exit(main())
