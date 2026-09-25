"""Does population diversity buy plasticity? The gymnax NE arms that differ in
how much diversity they keep, against how well they learn each new sub-task:

    ES             a Gaussian search distribution, the narrowest population
    GA             truncation selection + gaussian mutation
    GA + Novelty   the same GA with dominated-novelty selection (`dns_gaussian`),
                   the arm that keeps behavioural diversity on purpose

    .venv/bin/python scripts/analysis/plot_diversity_plasticity.py \
        [--diversity behavioural|genomic] [--metrics la cum rec] [--dns refresh|boundary]

    -> projects/iclr_2027/paper/visuals/diversity_plasticity[_genomic][_<metrics>].{pdf,png,md}

One column a task (continual_main's gymnax columns: noise, then action
reversal), one row a plasticity metric (default: the elite's cum. return
alone; `--metrics la cum` adds learning accuracy above it), one point a method: the mean over seeds with a 95%
percentile-bootstrap interval on both axes, each seed drawn faintly behind it.

    x  behavioural diversity: `bd_probe_disagreement`, how often two members
       choose different actions on a fixed probe batch, in [0, 1]; or, under
       `--diversity genomic`, `bd_genomic_diversity`, the mean pairwise weight
       distance (log axis). Both are the population tracker's
       (source/metrics/behaviour_tracking.py), logged every 10 generations,
       averaged over the whole run. Genomic width is not behaviour: the GA's
       weights spread further than GA + Novelty's on CartPole while its actions
       agree more, and ES's width is its sampling sigma.
    y  learning accuracy of the ELITE, LA = mean_i R[i][i], the plasticity axis
       of plot_stability_plasticity.py, read from the training records: the
       10-episode fresh-key `elite_eval_fitness` at the last generation of each
       phase (the forgetting pass's diagonal to within its evaluation noise,
       checked on the GA). Rescaled the same way, 0 = untrained network,
       1 = best mean LA of any continual method in the family, RL included.
       `cum`: the elite's curve mean over the whole run (the paper's Cum.
       elite up to generations / 1000), same rescaling -- reward collected
       throughout, so it also credits fast re-adaptation and keeping a
       sub-task. `rec`: the elite's mean over the first 20% of every phase
       after the first, how fast a switch is recovered from.

The elite by default; `--agent centroid` (-> `diversity_plasticity_centroid*`)
plots the population's weight mean instead. GA + Novelty keeps no
distribution, and averaging a behaviourally diverse repertoire's weights can
give a network worse than any of its members, so read its centroid as "has the
repertoire collapsed" as much as performance. `best` in the rescaling is the
best mean learning accuracy of the plotted agent or any RL method, as in
plot_stability_plasticity.py (which reads the centroid).

Where the runs come from. GA on MountainCar is the plain gymnax GA
(`<tree>/_ga_plain_backup`), not the paper's `ga_focus_explore`: GA + Novelty
is the plain GA plus novelty selection, so that is the comparison that
isolates novelty, and the focus runs log no behavioural diversity. The ES arm
is the family's (es_arm.json; NES in both). GA + Novelty is read from the
2026-09-16 re-runs that re-score the repertoire every generation
(`runs_*_dnsrefresh`, scripts/train/cluster/submit_gymnax_dns.sh) when all
their trials are there; `--dns boundary` -- or their absence -- falls back to
the earlier runs, which re-scored the population AT each switch (CLAUDE.md rule
(d)) and are stamped PRELIMINARY on the figure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.gridspec import GridSpec                   # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.ticker import MaxNLocator                  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr            # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_stability_plasticity as psp                    # noqa: E402

lp, PROJECT, REPO = psp.lp, psp.PROJECT, psp.REPO
OUT = PROJECT / 'paper/visuals/diversity_plasticity'
ENVS = psp.GYMNAX_ENVS
N_TRIALS = 10
RECOVERY_FRAC = 0.2

# (row title, continual tree, re-run tree, cell of each env, paper directory)
FAMILIES = [
    ('noise', 'runs_centroid', 'runs_centroid_dnsrefresh',
     {'CartPole_v1': 'CartPole_v1_sigma1.0', 'Acrobot_v1': 'Acrobot_v1_sigma1.0',
      'MountainCar_v0': 'MountainCar_v0_sigma0.1'}, 'gymnax/noise/10task'),
    ('action reversal', 'runs_actions', 'runs_actions_dnsrefresh',
     {e: f'{e}_sigma1.0' for e in ENVS}, 'gymnax/actions/2task'),
]
NE_ARMS = ('es', 'ga', 'dns_gaussian')
# Every continual arm whose LA can set the rescaling's `best`.
RL_ARMS = ('ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2')
DIVERSITY = {'behavioural': ('bd_probe_disagreement', 'Behavioural diversity'),
             'genomic': ('bd_genomic_diversity', 'Genomic diversity')}
# The network whose return is plotted, and its training-record column. The
# centroid is the ES distribution mean, the GA's archive mean and GA + Novelty's
# repertoire mean -- the last is a weight average of deliberately different
# networks, so expect it below that method's elite.
SCORE = {'elite': 'elite_eval_fitness', 'centroid': 'centroid_fitness'}
AGENT_TEXT = {'elite': 'ELITE (best individual)',
              'centroid': 'CENTROID (mean of the population weights)'}
# The rows. `la` is the stability-plasticity figure's learning accuracy: the
# elite's return at the end of each sub-task, averaged over sub-tasks.
METRIC_LABEL = {'la': 'Learning accuracy', 'cum': 'Cum. return',
                'rec': 'Return after a switch'}


def es_kept(sub):
    path = PROJECT / 'paper' / sub / 'es_arm.json'
    return json.loads(path.read_text())['kept'] if path.exists() else 'nes'


def trial_dirs(tree, method, cell):
    base = PROJECT / tree / 'gymnax/continual'
    if method == 'ga' and cell.startswith('MountainCar'):
        base = PROJECT / tree / '_ga_plain_backup/gymnax/continual'
    d = base / method / cell
    return sorted(p for p in d.glob('trial_*') if (p / 'training_metrics.json').exists())


def phase_ends(tasks):
    return list(np.flatnonzero(tasks[1:] != tasks[:-1])) + [len(tasks) - 1]


def read_trial(trial, column, score):
    """`{'div', 'la', 'cum', 'rec'}` of one run; `score` names the reported
    column (`elite_eval_fitness` for NE, `mean_reward` for RL)."""
    records = json.loads((trial / 'training_metrics.json').read_text())
    tasks = np.array([r['task'] for r in records])
    y = np.array([np.nan if r.get(score) is None else float(r[score]) for r in records])
    ends = phase_ends(tasks)
    starts = [0] + [e + 1 for e in ends[:-1]]
    early = [np.nanmean(y[a:a + max(1, int(RECOVERY_FRAC * (b + 1 - a)))])
             for a, b in zip(starts[1:], ends[1:])]
    div = (np.nan if column is None else
           float(np.nanmean([np.nan if r.get(column) is None else float(r[column])
                             for r in records])))
    return {'div': div, 'la': float(np.nanmean(y[ends])),
            # psp.curve_mean's definition, on the elite's column
            'cum': float(np.trapz(y, dx=1) / max(len(y) - 1, 1)),
            'rec': float(np.mean(early))}


def dns_tree(tree, rerun, cells, mode):
    """The tree GA + Novelty is read from, and whether it is the fixed one."""
    if mode == 'refresh':
        n = [len(trial_dirs(rerun, 'dns_gaussian', c)) for c in cells.values()]
        if min(n) >= N_TRIALS:
            return rerun, True
        print(f'note: {rerun} has {n} dns_gaussian trials; using the boundary-rescored '
              f'runs in {tree} (PRELIMINARY)')
    return tree, False


def load(column, dns_mode, score='elite_eval_fitness'):
    """`{(family, env): {method: {'div', 'la', 'rec'}}}`, the rescaling, and
    whether every GA + Novelty cell is the re-run."""
    data, scale, fixed = {}, {}, True
    for fam, tree, rerun, cells, sub in FAMILIES:
        d_tree, ok = dns_tree(tree, rerun, cells, dns_mode)
        fixed &= ok
        es = es_kept(sub)
        for env, cell in cells.items():
            panel = data[fam, env] = {}
            for m in (es, 'ga', 'dns_gaussian'):
                rows = [read_trial(t, column, score)
                        for t in trial_dirs(d_tree if m == 'dns_gaussian' else tree, m, cell)]
                if rows:
                    panel['es' if m == es else m] = {
                        k: np.array([r[k] for r in rows]) for k in rows[0]}
            best = max(float(np.mean(r['la'])) for r in panel.values())
            for m in RL_ARMS:
                rows = [read_trial(t, None, 'mean_reward') for t in trial_dirs(tree, m, cell)]
                if len(rows) >= N_TRIALS:
                    best = max(best, float(np.mean([r['la'] for r in rows])))
            scale[fam, env] = (psp.FLOOR[env], best)
    return data, scale, fixed


def boot(v, seed=0):
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    b = rng.choice(v, (psp.N_BOOT, v.size)).mean(axis=1)
    return float(v.mean()), *map(float, np.percentile(b, [2.5, 97.5]))


def boot(v, seed=0):
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    b = rng.choice(v, (psp.N_BOOT, v.size)).mean(axis=1)
    return float(v.mean()), *map(float, np.percentile(b, [2.5, 97.5]))


def panels():
    """`(family, env)` in drawing order: the noise columns, then action reversal."""
    return [(fam, env) for fam, *_ in FAMILIES for env in ENVS]


def draw(data, scale, metrics, x_label, log_x, stem, stamp):
    """One row a plasticity metric, one column a task; `{(fam, env, m, metric):
    (px, py, x, y)}` back for the tables."""
    cols = panels()
    nrows, ncols = len(metrics), len(cols)
    axes_in, left, right, gap_w, top, bottom, gap_h = (1.55, 1.4), 0.65, 0.2, 0.45, 0.85, 0.45, 0.3
    width = left + right + ncols * axes_in[0] + (ncols - 1) * gap_w
    height = top + bottom + nrows * axes_in[1] + (nrows - 1) * gap_h
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(nrows, ncols, figure=fig, left=left / width, right=1 - right / width,
                  top=1 - top / height, bottom=bottom / height,
                  wspace=gap_w / axes_in[0], hspace=gap_h / axes_in[1])
    points = {}
    for i, key in enumerate(metrics):
        for j, (fam, env) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            floor, best = scale[fam, env]
            for m in NE_ARMS:
                r = data[fam, env].get(m)
                if r is None or not np.isfinite(r['div']).any():
                    continue
                x = r['div']
                y = (r[key] - floor) / (best - floor)
                colour = psp._colour(m)
                ax.plot(x, y, ls='', marker='o', ms=2.4, color=colour, alpha=0.35,
                        mec='none', zorder=2)
                px, py = boot(x), boot(y)
                points[fam, env, m, key] = (px, py, x, y)
                ax.errorbar([px[0]], [py[0]], xerr=[[px[0] - px[1]], [px[2] - px[0]]],
                            yerr=[[py[0] - py[1]], [py[2] - py[0]]], fmt='o', ms=6,
                            color=colour, mec='white', mew=0.6, elinewidth=0.9,
                            capsize=0, zorder=4)
            if log_x:
                ax.set_xscale('log')
            else:
                ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(f'{lp.ENV_TITLES[env]}\n{fam}', fontweight='bold')
            if i == nrows - 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[key])
    handles = [Line2D([], [], ls='', marker='o', ms=6, color=psp._colour(m)) for m in NE_ARMS]
    fig.legend(handles, [psp._label(m) for m in NE_ARMS], loc='upper center',
               ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1 - 0.03 / height),
               handletextpad=0.3, columnspacing=1.2)
    if stamp:
        fig.text(0.005, 1 - 0.05 / height, 'PRELIMINARY: GA + Novelty\nre-scored at switches',
                 ha='left', va='top', color='#B22222', fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {stem}.pdf, {stem}.png  ({width:.1f} x {height:.1f} in)')
    return points


def write_markdown(points, metrics, stem, x_name, fixed, agent='elite'):
    ci = lambda p: f'{p[0]:.3g} [{p[1]:.3g}, {p[2]:.3g}]'   # noqa: E731
    names = [METRIC_LABEL[k] for k in metrics]
    lines = [f'# {pathlib.Path(stem).name}', '',
             'Built by `scripts/analysis/plot_diversity_plasticity.py` (definitions in its '
             f'docstring). The {AGENT_TEXT[agent]} of every method. Mean over seeds '
             '[95% percentile-bootstrap CI]; the plasticity values are rescaled per panel, '
             '0 = untrained network, 1 = best mean LA of any continual method there.', '',
             ('GA + Novelty: the every-generation re-score re-runs.' if fixed else
              '**PRELIMINARY**: GA + Novelty is the runs that re-scored their population at '
              'every switch (boundary information); the re-runs are not all in.'), '',
             f'| Task | Method | n | {x_name} | ' + ' | '.join(names) + ' |',
             '|---|---|---|---|' + '---|' * len(metrics)]
    for fam, env in panels():
        for m in NE_ARMS:
            got = [points.get((fam, env, m, k)) for k in metrics]
            if got[0] is None:
                continue
            lines.append(f'| {lp.ENV_TITLES[env]}, {fam} | {psp._label(m)} | {got[0][2].size} | '
                         f'{ci(got[0][0])} | ' + ' | '.join(ci(g[1]) for g in got) + ' |')
    lines += ['', '## GA + Novelty against GA', '',
              'Difference of means, two-sided Mann-Whitney U on the per-seed values (not '
              'corrected); rho is Spearman of diversity against the metric over every seed '
              'of the three methods in the panel.', '',
              f'| Task | {x_name} | ' + ' | '.join(f'{n} | rho' for n in names) + ' |',
              '|---|---|' + '---|---|' * len(metrics)]
    fmt = lambda a, b: f'{a.mean() - b.mean():+.3g} (p {mannwhitneyu(a, b).pvalue:.2g})'  # noqa: E731
    for fam, env in panels():
        a = {k: points.get((fam, env, 'dns_gaussian', k)) for k in metrics}
        b = {k: points.get((fam, env, 'ga', k)) for k in metrics}
        if None in a.values() or None in b.values():
            continue
        row = [fmt(a[metrics[0]][2], b[metrics[0]][2])]
        for k in metrics:
            xs = np.concatenate([points[fam, env, m, k][2] for m in NE_ARMS if (fam, env, m, k) in points])
            ys = np.concatenate([points[fam, env, m, k][3] for m in NE_ARMS if (fam, env, m, k) in points])
            ok = np.isfinite(xs) & np.isfinite(ys)
            rho = spearmanr(xs[ok], ys[ok])[0] if ok.sum() > 5 else np.nan
            row += [fmt(a[k][3], b[k][3]), f'{rho:+.2f}']
        lines.append(f'| {lp.ENV_TITLES[env]}, {fam} | ' + ' | '.join(row) + ' |')
    pathlib.Path(f'{stem}.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {stem}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--diversity', choices=sorted(DIVERSITY), default='behavioural')
    ap.add_argument('--metrics', nargs='+', choices=sorted(METRIC_LABEL), default=['cum'],
                    help='one row each, in this order')
    ap.add_argument('--dns', choices=['refresh', 'boundary'], default='refresh',
                    help='which GA + Novelty runs; refresh falls back to boundary '
                         'until the re-runs are complete')
    ap.add_argument('--agent', choices=sorted(SCORE), default='elite',
                    help="which network's return is the plasticity axis")
    args = ap.parse_args()
    plt.rcParams.update({'font.size': 9})
    column, x_label = DIVERSITY[args.diversity]
    data, scale, fixed = load(column, args.dns, SCORE[args.agent])
    stem = (str(OUT) + ('' if args.agent == 'elite' else f'_{args.agent}')
            + ('' if args.diversity == 'behavioural' else f'_{args.diversity}')
            + ('' if args.metrics == ['cum'] else '_' + '_'.join(args.metrics)))
    pathlib.Path(stem).parent.mkdir(parents=True, exist_ok=True)
    points = draw(data, scale, args.metrics, x_label, args.diversity == 'genomic', stem, not fixed)
    write_markdown(points, args.metrics, stem, x_label, fixed, args.agent)
    return 0


if __name__ == '__main__':
    sys.exit(main())
