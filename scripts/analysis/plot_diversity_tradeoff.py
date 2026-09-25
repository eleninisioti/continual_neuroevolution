"""The stability-plasticity trade-off on gymnax with the novelty arm added:
forgetting against cumulative return, for the elite or the centroid.

    .venv/bin/python scripts/analysis/plot_diversity_tradeoff.py [--agent elite|centroid]

    -> projects/iclr_2027/paper/visuals/diversity_tradeoff_<agent>.{pdf,png,md}

plot_stability_plasticity.py's figure (its `--plasticity cum` variant) on
the six gymnax tasks of plot_diversity_plasticity.py -- noise, then action
reversal -- with the continual figures' reported arms plus GA + Novelty
(`dns_gaussian`), drawn with the same panel code (RL-dominated region, ring
for the best x - F that beats every method of the other family).

    x  Cum. return of the chosen agent: its training curve's mean over the run
       (`elite_eval_fitness` or `centroid_fitness`; RL's `mean_reward`, one
       policy, so the same for both agents)
    y  forgetting F of the same agent, from the forgetting pass
       (scripts/analysis/behavioural_divergence.py, `--agent_source finalgen`
       for the elite, `centroid` for the centroid): final-agent forgetting on
       noise, switch forgetting on action reversal (make_lineplot.load_divergence)

Both rescaled per task as the stability-plasticity figure does: divided by
(best - untrained), `best` the highest mean learning accuracy of any plotted
method for that agent. The grey lines are equal x - F; with Cum. on x they are
a guide, not the final-average-performance identity.

Where the numbers come from: every arm but two is the paper directory's pass
(`paper/<family>/results/<agent>`). GA + Novelty is the every-generation
re-score re-run (`runs_*_dnsrefresh`), and GA on MountainCar is the plain
gymnax GA (`<tree>/_ga_plain_backup`) rather than the paper's
`ga_focus_explore` -- the GA that GA + Novelty adds novelty to, as in
plot_diversity_plasticity.py. Both were scored by the same pass into
`paper/diversity/results/<family>/<agent>/`.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_diversity_plasticity as pdp                    # noqa: E402

psp, lp, PROJECT, REPO = pdp.psp, pdp.lp, pdp.PROJECT, pdp.REPO
OUT = PROJECT / 'paper/visuals/diversity_tradeoff'
SOURCES = PROJECT / 'paper/diversity/results'
FAMILY_KEY = {'noise': 'noise', 'action reversal': 'actions'}


def divergence(fam, sub, agent):
    """`{run_dir: F}` for one family and agent: the paper pass, overlaid by
    the diversity passes (which own the two substituted cell sets)."""
    out = {rd: v['F'] for rd, v in
           lp.load_divergence(PROJECT / 'paper' / sub / 'results' / agent).items()}
    for d in sorted((SOURCES / FAMILY_KEY[fam] / agent).glob('*')):
        out.update({rd: v['F'] for rd, v in lp.load_divergence(d).items()})
    return out


def rel(path):
    return str(pathlib.Path(path).relative_to(REPO))


def load(agent):
    """`{(fam, env): {method: {'x', 'F', 'la'}}}` per trial, raw returns."""
    score = pdp.SCORE[agent]
    data = {}
    for fam, tree, rerun, cells, sub in pdp.FAMILIES:
        forget = divergence(fam, sub, agent)
        arms = psp.reported_arms(PROJECT / tree / 'gymnax', list(cells.values()))
        arms = [m for m in arms if m != 'dns_gaussian'] + ['dns_gaussian']
        for env, cell in cells.items():
            panel = data[fam, env] = {}
            for m in arms:
                ne = lp.FAMILY.get(m) == 'ne'
                col = score if ne else 'mean_reward'
                src_tree = rerun if m == 'dns_gaussian' else tree
                xs, fs, las = [], [], []
                for t in pdp.trial_dirs(src_tree, m, cell):
                    f = forget.get(rel(t))
                    if f is None:
                        continue
                    r = pdp.read_trial(t, None, col)
                    xs.append(r['cum']); fs.append(f); las.append(r['la'])
                if xs:
                    panel[m] = {'x': np.array(xs), 'F': np.array(fs), 'la': np.array(las)}
                else:
                    print(f'note: {fam} {env} {m}: no run with both records and F')
    return data


def scaled_points(data, fam, env):
    panel = data[fam, env]
    floor = psp.FLOOR[env]
    best = max(float(np.mean(r['la'])) for r in panel.values())
    span = best - floor
    scaled = {env: {m: {'x': (r['x'] - floor) / span, 'y': r['F'] / span,
                        's': (r['x'] - floor - r['F']) / span}
                    for m, r in panel.items()}}
    return psp.points_for(scaled, [env], 'F'), (floor, best)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', choices=sorted(pdp.SCORE), default='elite')
    args = ap.parse_args()
    os.chdir(REPO)      # load_divergence resolves the recorded repo-relative run_dir
    plt.rcParams.update({'font.size': 9})
    data = load(args.agent)
    panels, scales = [], {}
    for fam, *_ in pdp.FAMILIES:
        for env in pdp.ENVS:
            points, scale = scaled_points(data, fam, env)
            scales.setdefault(('gymnax', fam), {})[env] = scale
            panels.append((f'{lp.ENV_TITLES[env]}, {fam}', points))
    methods = [m for m in lp.METHOD_ORDER if any(m in p for _, p in panels)]
    stem = f'{OUT}_{args.agent}'
    ncols = len(pdp.ENVS)
    psp.draw([(i // ncols, i % ncols, label, pts) for i, (label, pts) in enumerate(panels)],
             [], (len(pdp.FAMILIES), ncols), stem, 'F', methods,
             axes_in=(2.0, 1.8), legend_rows=3, x_name='Cum.',
             x_label=f'Cum. return ({args.agent})')
    psp.write_markdown(panels, scales, stem, 'F', 'cum')
    for label, pts in panels:
        s = psp.summary(pts)
        if s:
            print(f"{label:32s} best NE {s['ne']}, best RL {s['rl']}, ringed {s['ringed']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
