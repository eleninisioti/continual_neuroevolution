"""The paper's metrics figure: the lineplot's table, drawn.

    .venv/bin/python scripts/make_metrics_figure.py projects/iclr_2027/runs_centroid/gymnax \\
        --phase continual --cells CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 \\
        MountainCar_v0_sigma0.1 --metric elite_eval --methods ... \\
        --results-dir <results>/elite --out <dir>/metrics_elite

One panel per (metric, environment): every method as a point at its MEAN over
seeds with a horizontal 95% percentile-bootstrap confidence interval, in the
lineplot's colours and order, with the cross-family significance mark beside
it. Four outputs beside `--out`:

    <out>.png / .pdf    the figure
    <out>_table.md      the number under every point
    <out>_stats.md      every test behind every mark: U, raw and Holm p,
                        effect size, and the smallest p the seed counts allow
    <out>_values.json   the per-seed values behind every panel, which
                        `scripts/analysis/plot_metrics_overview.py` redraws
                        across families without reloading a run tree

Every number here is read through `make_lineplot.load_report`,
`make_lineplot.metric_table` and `make_lineplot.posthoc_table`, and every
interval through `source.metrics.continual_metrics.bootstrap_ci`, so the
figure, the table and the lineplot's band cannot come to disagree: same runs,
same arms, same cells, same estimator. The flags are the lineplot's
(`--cells`, `--metric`, `--methods`, `--results-dir`, `--ref-root`), plus
`--rows` to pick which rows are drawn.

## Reading it

    Cum.   the area under the training curve, /1000: the headline row, named
           after `--metric` (Cum. elite, Cum. centroid).
    Final  final performance: each sub-task's own agent, saved at the end of
           that sub-task and re-scored on it by the `evaluate` pass, meaned
           over sub-tasks. How well every task was finally learnt, with no
           credit for speed (Cum. has that) and no charge for what is
           forgotten afterwards (F has that). Not drawn under `--metric
           centroid` unless `--rows` names it (the paper's centroid figure
           omits it, 2026-09-15).
    FT     forward transfer against the method's OWN stationary run: read it
           beside Cum. and Final, because a weak stationary run has little to
           lose.
    F      forgetting: the final agent's drop on every earlier sub-task, or,
           with two alternating sub-tasks, the drop at every switch. Lower is
           better.
    BD     behavioural divergence at the switch, in [0, 1], lower is better.
    ZT     zero-shot transfer to the next sub-task, higher is better.

Not drawn unless asked for (`--rows ... final_all`):

    Final, all sub-tasks   the END-of-sequence agent on every sub-task,
           meaned: Continual World's average performance, from the forgetting
           pass's reward matrix. It is about Final minus F, so it adds no
           information to the default rows; under a two-regime alternation it
           is the generalist score.

A post-hoc row with no values in any cell is not drawn, and the script says
which pass fills it. A dotted rule separates the NE arms from the RL ones.

## The mark

A method is marked when it beat EVERY member of the other family in that
row: one-sided Mann-Whitney U per pair, Holm-corrected within the method's
own comparisons, graded on the weakest adjusted p (`*` < 0.05, `**` < 0.01,
`***` < 0.001). `<out>_stats.md` says what that does and does not control.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import make_lineplot as lp                                # noqa: E402
from source.metrics.continual_metrics import (            # noqa: E402
    SIG_LEVELS, bootstrap_ci, holm, mann_whitney_tests)

# Row key -> (label, higher_is_better, column). The columns in CURVE_COLUMNS
# come from `metric_table`, every other one from `posthoc_table`.
ROWS = {
    'cum':       (None,        True,  'cum_max'),   # label follows --metric
    'cum_mean':  ('Cum. mean', True,  'cum_mean'),
    'final':     ('Final',     True,  'End'),
    'final_all': ('Final, all sub-tasks', True, 'FinalAll'),
    'ft':        ('FT',        True,  'ft'),
    'F':         ('F',         False, 'F'),
    'BD':        ('BD',        False, 'BD'),
    'ZT':        ('ZT',        True,  'ZT'),
}
CURVE_COLUMNS = ('cum_max', 'cum_mean', 'ft')
DEFAULT_ROWS = ['cum', 'final', 'ft', 'F', 'BD', 'ZT']
# `--rows` when not given: the paper's centroid figure has no Final row.
DEFAULT_ROWS_FOR = {'centroid': [r for r in DEFAULT_ROWS if r != 'final']}
# Which pass fills a post-hoc column, for the note when one is empty.
FILLED_BY = {'End': 'the `evaluate` pass', 'ZT': 'the `evaluate` pass',
             'F': 'the `diverge` pass', 'BD': 'the `diverge` pass',
             'FinalAll': 'the `diverge` pass'}
FAMILY_NAMES = {'ne': 'NE', 'rl': 'RL'}


def _label(m):
    return lp.METHOD_STYLE.get(m, {}).get('label', m)


def _p(p):
    return '--' if p is None else f'{p:.2g}'


def draw_panel(ax, cell, methods, key, hib, fs, labels=True):
    """One (row, environment) panel: every method in `cell` ({method:
    [per-seed values]}) as its mean over seeds with a 95% bootstrap CI and
    the cross-family mark, one line per entry of `methods` top to bottom
    whether or not it has values, so panels drawn with the same `methods`
    line up. Returns the Mann-Whitney tests and one (method, mean, lo, hi, n,
    mark) per point drawn. Shared with `plot_metrics_overview.py`."""
    ypos = {m: len(methods) - 1 - i for i, m in enumerate(methods)}
    fam = [lp.FAMILY.get(m) for m in methods]
    split = next((i for i in range(1, len(fam)) if fam[i] != fam[i - 1]), None)
    tests = mann_whitney_tests(cell, lp.FAMILY, hib)
    points = []
    for m in methods:
        v = np.asarray([x for x in cell.get(m, []) if np.isfinite(x)])
        if not v.size:
            continue
        mark = tests[m]['mark']
        mean, lo, hi = bootstrap_ci(v)
        colour = lp.METHOD_STYLE.get(m, {}).get('color')
        ax.plot([lo, hi], [ypos[m]] * 2, color=colour, lw=1.0,
                solid_capstyle='butt', zorder=2)
        ax.plot([mean], [ypos[m]], 'o', color=colour, ms=3.0,
                mec='white', mew=0.4, zorder=3)
        if mark:
            ax.annotate(mark, (hi, ypos[m]), xytext=(2, 0),
                        textcoords='offset points', va='center',
                        ha='left', fontsize=fs - 1, color=colour,
                        annotation_clip=False)
        points.append((m, mean, lo, hi, v.size, mark))
    if split is not None:
        ax.axhline(len(methods) - split - 0.5, color='0.5', lw=0.5,
                   ls=':', zorder=1)
    if key == 'BD':
        ax.set_xlim(0, 1)
    # Not `sharey`: a shared axis shares its formatter, so blanking the inner
    # columns' labels blanked the first column's too.
    ax.set_yticks([ypos[m] for m in methods])
    ax.set_yticklabels([_label(m) for m in methods] if labels else [])
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.tick_params(axis='y', length=0)
    # Room on the right for the significance mark beside the widest interval;
    # the marks are drawn unclipped, so without this they sit on the
    # neighbouring panel.
    ax.margins(x=0.12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='0.9', lw=0.5, zorder=0)
    return tests, points


def mark_floor(rec):
    """The smallest weakest-adjusted p a method's comparisons can reach with
    its seed counts: every comparison perfectly separated and tie-free, put
    through the SAME scipy call the marks use, then Holm. Not the exact
    test's 1 / C(n1 + n2, n1): from nine seeds a side scipy switches to the
    normal approximation, whose floor is an order of magnitude higher (10 vs
    10: 9.1e-05 per pair, 3.7e-04 after Holm over four rivals). At or above
    0.05 no mark was possible whatever the data, and a blank means "too few
    seeds", not "no difference"."""
    from scipy.stats import mannwhitneyu
    comps = rec['comparisons']
    if not comps:
        return None
    n = rec['n']
    return float(np.max(holm([
        mannwhitneyu(np.arange(c['n_rival'], c['n_rival'] + n),
                     np.arange(c['n_rival']), alternative='greater').pvalue
        for c in comps])))


def write_stats(path, title, context, stats, methods):
    """`<out>_stats.md`: the test, what it controls, and every comparison."""
    fam_of = {m: lp.FAMILY.get(m) for m in methods}
    families = list(dict.fromkeys(f for f in fam_of.values() if f))
    members = {f: [m for m in methods if fam_of[m] == f] for f in families}
    untested = [m for m in methods if not fam_of[m]]
    claims = sum(1 for *_, tests in stats for r in tests.values()
                 if r['comparisons'])
    pairs = sum(len(r['comparisons']) for *_, tests in stats
                for r in tests.values())
    levels = ', '.join(f'`{mark}` p < {level:g}'
                       for level, mark in reversed(SIG_LEVELS))

    md = [f'# {title}: significance tests', '', context, '',
          '## The test', '',
          '- **Unit.** One trial (seed) is one observation. Each panel '
          'compares the per-seed values drawn as that panel\'s points.',
          '- **Families.** ' + '; '.join(
              f'{FAMILY_NAMES.get(f, f)}: '
              + ', '.join(_label(m) for m in members[f]) for f in families)
          + '.' + (f' Not in either family, so never tested: '
                   f'{", ".join(_label(m) for m in untested)}.'
                   if untested else ''),
          '- **Question.** Is a method better than EVERY method of the other '
          'family on this row? Methods of the same family are never compared '
          'with each other.',
          '- **Test.** One-sided Mann-Whitney U (`scipy.stats.mannwhitneyu`, '
          'default `auto` method: the exact distribution for small tie-free '
          'samples, otherwise the normal approximation with continuity and '
          'tie correction), in the direction '
          'the row is read: greater for Cum., Final, FT and ZT, less for F '
          'and BD. It assumes independent seeds and no particular '
          'distribution, which matters here: per-seed outcomes are often '
          'bimodal (a seed solves a sub-task or it does not).',
          '- **Correction.** Holm-Bonferroni over one method\'s k comparisons '
          'against the other family.',
          f'- **Mark.** Graded on the LARGEST Holm-adjusted p of those k '
          f'comparisons: {levels}. A mark therefore means "better than all '
          'of them", never "better than one of them".',
          '- **Effect size.** PS, the probability of superiority: the chance '
          'that a random seed of the method is better than a random seed of '
          'the rival, ties counted half. 0.5 is no effect, 1.0 is every seed '
          'better than every rival seed. It equals (1 + Cliff\'s delta) / 2 '
          'and U / (n1 n2) in the direction the row is read.',
          '- **Floor.** The smallest weakest-adjusted p this test can return '
          'for the method\'s seed counts: every comparison perfectly '
          'separated, then Holm. It is the strongest mark the seeds allow '
          '(with ten seeds a side and four rivals it is 3.7e-04, so `***` is '
          'the ceiling and a perfect separation earns nothing more). A floor '
          'at or above 0.05 means no mark was possible: the blank is a '
          'sample-size limit, not evidence of no difference.', '',
          '## What the marks do not control', '',
          '- **Conservative for the claim they make.** "Better than every '
          'rival" is an intersection-union claim, and the largest RAW p '
          'already controls its error rate. Holm on top makes a mark harder '
          'to earn, so read PS where a mark is missing.',
          f'- **No correction across the figure.** It makes {claims} such '
          f'claims ({pairs} one-sided tests) over every row, environment and '
          'method. If no method differed from any other, up to '
          f'{0.05 * claims:.1f} single `*` marks could appear by chance. '
          'Treat an isolated `*` as suggestive, and `**` or `***`, or a mark '
          'that repeats across environments, as the result.',
          '- **Rows are not independent.** Cum., Final, F and ZT are '
          'measured on the same seeds and partly on the same agents.', '']

    md += ['## Marks', '',
           'One line per method with a value. `against` is the rival behind '
           'the weakest adjusted p; `min PS` the method\'s smallest effect '
           'size over all its rivals.', '']
    rows_seen = list(dict.fromkeys(label for label, *_ in stats))
    for label in rows_seen:
        hib = next(h for lab, h, *_ in stats if lab == label)
        md += [f'### {label}' + ('' if hib else ' (lower is better)'), '',
               '| Env | Method | n | mean | mark | weakest Holm p | against '
               '| min PS | floor |',
               '|---|---|---|---|---|---|---|---|---|']
        for lab, _hib, env, cell, tests in stats:
            if lab != label:
                continue
            for m in methods:
                rec = tests.get(m)
                if not rec or not rec['n']:
                    continue
                vals = [v for v in cell[m] if np.isfinite(v)]
                comps = rec['comparisons']
                if comps:
                    worst = max(comps, key=lambda c: c['p_holm'])
                    tail = (f"{rec['mark'] or '--'} | {_p(worst['p_holm'])} | "
                            f"{_label(worst['rival'])} | "
                            f"{min(c['ps'] for c in comps):.2f} | "
                            f"{_p(mark_floor(rec))}")
                else:
                    tail = 'not tested | -- | -- | -- | --'
                md.append(f'| {lp.ENV_TITLES.get(env, env)} | {_label(m)} | '
                          f'{rec["n"]} | {np.mean(vals):.4g} | {tail} |')
        md.append('')

    if len(families) != 2:
        md += ['## Every comparison', '',
               f'Not written: the drawn methods fall into {len(families)} '
               'families, and the comparisons are defined between two.', '']
    else:
        a, b = (FAMILY_NAMES.get(f, f) for f in families)
        md += ['## Every comparison', '',
               f'One line per ({a}, {b}) pair. Each pair is tested twice, '
               f'once per direction, and each direction is Holm-adjusted '
               f'within the comparisons of the method it asks about. PS is '
               f'for {a} being better; {b} being better is 1 - PS.', '']
        for label, hib, env, cell, tests in stats:
            md += [f'### {label} · {lp.ENV_TITLES.get(env, env)}'
                   + ('' if hib else ' (lower is better)'), '',
                   f'| {a} | {b} | n | {a} mean | {b} mean | PS | '
                   f'p {a} better | Holm | p {b} better | Holm |',
                   '|---|---|---|---|---|---|---|---|---|---|']
            for ma in members[families[0]]:
                for mb in members[families[1]]:
                    ra, rb = tests.get(ma), tests.get(mb)
                    if not ra or not rb or not ra['n'] or not rb['n']:
                        continue
                    ca = next((c for c in ra['comparisons']
                               if c['rival'] == mb), None)
                    cb = next((c for c in rb['comparisons']
                               if c['rival'] == ma), None)
                    ps = (ca['ps'] if ca else
                          1.0 - cb['ps'] if cb else None)
                    mean = {m: np.mean([v for v in cell[m] if np.isfinite(v)])
                            for m in (ma, mb)}
                    md.append(
                        f'| {_label(ma)} | {_label(mb)} | '
                        f'{ra["n"]} vs {rb["n"]} | {mean[ma]:.4g} | '
                        f'{mean[mb]:.4g} | '
                        f'{"--" if ps is None else f"{ps:.2f}"} | '
                        f'{_p(ca and ca["p"])} | {_p(ca and ca["p_holm"])} | '
                        f'{_p(cb and cb["p"])} | {_p(cb and cb["p_holm"])} |')
            md.append('')
    path.write_text('\n'.join(md) + '\n')


def main() -> int:
    ap = lp.build_parser()
    ap.description = __doc__
    ap.add_argument('--rows', nargs='*', default=None,
                    choices=list(ROWS),
                    help='which rows to draw, top to bottom (default: '
                         f'{" ".join(DEFAULT_ROWS)}; without final under '
                         '--metric centroid)')
    ap.add_argument('--row-height', type=float, default=1.05,
                    help='height of one panel row in inches (default 1.05)')
    args = ap.parse_args()
    if args.rows is None:
        args.rows = DEFAULT_ROWS_FOR.get(args.metric, DEFAULT_ROWS)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rep = lp.load_report(args)
    rows, envs = lp.metric_table(rep.data, rep.pop_data, rep.ref_data,
                                 rep.per_gen, rep.edges, args)
    posthoc = lp.posthoc_table(rep.root, args.phase, rep.cell_filter,
                               args.results_dir, args.metric)

    # {row: {env: {method: [per-seed values]}}}
    values = {}
    for key in args.rows:
        col = ROWS[key][2]
        values[key] = {}
        for env in envs:
            if col in CURVE_COLUMNS:
                values[key][env] = dict(rows[env][col])
            else:
                values[key][env] = {m: list(v.get(col) or [])
                                    for m, v in posthoc.get(env, {}).items()
                                    if m in rep.data[env]}
    # Drop rows nothing filled, and say so: an absent F is a pass not run,
    # not a zero.
    drawn_rows = []
    for key in args.rows:
        col = ROWS[key][2]
        if any(v for env in envs for v in values[key][env].values()):
            drawn_rows.append(key)
        else:
            print(f'note: row {key!r} has no values in any cell and is not '
                  f'drawn' + (f' -- run {FILLED_BY[col]} and pass '
                              '--results-dir' if col in FILLED_BY else ''))
    if not drawn_rows:
        sys.exit('nothing to draw')

    methods = [m for m in lp.METHOD_ORDER
               if any(m in rep.data[e] for e in envs)]
    methods += sorted({m for e in envs for m in rep.data[e]} - set(methods))

    fs = args.font_size
    plt.rcParams.update({
        'font.size': fs, 'axes.labelsize': fs, 'axes.titlesize': fs + 1,
        'xtick.labelsize': fs - 0.5, 'ytick.labelsize': fs - 0.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
    })
    fig, axes = plt.subplots(len(drawn_rows), len(envs),
                             figsize=(args.width,
                                      args.row_height * len(drawn_rows)),
                             squeeze=False)
    headline = lp.METRIC_TABLE_LABEL[args.metric]
    table, stats = [], []
    for r, key in enumerate(drawn_rows):
        label, hib, _col = ROWS[key]
        label = label or headline
        for c, env in enumerate(envs):
            ax = axes[r][c]
            cell = values[key][env]
            tests, points = draw_panel(ax, cell, methods, key, hib, fs,
                                       labels=c == 0)
            stats.append((label, hib, env, cell, tests))
            table += [(label, env, *p) for p in points]
            ax.set_xlabel(label + (' ↓' if not hib else ''),
                          labelpad=1.5)
            if r == 0:
                ax.set_title(lp.ENV_TITLES.get(env, env), fontweight='bold')
    fig.tight_layout(h_pad=0.6, w_pad=0.8)

    stem = pathlib.Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ('png', 'pdf'):
        out = stem.parent / f'{stem.name}.{ext}'
        fig.savefig(out, dpi=400, bbox_inches='tight')
        written.append(out.name)
    context = (f'`{", ".join(sorted(rep.used_columns))}` &middot; {args.phase}'
               + (f' &middot; {rep.cell_label}' if rep.cell_label else ''))
    # The numbers under every point, for the text of the paper.
    md = [f'# {stem.name}', '', context, '',
          'Mean over seeds [95% percentile-bootstrap CI], n seeds, and the '
          'cross-family significance mark (`*` p<0.05, `**` p<0.01, '
          '`***` p<0.001; the WEAKEST comparison against the other family '
          'survived Holm correction). F and BD are lower-is-better. Every '
          f'test behind the marks is in `{stem.name}_stats.md`.', '',
          '| Row | Env | Method | mean | CI lo | CI hi | n | mark |',
          '|---|---|---|---|---|---|---|---|']
    for label, env, m, mean, lo, hi, n, mark in table:
        md.append(f'| {label} | {lp.ENV_TITLES.get(env, env)} | {_label(m)} | '
                  f'{mean:.3g} | {lo:.3g} | {hi:.3g} | {n} | {mark} |')
    (stem.parent / f'{stem.name}_table.md').write_text('\n'.join(md) + '\n')
    written.append(f'{stem.name}_table.md')
    write_stats(stem.parent / f'{stem.name}_stats.md', stem.name, context,
                stats, methods)
    written.append(f'{stem.name}_stats.md')
    # The per-seed values, finite only, for the cross-family overview.
    (stem.parent / f'{stem.name}_values.json').write_text(json.dumps({
        'metric': args.metric, 'context': context, 'methods': methods,
        'rows': [{'key': key, 'label': ROWS[key][0] or headline,
                  'higher_is_better': ROWS[key][1],
                  'values': {env: {m: [float(x) for x in v if np.isfinite(x)]
                                   for m, v in values[key][env].items()}
                             for env in envs}}
                 for key in drawn_rows]}, indent=1) + '\n')
    written.append(f'{stem.name}_values.json')
    print(f'wrote into {stem.parent}/: ' + ', '.join(written))
    print(f'  rows: {" ".join(drawn_rows)}; figure {args.width} x '
          f'{args.row_height * len(drawn_rows):.2f} in -- include at 1:1')
    return 0


if __name__ == '__main__':
    sys.exit(main())
