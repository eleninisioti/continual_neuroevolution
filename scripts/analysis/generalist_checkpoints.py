"""Generalist or switching specialist? Found / Held / Retention and the phase
outcomes, from `evaluation.json`, for any continual tree.

    .venv/bin/python scripts/analysis/generalist_checkpoints.py \\
        projects/iclr_2027/runs_centroid/minigrid --cells MiniGrid_8x8_16x16 \\
        --threshold 0.8 --agent centroid --methods ga es nes ... \\
        --out projects/iclr_2027/paper/minigrid/rooms/generalist_checkpoints_centroid

Writes `<out>.md` (the tables) and `<out>.json` (per-trial returns).

The sub-task-resolution analogue of `scripts/analysis/actions_generalist.py`,
which re-rolls the gymnax action-reversal agents under both action orders.
This one re-rolls nothing: `source/studies/evaluate_continual.py` already
scores every saved end-of-sub-task agent on its OWN sub-task (`returns`), on
the PREVIOUS one (`prev_returns`) and on the NEXT one
(`zero_shot_next_returns`), and those three are what the classification needs.

Per checkpoint t (the agent saved at the end of sub-task phase t):

    shown    it clears the threshold on the sub-task it was just trained on
    other    it clears it on the PREVIOUS sub-task -- the one it was trained
             on before this phase, which is what it stands to have forgotten

    generalist   shown and other        -- one agent good at both
    switching    shown only             -- the specialist that relearns each
                                           sub-task and drops the last one
    stuck        other only             -- did not learn what it was shown,
                                           still holds the previous sub-task
    neither      neither

Checkpoint 0 has no previous sub-task and is left out of the outcome table
(19 post-switch checkpoints on a 20-phase run). The Found / Held / Retention
table counts, per trial, whether ANY checkpoint is a generalist, whether the
LAST one is, and the fraction after the first discovery that still are; the
last checkpoint's "other" score comes from `prev_returns`.

`prev_returns` was added to the evaluator on 2026-09-10. On an older
`evaluation.json` without it, this falls back to `zero_shot_next_returns` of
the SAME checkpoint -- valid only when the run alternates between exactly two
sub-tasks, so that the next sub-task IS the previous one -- and says so; the
last checkpoint is then unclassifiable and the tables say how many were
dropped. Re-run the evaluator (idempotent: it re-scores what is stale) to
close the gap.

THE THRESHOLD is the one thing this cannot read off the run. The gymnax
environments have solved thresholds in `source/envs/registry.threshold_for`;
MiniGrid and the mjx bodies have none, so `--threshold` is required there.
On MiniGrid the paper uses 0.8: the 16x16 specialist scores 0.95 / 0.97 on
the two rooms and the 8x8 specialist 0.73 on the 16x16 one, so 0.8 on BOTH
is what neither specialist reaches and a generalist must.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from source.envs.registry import threshold_for            # noqa: E402

SOURCES = {'centroid': ('centroid', 'final'),
           'elite': ('finalgen', 'final')}
OUTCOMES = ('generalist', 'switching', 'stuck', 'neither')


def per_checkpoint(entries):
    """`(own, prev, fallback)`: mean return per checkpoint on the shown and on
    the previous sub-task; `fallback` says the previous came from the next."""
    entries = sorted(entries, key=lambda e: e['task_idx'])
    own = np.array([np.mean(e['returns']) for e in entries])
    prev = np.full(len(entries), np.nan)
    fallback = False
    for i, e in enumerate(entries):
        if e.get('prev_returns'):
            prev[i] = np.mean(e['prev_returns'])
        elif i > 0 and e.get('zero_shot_next_returns'):
            prev[i] = np.mean(e['zero_shot_next_returns'])
            fallback = True
    return own, prev, fallback


def classify(own, prev, thr):
    shown = own >= thr
    other = prev >= thr
    cls = np.where(shown & other, 'generalist',
                   np.where(shown, 'switching', np.where(other, 'stuck', 'neither')))
    cls = cls.astype(object)
    cls[np.isnan(prev)] = None
    return cls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='run tree, e.g. projects/iclr_2027/runs_centroid/minigrid')
    ap.add_argument('--phase', default='continual')
    ap.add_argument('--cells', nargs='+', required=True)
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--agent', default='centroid', choices=sorted(SOURCES))
    ap.add_argument('--threshold', type=float, default=None,
                    help='solved threshold; default is the registry\'s for '
                         'the run\'s env, and required where it has none')
    ap.add_argument('--out', required=True, help='output stem, no extension')
    args = ap.parse_args()

    root = pathlib.Path(args.root) / args.phase
    rows, detail, notes = [], {}, set()
    for cell in args.cells:
        methods = args.methods or sorted(
            p.name for p in root.iterdir() if (p / cell).is_dir())
        for m in methods:
            cell_dir = root / m / cell
            if not cell_dir.is_dir():
                continue
            found = held = n = 0
            ret, curves, outcomes, dropped = [], [], [], 0
            thr = args.threshold
            for trial_dir in sorted(cell_dir.glob('trial_*')):
                path = trial_dir / 'evaluation.json'
                if not path.exists():
                    continue
                blob = json.loads(path.read_text())
                if thr is None:
                    thr = threshold_for(blob['env'])
                    if thr is None:
                        sys.exit(f'{blob["env"]} has no solved threshold in the '
                                 'registry; pass --threshold')
                src = next((s for s in SOURCES[args.agent]
                            if s in blob['agent_sources']), None)
                if src is None:
                    notes.add(f'{m}/{cell}: no {args.agent} agent in '
                              f'{path}; evaluate again')
                    continue
                own, prev, fallback = per_checkpoint(
                    [e for e in blob['per_task'] if e['source'] == src])
                if fallback:
                    notes.add(f'{m}/{cell}: `prev_returns` missing, the previous '
                              'sub-task was read from `zero_shot_next_returns` '
                              '(two-sub-task alternation assumed)')
                cls = classify(own, prev, thr)
                g = (cls == 'generalist')
                n += 1
                curves.append(g.astype(float))
                post = [c for c in cls[1:] if c is not None]
                dropped += int(sum(c is None for c in cls[1:]))
                outcomes.extend(post)
                if g.any():
                    found += 1
                    first = int(np.argmax(g))
                    ret.append(float(g[first:].mean()))
                    held += int(bool(g[-1]))
                detail[f'{cell}/{m}/{trial_dir.name}'] = {
                    'shown': np.round(own, 3).tolist(),
                    'previous': [None if np.isnan(v) else round(float(v), 3)
                                 for v in prev],
                    'outcome': [None if c is None else str(c) for c in cls],
                }
            if n:
                allc = np.asarray(outcomes, dtype=object)
                frac = {c: float(np.mean(allc == c)) if len(allc) else float('nan')
                        for c in OUTCOMES}
                rows.append(dict(cell=cell, method=m, found=found, held=held, n=n,
                                 retention=float(np.mean(ret)) if ret else 0.0,
                                 curve=np.mean(curves, 0), frac=frac,
                                 dropped=dropped, thr=thr))
                print(f'{cell:22s} {m:13s} found {found:2d}/{n} held {held:2d}/{n} '
                      f'retention {rows[-1]["retention"]:.2f}  '
                      + ' '.join(f'{c[:4]} {frac[c]:.2f}' for c in OUTCOMES))
    if not rows:
        sys.exit('no evaluation.json found; run source.studies.evaluate_continual first')

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    thr_text = ', '.join(sorted({f'{r["thr"]:g}' for r in rows}))
    lines = [f'# {out.name}', '',
             f'Saved agent: `{args.agent}` ({" / ".join(SOURCES[args.agent])}). '
             f'Solved threshold: {thr_text}. A checkpoint is a GENERALIST when '
             'the agent saved at the end of a sub-task phase clears the '
             'threshold on that sub-task AND on the previous one.', '',
             '| Cell | Method | Found | Held | Retention |', '|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["cell"]} | {r["method"]} | {r["found"]}/{r["n"]} | '
                     f'{r["held"]}/{r["n"]} | {r["retention"]:.2f} |')
    lines += ['', 'Found = any of the checkpoints is a generalist; Held = the '
              'last one is; Retention = fraction of checkpoints after the first '
              'discovery that still are.', '',
              'Phase outcomes, fraction of the post-switch checkpoints pooled '
              'over trials (`learned shown` = generalist + switching):', '',
              '| Cell | Method | Generalist | Switching | Stuck | Neither | '
              'Learned shown |', '|---|---|---|---|---|---|---|']
    for r in rows:
        f = r['frac']
        lines.append(f'| {r["cell"]} | {r["method"]} | {f["generalist"]:.2f} | '
                     f'{f["switching"]:.2f} | {f["stuck"]:.2f} | {f["neither"]:.2f} '
                     f'| {f["generalist"] + f["switching"]:.2f} |')
    lines += ['', 'Fraction of trials whose checkpoint t is a generalist '
              '(t = 0..T-1):', '']
    for r in rows:
        lines.append(f'- {r["cell"]} {r["method"]}: '
                     + ' '.join(f'{c:.1f}' for c in r['curve']))
    if any(r['dropped'] for r in rows):
        lines += ['', 'Checkpoints left unclassified (no previous-sub-task '
                  'score): ' + ', '.join(f'{r["method"]} {r["dropped"]}'
                                         for r in rows if r['dropped'])]
    if notes:
        lines += ['', 'Notes:'] + [f'- {n}' for n in sorted(notes)]
    out.with_name(out.name + '.md').write_text('\n'.join(lines) + '\n')
    out.with_name(out.name + '.json').write_text(json.dumps(detail))
    print(f'wrote {out}.md, {out}.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
