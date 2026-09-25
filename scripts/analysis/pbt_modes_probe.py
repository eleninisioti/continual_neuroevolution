"""PBT-PPO mode probe: selection or hyperparameter exploration?

    .venv/bin/python scripts/analysis/pbt_modes_probe.py
    -> results/pbt_modes/pbt_modes_probe.md

Lines up, per stage of training (tasks 1-3, 4-9, 10-14, 15-19), the paper's
PBT-PPO (mode full) and PPO on the noise two-task cells of CartPole and
Acrobot against the two ablation arms of queue_pbt_modes_probe.sh,
pbt_weights (exploit only) and pbt_hp (explore only), on every measure the
overlap probe records around the end-of-task centroid: the share of
perturbations at relative radius 0.03 and 0.1 that keep the task (width), the
shared basin (random moves of the own step size that solve both tasks), the
rescaled return on the own task, and the relative step to the next checkpoint.
Sources: results/overlap_probe + results/overlap_first_half (paper arms) and
results/pbt_modes (the new arms; `child_survival.py overlap --panels
cartpole_noise2_modes acrobot_noise2_modes --arms pbt_weights pbt_hp
--out pbt_modes`). The action-change width of curvature_width.py is added per
task when a pass exists for the root (paper: paper/gymnax/noise/2task/results/
centroid; probe: projects/iclr_2027/probe_pbt_modes/gymnax/results/centroid).
"""
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/analysis'))
sys.path.insert(0, str(REPO / 'scripts'))
import child_survival as cs                                 # noqa: E402

STAGES = [('1-3', range(0, 3)), ('4-9', range(3, 9)), ('10-14', range(9, 14)), ('15-19', range(14, 19))]
SOURCES = {'pbt': ['overlap_probe', 'overlap_first_half'], 'ppo': ['overlap_probe', 'overlap_first_half'],
           'pbt_weights': ['pbt_modes'], 'pbt_hp': ['pbt_modes'], 'pbt_hp_member': ['pbt_modes']}
PANELS = [('CartPole, noise 0.5', 'cartpole_noise2', 'CartPole_v1_sigma0.5'),
          ('Acrobot, noise 0.5', 'acrobot_noise2', 'Acrobot_v1_sigma0.5')]
WIDTH_PASS = {'pbt': 'paper/gymnax/noise/2task/results/centroid/curvature_width.json',
              'ppo': 'paper/gymnax/noise/2task/results/centroid/curvature_width.json',
              'pbt_weights': 'probe_pbt_modes/gymnax/results/centroid/curvature_width.json',
              'pbt_hp': 'probe_pbt_modes/gymnax/results/centroid/curvature_width.json',
              'pbt_hp_member': 'probe_pbt_modes/gymnax/results/centroid/curvature_width.json'}
OUT = REPO / 'results' / 'pbt_modes' / 'pbt_modes_probe.md'


def rows_of(arm, panel):
    name = panel if arm in ('pbt', 'ppo') else panel + '_modes'
    out = []
    for d in SOURCES[arm]:
        path = REPO / 'results' / d / 'raw' / f'{name}__{arm}.json'
        if path.exists():
            out += json.loads(path.read_text())
    return out


def action_change(arm, cell):
    path = REPO / 'projects/iclr_2027' / WIDTH_PASS[arm]
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    arm_dir = 'pbt' if arm == 'pbt' else arm
    trials = blob['cells'].get(cell, {}).get(arm_dir, {}).get('trials', {})
    vals = [t['width_0.1'] for t in trials.values() if 'width_0.1' in t]
    return np.array(vals, float) if vals else None


def main():
    md = ['# PBT-PPO mode probe', '', 'See the docstring of `scripts/analysis/pbt_modes_probe.py`. '
          'Mean over the (trial, checkpoint) pairs of a stage; keep = within 0.05 x span of the '
          'parent\'s re-score; both = rescaled >= 0.8 on both tasks; solved = parent >= 0.8 on its '
          'own task (the width and shared-basin columns use solved checkpoints only).', '']
    for title, panel, cell in PANELS:
        rows = {arm: rows_of(arm, panel) for arm in SOURCES}
        rows = {a: r for a, r in rows.items() if r}
        best = max(np.mean([np.asarray(r['scores']['copies:0'])[:, 0].mean() for r in rs]) for rs in rows.values())
        md += [f'## {title}', '', '| Arm | stage | n (solved/all) | keep @0.03 | keep @0.1 | shared basin | '
               'own-task return | step | action change @0.1 |', '|---|---|---|---|---|---|---|---|---|']
        for arm, rs in rows.items():
            st = {(r['trial'], r['phase']): (cs.overlap_stats(r, best, 0.05, 0.8), r['step_rel']) for r in rs}
            ac = action_change(arm, cell)
            for stage, ph in STAGES:
                sel = [v for (t, p), v in st.items() if p in ph]
                if not sel:
                    continue
                solved = [v for v in sel if v[0]['parent_A'] >= 0.8]
                f = lambda key, sub: np.mean([v[0][key]['keep'] for v in solved]) if solved else np.nan      # noqa: E731
                both = np.mean([v[0]['matched:1']['both'] for v in solved]) if solved else np.nan
                acs = f'{np.nanmean(ac[:, list(ph)]):.3f}' if ac is not None and ac.shape[1] > max(ph) else '-'
                md.append(f'| {arm} | {stage} | {len(solved)}/{len(sel)} | {f("fixed:0.03", 0):.2f} | '
                          f'{f("fixed:0.1", 0):.2f} | {both:.2f} | {np.mean([v[0]["parent_A"] for v in sel]):.2f} | '
                          f'{np.mean([v[1] for v in sel]):.3f} | {acs} |')
        md.append('')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
