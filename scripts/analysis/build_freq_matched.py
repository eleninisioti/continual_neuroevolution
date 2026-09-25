"""Build the switch-interval sweep's MATCHED run trees, as symlinks.

    .venv/bin/python scripts/analysis/build_freq_matched.py

A sub-task sequence is drawn from the trial seed, so trial k faces the same
ten observation offsets under every method and trial j != k faces different
ones. A method comparison at one interval is therefore fair only over the
SAME trials for every arm. The raw trees under runs_freq/ are not like that:
the launchers that ran before the 2-seed cap (2026-09-21) finished seeds 3-5
for some arms at interval 400 and not others, so a per-arm mean over "all
seeds on disk" would put GA's five sub-task draws against TRAC's two.

The seed set is fixed by the DESIGN, not by what happened to finish:

    interval 50    trials 1-5
    interval 400   trials 1-2     (the cap the user chose on 2026-09-21)

Each arm links the trials it has within that set and nothing outside it. An
arm short of the set is printed, so the caption can name it -- it is not
padded, and no other arm is cut down to match it.

Output: projects/iclr_2027/runs_freq_matched/interval<N>/gymnax/continual/
<arm>/<cell>/trial_<k> -> the real trial directory. finish_iclr.sh freq50 /
freq400 read these trees; the raw runs_freq/ is left untouched. The same
layout as runs_kinetix_paper, which is also a symlink tree.

Re-run it whenever a missing trial lands: it rebuilds from scratch.
"""

from __future__ import annotations

import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
RAW = REPO / 'projects/iclr_2027/runs_freq'
OUT = REPO / 'projects/iclr_2027/runs_freq_matched'

# Five seeds at both new lengths since 2026-09-22 (400 was capped at two on
# 09-21): below five, the ring test (one-sided Mann-Whitney U, Holm over the
# five RL arms) cannot reach p < 0.05 even with a clean sweep -- 0.071 at four.
DESIGN = {50: range(1, 6), 400: range(1, 6)}

# THE PAPER'S GA ON MOUNTAINCAR IS `ga_focus_explore`, not the plain gymnax GA
# (scripts/train/ga_focus_mountaincar.py: the plain GA's centroid does not
# track its elite there). The reported tree files those runs under `ga`, so the
# 200 point of this sweep is ga_focus. The 50 and 400 points must be the same
# arm, run by the same script at the other task lengths (OUT_ROOT /
# TASK_INTERVAL), or the GA line would join two different algorithms -- which
# is what the first version of the appendix did. The plain-GA MountainCar runs
# stay in runs_freq/ untouched and are simply not linked.
#
# Since 2026-09-23 that arm is `ga_focus_explore_nox`, the same variant with
# no crossover: the paper dropped crossover, the 200 point was re-run without
# it, and these two lengths follow so the GA line stays one algorithm.
OVERRIDE = {('ga', 'MountainCar_v0_sigma0.1'):
            'projects/iclr_2027/runs_ga_focus_freq/interval{interval}/noise_10task/'
            'gymnax/continual/ga_focus_explore_nox/MountainCar_v0_sigma0.1'}
ARMS = ['ga', 'nes', 'ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2']
CELLS = ['CartPole_v1_sigma1.0', 'Acrobot_v1_sigma1.0', 'MountainCar_v0_sigma0.1']


def main():
    if OUT.exists():
        shutil.rmtree(OUT)                 # symlinks only: never the data
    short = []
    for interval, trials in DESIGN.items():
        src = RAW / f'interval{interval}/gymnax/continual'
        dst = OUT / f'interval{interval}/gymnax/continual'
        linked = 0
        for arm in ARMS:
            for cell in CELLS:
                have = []
                base = OVERRIDE.get((arm, cell))
                base = (REPO / base.format(interval=interval)) if base else src / arm / cell
                for t in trials:
                    d = base / f'trial_{t}'
                    if (d / 'training_metrics.json').exists():
                        link = dst / arm / cell / f'trial_{t}'
                        link.parent.mkdir(parents=True, exist_ok=True)
                        link.symlink_to(d.resolve())
                        have.append(t)
                        linked += 1
                if len(have) < len(trials):
                    missing = sorted(set(trials) - set(have))
                    short.append(f'  interval {interval:>3}  {arm:7s} {cell:26s} '
                                 f'{len(have)}/{len(trials)} (missing trial {missing})')
        print(f'interval {interval:>3}: {linked} trials linked, design trials '
              f'{list(trials)}  -> {dst.relative_to(REPO)}')
    print('arms short of the design:' if short else 'every arm complete.')
    for line in short:
        print(line)


if __name__ == '__main__':
    main()
