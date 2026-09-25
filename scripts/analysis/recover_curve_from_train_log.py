"""Rebuild a run's reward curve from its `train.log`, for a trial that finished
training but never wrote its outputs.

Long ant-shape MJX PPO runs on the home server hang after their last update
and write nothing (memory: mjx-ppo-probe-hung-at-finalise), so the stationary
cheetah TRAC re-run at the ant's PPO shape (runs_mjx_antshape, 2026-09-16) has
eight trials that reached update 47999 with only a train.log. The log prints
every 300 updates the same evaluation `training_metrics.json` records every
150 (`t<i>` == `centroid_task<i>`, and `t0` == `mean_reward` on a stationary
run -- checked on runs_mjx), so the recovered curve is the recorded one at half
the resolution. Dormancy, churn and checkpoints are not in the log: this tree
feeds reward curves only.

    .venv/bin/python scripts/analysis/recover_curve_from_train_log.py \\
        runs_mjx_antshape/mjx/noncontinual/trac/cheetah \\
        runs_mjx_antshape_fromlog/mjx/noncontinual/trac/cheetah \\
        --config '{"num_envs": 512, ...}'

The output goes to a SEPARATE tree, never beside the log: ship_home.sh skips a
trial that has a training_metrics.json at home, so a recovered file in the
source tree would stop CLUSTER's complete run of the same trial from arriving.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

PROJECT = pathlib.Path(__file__).resolve().parent.parent.parent / 'projects/iclr_2027'
LINE = re.compile(r'^\s*update\s+(\d+)\s+task=\s*(\d+)\s+H=\s*(\S+)\s+(.*?)\s+gen\'ist=\s*(\S+)')
TASK = re.compile(r't(\d+)=\s*(\S+)')


def parse(log: pathlib.Path):
    records, best = [], -float('inf')
    for line in log.read_text().splitlines():
        m = LINE.match(line)
        if not m:
            continue
        update, task, entropy, tasks, generalist = m.groups()
        rec = {'generation': int(update), 'task': int(task), 'entropy': float(entropy)}
        for i, v in TASK.findall(tasks):
            rec[f'centroid_task{i}'] = float(v)
        rec['centroid_generalist'] = float(generalist)
        rec['mean_reward'] = rec[f'centroid_task{task}']
        best = max(best, rec['mean_reward'])
        rec['best_reward'] = best
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='cell directory under projects/iclr_2027 holding trial_*/train.log')
    ap.add_argument('dst', help='cell directory to write, under projects/iclr_2027')
    ap.add_argument('--config', required=True,
                    help='JSON object: the run settings, written as each trial\'s config.json')
    ap.add_argument('--last-update', type=int, required=True,
                    help='a trial whose log does not reach this update is skipped')
    args = ap.parse_args()
    cfg = json.loads(args.config)
    src, dst = PROJECT / args.src, PROJECT / args.dst
    for trial in sorted(src.glob('trial_*')):
        records = parse(trial / 'train.log')
        if not records or records[-1]['generation'] != args.last_update:
            print(f'{trial.name}: skipped, log ends at '
                  f'{records[-1]["generation"] if records else "nothing"}')
            continue
        out = dst / trial.name
        out.mkdir(parents=True, exist_ok=True)
        (out / 'training_metrics.json').write_text(json.dumps(records))
        (out / 'config.json').write_text(json.dumps(
            {**cfg, 'trial': int(trial.name.split('_')[1]),
             'recovered_from': str((trial / 'train.log').relative_to(PROJECT))}, indent=2))
        print(f'{trial.name}: {len(records)} records, last {records[-1]["mean_reward"]:.0f}')


if __name__ == '__main__':
    main()
