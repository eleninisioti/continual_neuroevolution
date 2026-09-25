"""Give a shared-runner tree (MiniGrid, brax, kinetix) the gymnax column names.

    .venv/bin/python scripts/analysis/migrate_shared_runner_columns.py \\
        projects/iclr_2027/runs_centroid/minigrid [--dry-run]

The suite-generic runners (`source/studies/generalists/train_{nes,ppo}.py`)
record a different vocabulary from the gymnax trainers, and the paper's figure
scripts (`scripts/make_lineplot.py`, `scripts/make_plasticity_figure.py`,
`scripts/verify_runs.py`) resolve the gymnax one. Rather than teach every
script a second spelling of each column, this pass rewrites the runs ONCE, in
place, so one tree looks like the other. It is idempotent: a run that already
carries the aliases is left alone, and the original columns are kept beside
the aliases rather than replaced, so nothing is lost.

## What it does, per run

1. **The incumbent / centroid swap** (NE runs made before commit e08bd10,
   2026-09-10). Those runs logged the INCUMBENT -- what the search hands back,
   `archive[0]` on the GA -- under the `centroid_*` prefix and the population's
   weight mean under `popmean_*`. The runner has since been corrected to the
   gymnax convention (`centroid_*` IS the weight mean, the incumbent gets its
   own prefix), and the checkpoints were always right. A run with `popmean_*`
   columns and no `incumbent_*` ones is renamed:

        centroid_<agg>  ->  incumbent_<agg>
        popmean_<agg>   ->  centroid_<agg>

   On ES/NES the two are the same point and nothing observable changes; on
   the GA and DNS this is the difference between the elite and the archive
   mean, and it is the whole reason the centroid figures exist.

2. **The gymnax aliases**, every one the score of the ACTIVE sub-task -- the
   record's `task` picks the `<prefix>_task<k>` column -- because that is what
   the gymnax columns are, and it is what makes the curve dip at a switch:

        NE   elite_eval_fitness  = incumbent_task<task>
                  what the search hands back, re-scored on fresh keys at the
                  reported episode count. The gymnax column is the best
                  MEMBER of the generation re-scored; on the GA and DNS the
                  incumbent is exactly that member, on ES/NES it is the
                  distribution mean, which is also what those searches hand
                  back. NOT `train_fitness_max`, which is the winner's curse.
             centroid_fitness    = centroid_task<task>
             best_fitness        = train_fitness_max
             mean_fitness        = train_fitness_mean
             best_overall        = running max of best_fitness
        RL   mean_reward         = centroid_task<task>
                  the single policy, fresh keys, argmax actions -- the same
                  ruler the NE arms are scored with (train_ppo builds it from
                  the same `make_scoring_fn`).
             best_reward         = running max of mean_reward
             policy_dormant_fraction_probe = dormant_fraction
             policy_churn_action = rl_churn      (argmax disagreement)
             policy_churn        = rl_churn_ce   (the cross-entropy C-CHAIN
                                                  regularises)

   The `<agg>` columns (`generalist`, `mean_over_tasks`, `task0`, `task1`) are
   kept: `--metric incumbent --aggregate generalist` still reads them.

3. **`num_timesteps` on the RL runs**, `num_updates x num_envs x num_steps`.
   The gymnax RL trainers record their budget in steps under that name and
   every reader (`verify_runs.py`, `make_lineplot.budget`) tells an RL run
   from an NE one by its presence. Without it a shared-runner PPO run is read
   as `generations x pop_size x num_evals x episode_length` = 20x its real
   budget, and the tree fails verification for a mismatch it does not have.

Each rewritten run gets `column_naming: "gymnax_aliases_v1"` in its
`results.json` (and `config.json`, which the CLI writes as a copy of the
config) so a reader can tell a migrated run from a native one.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

MARKER = 'gymnax_aliases_v1'
AGGS = ('generalist', 'mean_over_tasks')


def _atomic_write(path: pathlib.Path, text: str):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def _prefixed(record: dict, prefix: str):
    return [k for k in record if k.startswith(prefix + '_')]


def migrate_records(records: list, is_rl: bool):
    """Rewrite the record list in place. Returns a summary dict."""
    if not records:
        return {'records': 0}
    first = records[0]
    swapped = False
    # An RL run WITH a population (PBT) writes the NE prefixes -- `incumbent_*`
    # the elite, `centroid_*` the weight mean -- so its aliases are the NE
    # ones AND `mean_reward`, which reads the elite: the curve every other
    # RL arm's `mean_reward` is (the deployed policy, fresh keys, argmax).
    populated = bool(is_rl and _prefixed(first, 'incumbent'))
    if not is_rl and _prefixed(first, 'popmean') and not _prefixed(first, 'incumbent'):
        swapped = True
        for r in records:
            for k in _prefixed(r, 'centroid'):
                r['incumbent_' + k[len('centroid_'):]] = r.pop(k)
            for k in _prefixed(r, 'popmean'):
                r['centroid_' + k[len('popmean_'):]] = r.pop(k)
    running = float('-inf')
    for r in records:
        task = int(r['task'])
        if is_rl:
            src = 'incumbent' if populated else 'centroid'
            r['mean_reward'] = float(r[f'{src}_task{task}'])
            if populated:
                r['elite_eval_fitness'] = r['mean_reward']
                r['centroid_fitness'] = float(r[f'centroid_task{task}'])
                r['best_fitness'] = float(r['train_fitness_max'])
                r['mean_fitness'] = float(r['train_fitness_mean'])
            running = max(running, r['mean_reward'])
            r['best_reward'] = running
            if 'dormant_fraction' in r:
                r['policy_dormant_fraction_probe'] = r['dormant_fraction']
            if 'rl_churn' in r:
                r['policy_churn_action'] = r['rl_churn']
            if 'rl_churn_ce' in r:
                r['policy_churn'] = r['rl_churn_ce']
        else:
            # The best-performing agent where the run logged it (`elite_*`,
            # since 2026-09-13); older runs fall back to the incumbent, which
            # on NES/OpenES is the distribution mean.
            r['elite_eval_fitness'] = float(r[f'elite_task{task}']
                                            if f'elite_task{task}' in r
                                            else r[f'incumbent_task{task}'])
            r['centroid_fitness'] = float(r[f'centroid_task{task}'])
            r['best_fitness'] = float(r['train_fitness_max'])
            r['mean_fitness'] = float(r['train_fitness_mean'])
            running = max(running, r['best_fitness'])
            r['best_overall'] = running
    return {'records': len(records), 'swapped_incumbent_centroid': swapped}


def migrate_run(trial_dir: pathlib.Path, dry_run: bool):
    results_path = trial_dir / 'results.json'
    config_path = trial_dir / 'config.json'
    metrics_path = trial_dir / 'training_metrics.json'
    if not (results_path.exists() and metrics_path.exists()):
        return 'incomplete'
    results = json.loads(results_path.read_text())
    cfg = results.get('config') or results
    if cfg.get('column_naming') == MARKER:
        return 'already'
    is_rl = 'num_envs' in cfg and 'num_steps' in cfg and 'num_epochs' in cfg
    records = json.loads(metrics_path.read_text())
    # A run made by the per-suite gymnax trainers already speaks the gymnax
    # vocabulary and has no `centroid_task*` to alias from; leave it alone,
    # so this can be run over a gymnax tree that mixes the two (a PBT arm
    # from the shared runner beside the gymnax trainers' arms).
    if records and ('mean_reward' in records[0] or 'elite_eval_fitness' in records[0]) \
            and 'centroid_task0' not in records[0]:
        return 'native'
    summary = migrate_records(records, is_rl)
    if is_rl:
        # `num_generations` is the update count on these runners.
        cfg['num_timesteps'] = (int(cfg['num_generations']) * int(cfg['num_envs'])
                                * int(cfg['num_steps']))
    cfg['column_naming'] = MARKER
    if dry_run:
        return f"would migrate ({'rl' if is_rl else 'ne'}, {summary})"
    _atomic_write(metrics_path, json.dumps(records, indent=2))
    _atomic_write(results_path, json.dumps(results, indent=2))
    if config_path.exists():
        c = json.loads(config_path.read_text())
        if is_rl:
            c['num_timesteps'] = cfg['num_timesteps']
        c['column_naming'] = MARKER
        _atomic_write(config_path, json.dumps(c, indent=2))
    return f"migrated ({'rl' if is_rl else 'ne'}, {summary})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='a run tree, e.g. projects/iclr_2027/runs_centroid/minigrid')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    counts = {}
    for trial_dir in sorted(root.glob('*/*/*/trial_*')):
        if not trial_dir.is_dir():
            continue
        status = migrate_run(trial_dir, args.dry_run)
        key = status.split(' (')[0]
        counts[key] = counts.get(key, 0) + 1
        if key not in ('already',):
            print(f'{trial_dir.relative_to(root)}: {status}')
    print('summary:', counts)
    return 0


if __name__ == '__main__':
    sys.exit(main())
