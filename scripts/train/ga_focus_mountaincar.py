"""One `ga_focus_explore` run on MountainCar, for the paper's gymnax GA.

    .venv/bin/python scripts/train/ga_focus_mountaincar.py <family> <trial> <gpu>

The plain GA's centroid does not track its elite on MountainCar (-500 in half
the stationary seeds); `ga_focus_explore` (source/studies/kinetix/settings.py
NE_ARMS, unchanged) fixed that on Kinetix and, in a 3-seed probe, on
stationary MountainCar. CartPole and Acrobot keep the plain GA.

The arm exists only in the shared runner (`train_nes.run_nes`), not in the
gymnax GA trainer the rest of the gymnax tree was made with. Each family's
sub-task arguments mirror the gymnax PBT block of scripts/train/run_experiments.sh
(the other shared-runner arm on these trees): `schedule switch`, `num_tasks`
= the family's task period, the cell's offset width, and `task_mod` physics or
actions. PBT's saved sequences were checked against the old GA's for four
families x trials 1, 2, 7 (offsets, gravity multipliers, flips identical);
`run_nes` calls the same `suite.task_vectors`. Budget: 4000 generations x 512
x 3 evaluations x 500 steps = 3.072e9 steps, a switch every 200 generations,
10 report episodes; stationary 600 generations. Seeds follow the gymnax GA
trainers: 42 + trial continual, 41 + trial stationary.

Output: projects/iclr_2027/runs_ga_focus_mountaincar/<family>/gymnax/
<continual|noncontinual>/ga_focus_explore/<cell>/trial_<k>. The paper trees
are switched over to these only after the runs are checked.
"""
import os
import sys

FAMILIES = {
    # family: (cell, num_tasks, noise_range, task_options); None = stationary
    'stationary':     ('MountainCar_v0', None, 0.0, None),
    'noise_10task':   ('MountainCar_v0_sigma0.1', 10, 0.1, None),
    'physics_10task': ('MountainCar_v0_sigma1.0', 10, 1.0,
                       {'task_mod': 'physics', 'physics_param': 'gravity',
                        'physics_mult_range': '0.6666666666666666,1.5'}),
    'actions_2task':  ('MountainCar_v0_sigma1.0', 10, 1.0, {'task_mod': 'actions'}),
    'noise_2task':    ('MountainCar_v0_sigma0.05', 2, 0.05, None),
    'physics_2task':  ('MountainCar_v0_sigma1.0', 2, 1.0,
                       {'task_mod': 'physics', 'physics_param': 'gravity',
                        'physics_mult_range': '1.5,1.5'}),
}
ARM = os.environ.get('ARM', 'ga_focus_explore')  # any NE_ARMS entry
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Both overridable for the switch-interval appendix (Appendix E), which
# re-runs the noise family at other task lengths into its own tree; the
# defaults are the paper's runs, unchanged.
ROOT = os.environ.get('OUT_ROOT', os.path.join(REPO, 'projects/iclr_2027/runs_ga_focus_mountaincar'))
TASK_INTERVAL = int(os.environ.get('TASK_INTERVAL', '200'))  # generations per task


def main():
    family, trial, gpu = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    sys.path.insert(0, REPO)
    from source.studies.kinetix import settings as S
    from source.studies.generalists.train_nes import run_nes

    cell, num_tasks, noise_range, task_options = FAMILIES[family]
    stationary = num_tasks is None
    out = os.path.join(ROOT, family, 'gymnax',
                       'noncontinual' if stationary else 'continual', ARM, cell,
                       f'trial_{trial}')
    if os.path.exists(os.path.join(out, 'training_metrics.json')):
        print(f'done already: {out}')
        return
    os.makedirs(out, exist_ok=True)
    schedule = dict(schedule='task0', num_tasks=1, num_generations=600,
                    task_interval=60, seed=41 + trial) if stationary else \
        dict(schedule='switch', num_tasks=num_tasks, num_generations=4000,
             task_interval=TASK_INTERVAL, seed=42 + trial)
    run_nes(env_name='MountainCar-v0', noise_range=noise_range,
            task_options=task_options, first_task_clean=True, pop_size=512,
            num_evals=3, episode_length=500, eval_episodes=10, trial=trial,
            output_dir=out, **schedule, **S.NE_ARMS[ARM])


if __name__ == '__main__':
    main()
