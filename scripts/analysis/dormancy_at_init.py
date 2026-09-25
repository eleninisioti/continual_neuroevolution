"""Dormant units of an UNTRAINED network on each gymnax setting's probe states.

    JAX_PLATFORMS=cpu .venv/bin/python scripts/analysis/dormancy_at_init.py

Backs tab:dormant_init (Appendix, app:plasticity_all): the GA and ES hold, for
the whole continual run, about the dormant fraction a freshly initialised
network already has on that environment's states, so their constant level is
set by the environment (MountainCar's observations are almost constant, so
half the ReLU units are silent from the start), not by the method. The
untrained networks are the trainers' own 16x16 ReLU policy at its flax
initialisation, scored with the paper's ReDo threshold on the same `pooled`
probe as `plasticity_checkpoints.py --probe pooled`. The trained columns come
from the saved figure data of `plot_plasticity_overview.py --set main`. Writes
the rows to visuals/final/data/dormancy_at_init.{md,tex} and prints them.
"""
import pathlib, sys
import numpy as np, jax, jax.numpy as jnp
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'scripts/analysis')]
import plasticity_checkpoints as pc                          # noqa: E402
from source.algorithms.rl import redo                         # noqa: E402
from source.envs.gymnax_classic import make_gymnax_env, task_noise_vectors  # noqa: E402
from source.algorithms.networks import create_policy_network  # noqa: E402

DATA = REPO / 'projects/iclr_2027/paper/visuals/final/data'
SIGMA = {'CartPole-v1': 1.0, 'Acrobot-v1': 1.0, 'MountainCar-v0': 0.1}
COLUMN = {'CartPole-v1': 'cartpole', 'Acrobot-v1': 'acrobot', 'MountainCar-v0': 'mountaincar'}
NUM_INIT, NUM_PROBE, NUM_TASKS = 50, 512, 10


def untrained_fraction(env, offsets, cache):
    base = np.asarray(pc._base_rollout(env, NUM_PROBE, 0, cache))
    rows = np.concatenate([offsets, np.ones((len(offsets), 1))], 1)
    probe = jnp.asarray(pc.pooled_probe(lambda t: base + offsets[t], rows, NUM_PROBE))
    e, p = make_gymnax_env(env)
    fr = []
    for s in range(NUM_INIT):
        _, params = create_policy_network(jax.random.key(s), base.shape[-1],
                                          int(e.action_space(p).n), (16, 16))
        acts = redo.hidden_activations(params, probe, 2)
        sc = jnp.concatenate(redo.score_layers(acts, redo.CRITERION_MAGNITUDE))
        fr.append(float((sc <= redo.DEFAULT_TAU).mean()))
    return np.mean(fr), np.std(fr)


def main():
    saved = np.load(DATA / 'plasticity_main_continual.npz', allow_pickle=True)
    cache, md, tex = {}, [], []
    hdr = ('Setting', 'Untrained', 'GA', 'ES', 'PPO, task 1', 'PPO, task 20')
    md.append('| ' + ' | '.join(hdr) + ' |'); md.append('|' + '---|' * len(hdr))
    for env, sig in SIGMA.items():
        dim = np.asarray(pc._base_rollout(env, NUM_PROBE, 0, cache)).shape[-1]
        for change, offs in (('noise', np.stack([np.asarray(v) for v in
                                                 task_noise_vectors(0, NUM_TASKS, dim, sig)])),
                             ('actions', np.zeros((1, dim)))):
            init, sd = untrained_fraction(env, offs, cache)
            col = f'{COLUMN[env]}-{change}'
            def arm(a): return np.asarray(saved[f'{col}|dormant_pooled|{a}'], float)
            ga, es, ppo = np.nanmean(arm('ga')), np.nanmean(arm('es')), arm('ppo')
            name = f"{env.split('-')[0]}, {'action reversal' if change == 'actions' else 'noise'}"
            vals = (f'{init:.2f} $\\pm$ {sd:.2f}', f'{ga:.2f}', f'{es:.2f}',
                    f'{np.nanmean(ppo[:, 0]):.2f}', f'{np.nanmean(ppo[:, -1]):.2f}')
            md.append(f'| {name} | ' + ' | '.join(v.replace("$\\pm$", "±") for v in vals) + ' |')
            tex.append(f'{name} & ' + ' & '.join(vals) + ' \\\\')
    (DATA / 'dormancy_at_init.md').write_text('\n'.join(md) + '\n')
    (DATA / 'dormancy_at_init.tex').write_text('\n'.join(tex) + '\n')
    print('\n'.join(md)); print(); print('\n'.join(tex))


if __name__ == '__main__':
    main()
