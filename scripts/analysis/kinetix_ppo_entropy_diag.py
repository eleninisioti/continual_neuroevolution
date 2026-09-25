"""PPO entropy collapse on Kinetix, pixels vs transformer: per saved policy
(end of each level), entropy / logit gap / tanh saturation / output weight norm
on the NEXT level's observations. See docs/kinetix_transformer_tried_20260924.md, section 4.

    PYTHONPATH=. .venv/bin/python scripts/analysis/kinetix_ppo_entropy_diag.py
"""
import json, sys, numpy as np, jax, jax.numpy as jnp
from source.envs import kinetix as K
from source.algorithms.networks import unflatten_params
P='projects/iclr_2027'
OFF=[0,3,6,9,12,14]; DIMS=[3,3,3,3,2,2]
def diag(obs_opt, ckpt):
    env, _ = K.build_env('Kinetix20', 128, obs_opt)
    pol, tmpl, _ = K.build_policy(jax.random.key(0), env.obs_dim, 16, (128,)*5)
    W = np.load(ckpt)['final']
    obs = {k: K.random_policy_observations(env, None, jnp.array([float(k)]), 256, seed=k) for k in range(20)}
    for k in (0, 2, 4, 9, 14, 19):
        p = unflatten_params(jnp.asarray(W[k]), tmpl)
        o = obs[min(k+1, 19)]
        lg = pol.apply(p, o)
        H_ = sum(-(jax.nn.softmax(lg[:, a:a+d]) * jax.nn.log_softmax(lg[:, a:a+d])).sum(-1) for a, d in zip(OFF, DIMS)).mean()
        spread = np.mean([float((lg[:, a:a+d].max(-1) - lg[:, a:a+d].min(-1)).mean()) for a, d in zip(OFF, DIMS)])
        hs = pol.hidden_activations(p, o)
        dense = [h for h in hs if h.ndim == 2][-5:]
        sat = [float((jnp.abs(h) > 0.99).mean()) for h in dense]
        wout = float(jnp.linalg.norm(p['params']['Dense_5']['kernel']))
        print(f'  after level {k+1:2d}: entropy {float(H_):.2f}  logit gap {spread:6.2f}  tanh saturated per layer {" ".join(f"{s:.2f}" for s in sat)}  |W_out| {wout:.2f}')
print('PIXELS (PPO trial 1)'); diag({}, f'{P}/runs_kinetix_paper/kinetix/continual/ppo/Kinetix20/trial_1/checkpoints.npz')
print('TRANSFORMER (PPO trial 1)'); diag({'observation': 'entity'}, f'{P}/runs_kinetix_tf/kinetix/continual/ppo/Kinetix20/trial_1/checkpoints.npz')
