import os
import sys
import argparse
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

# Ensure project root is on path
sys.path.append(".")

from flax.serialization import to_state_dict

from kinetix.environment.env import make_kinetix_env
from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.render import make_render_pixels
from kinetix.util.config import normalise_config
from methods.Kinetix.kinetix.util.saving import load_from_json_file


def create_gif_from_frames(frames: List[jnp.ndarray], outfile: str, fps: int = 10) -> None:
    if not frames:
        return
    pil_frames = [Image.fromarray(np.array(f)) for f in frames]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    pil_frames[0].save(
        outfile,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved GIF to: {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Play Kinetix level with random actions and render GIF.")
    parser.add_argument("--level", type=str, default="l/lever_puzzle", help="Kinetix level path (e.g., l/lever_puzzle)")
    parser.add_argument("--steps", type=int, default=300, help="Max steps to simulate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default="projects/benchmarking/playground/kinetix_random.gif", help="Output GIF path")
    parser.add_argument("--config", type=str, default="scripts/train/evosax/kinetix_config.yaml", help="Kinetix config YAML path")
    args = parser.parse_args()

    # Lazy import to avoid any JAX/GPU init before env var setup in parent process
    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = normalise_config(cfg, name="PPO")

    env_state, static_env_params, env_params = load_from_json_file(args.level)

    # Build env consistent with training setup
    cfg["env_params"] = to_state_dict(env_params)
    cfg["static_env_params"] = to_state_dict(static_env_params)

    reset_fn = make_reset_fn_from_config(cfg, env_params, static_env_params)
    env = make_kinetix_env(
        observation_type=cfg["observation_type"],
        action_type=cfg["action_type"],
        reset_fn=reset_fn,
        env_params=env_params,
        static_env_params=static_env_params,
    )

    renderer = make_render_pixels(env_params, static_env_params)

    # Deterministic reset to the JSON level state
    rng = jax.random.PRNGKey(args.seed)
    rng, key_reset = jax.random.split(rng)
    obs, state = env.reset(key_reset, env_params=env_params, override_reset_state=env_state)

    # Print some diagnostics about the level
    num_circles = int(static_env_params.num_circles)
    num_polys = int(static_env_params.num_polygons)
    print(f"Loaded level: {args.level}")
    print(f"Static shapes -> circles: {num_circles}, polygons: {num_polys}, joints: {int(static_env_params.num_joints)}")

    # Rollout with random actions
    frames: List[jnp.ndarray] = []
    action_space = env.action_space(env_params)

    for t in range(args.steps):
        # Render frame (H, W, 3), convert to uint8 and flip like training eval
        pixels = renderer(state)
        frame = pixels.astype(jnp.uint8).transpose(1, 0, 2)[::-1]
        frames.append(frame)

        rng, key_act, key_step = jax.random.split(rng, 3)
        action = action_space.sample(key_act)

        obs, state, reward, done, info = env.step(key_step, state, action, env_params)

        if bool(done):
            print(f"Episode ended at step {t + 1}, reward={float(reward):.3f}")
            break

    create_gif_from_frames(frames, args.out, fps=10)


if __name__ == "__main__":
    # If you need to force a particular GPU in debug runs, export before launching:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    main()


