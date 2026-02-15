import jax.numpy as jnp
from flax import struct
from kinetix.environment.env_state import EnvState, StaticEnvParams
from kinetix.render.renderer_pixels import make_render_pixels


@struct.dataclass
class PixelsObservation:
    image: jnp.ndarray
    global_info: jnp.ndarray


def make_render_pixels_rl(env_params, static_params: StaticEnvParams):
    render_fn = make_render_pixels(env_params, static_params)

    def inner(state):
        pixels = render_fn(state) / 255.0
        return PixelsObservation(
            image=pixels,
            global_info=jnp.array([state.gravity[1] / 10.0]),
        )

    return inner
