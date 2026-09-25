"""Per-rollout behaviour signals for the brax ant trainers.

The counterpart of `source/studies/mujoco/behaviour_mujoco.py`: the ant's hand-designed
descriptor is the duty factor of each of its four feet, and foot contact is not
in `state.obs`, which is `concat(qpos[2:], qvel)`. Brax's own contact array is
no help either -- `pipeline_state.contact` is None on the generalized pipeline
that `envs.create('ant')` uses -- so contact is recovered geometrically from the
foot spheres, exactly as the cheetah does it from the foot capsules.

The geometry is done differently from the MJX version for one reason: a brax
pipeline state carries per-*link* transforms (`x.pos`, `x.rot`), not the
per-geom world frames MJX exposes as `data.geom_xpos`. So the foot's world
position is reconstructed as `link_pos + rotate(geom_local_pos, link_rot)`,
which was checked against `mj_forward` on the same qpos and agrees to 1e-7.

    feet = AntFeet.from_env(env)
    ...
    contact = feet.contact(state.pipeline_state)   # (4,) in {0, 1}

This module holds the model-specific part (which geoms are feet, where they sit
on their link, how big they are) so GA, ES and DNS all measure the same thing.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax import math as brax_math

from source.metrics.behaviour_descriptors import FOOT_CONTACT_HEIGHT

# Order fixed by the descriptor's handcrafted_names in
# source/metrics/behaviour_descriptors.py. These are the sphere geoms at the tip
# of each ankle capsule -- the only parts of the ant that touch the floor in a
# normal gait.
ANT_FOOT_GEOMS = ("left_foot_geom", "right_foot_geom",
                  "third_foot_geom", "fourth_foot_geom")


@dataclasses.dataclass(frozen=True)
class AntFeet:
    """The foot spheres of a brax legged model, resolved to links and offsets.

    Attributes:
        link_ids: index into `pipeline_state.x` of the link each foot rides on.
            Brax link i is MuJoCo body i + 1, since body 0 is the world.
        local_pos: (n_feet, 3) foot centre in that link's frame.
        radii: (n_feet,) sphere radius per foot.
    """

    link_ids: np.ndarray
    local_pos: np.ndarray
    radii: np.ndarray
    names: tuple = ANT_FOOT_GEOMS

    @classmethod
    def from_env(cls, env, names=ANT_FOOT_GEOMS):
        sys = env.sys
        model = sys.mj_model
        ids = []
        for name in names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(
                    f"geom {name!r} not found in the model; the foot-contact "
                    "descriptor is specific to brax's ant xml."
                )
            ids.append(gid)
        ids = np.asarray(ids, dtype=np.int32)

        body_ids = np.asarray(sys.geom_bodyid)[ids]
        if np.any(body_ids < 1):
            raise ValueError(
                "a foot geom is attached to the world body, which has no link "
                "transform to read; check ANT_FOOT_GEOMS."
            )
        return cls(
            link_ids=(body_ids - 1).astype(np.int32),
            local_pos=np.asarray(sys.geom_pos)[ids].astype(np.float32),
            # size = (radius, ., .) for a sphere.
            radii=np.asarray(sys.geom_size)[ids, 0].astype(np.float32),
            names=tuple(names),
        )

    def contact(self, pipeline_state, threshold=FOOT_CONTACT_HEIGHT):
        """(n_feet,) 0/1 contact flags for one brax pipeline state.

        A foot counts as down when the bottom of its sphere -- centre height
        minus radius -- is at or below `threshold`. Verified against the
        simulator: with zero torques the ant settles on all four feet and this
        returns duty factors around 0.9, while under uniform random torques the
        feet leave the ground and the duty factors drop to ~0.15.
        """
        x = pipeline_state.x
        link_ids = jnp.asarray(self.link_ids)
        local_pos = jnp.asarray(self.local_pos)

        world = jax.vmap(
            lambda i, p: x.pos[i] + brax_math.rotate(p, x.rot[i])
        )(link_ids, local_pos)
        bottom = world[:, 2] - jnp.asarray(self.radii)
        return (bottom < threshold).astype(jnp.float32)

    def torso_xy(self, pipeline_state):
        """(2,) world x/y of the root link, for the ant's novelty descriptor."""
        return pipeline_state.x.pos[0, :2]
