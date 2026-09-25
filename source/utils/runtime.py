"""Process setup that has to happen before JAX is imported.

`_get_gpu_arg` reads `--gpus` out of `sys.argv` by hand rather than through
argparse: `CUDA_VISIBLE_DEVICES` has no effect once JAX has initialised its
backend, so the value is needed at import time, which is before a parser exists.
All ten gymnax trainers carried an identical copy of this, and eight carried an
identical `Tee`.
"""

import os
import sys


def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpus' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def select_gpus():
    """Honour `--gpus` and disable XLA preallocation. Call before importing jax.

    Returns the device string that was applied, or None if `--gpus` was absent.
    Preallocation is off because several trainers share a GPU in every sweep in
    `scripts/outdated/train/`; with it on the first process claims the whole card.
    """
    gpu = _get_gpu_arg()
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        print(f"Setting CUDA_VISIBLE_DEVICES={gpu}")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return gpu


class Tee:
    """Duplicate stdout to a file and console."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', buffering=1)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def write_run_config(output_dir, config, policy_arch=None):
    """Write `config.json` next to a run, so the run says what it trained.

    The RL trainers already did this; the NE ones recorded their config only
    inside the checkpoint pickles, so a finished GA run on cheetah or ant had no
    cheap, readable record of the network it searched. That is what
    `scripts/check_architectures.py` needs to confirm that every method compared
    on a task used the same policy -- and it could not check the NE side.

    `policy_arch` is a suite key from `source.algorithms.networks.POLICY_ARCH`; passing
    it stamps the hidden activation AND the hidden widths into the record. NE
    policies take both from the class rather than a flag, so there is otherwise
    nothing to write down -- and `check_architectures.py` needs the widths, not
    just the activation: `_policy_arch_of` returns None without them, so a run
    that recorded only the activation was still reported UNVERIFIABLE. That is
    what every brax and mujoco NE run looked like, including freshly written
    ones, which defeats the point of writing the file.

    `setdefault` on both, so a trainer that takes the architecture as a flag
    (gymnax, and the RL trainers) keeps what it was actually given rather than
    having POLICY_ARCH asserted over it -- a disagreement between the two is
    exactly what the check exists to catch, and silently overwriting it here
    would hide it.

    Best-effort: a run that cannot write this file has still trained, and the
    metrics are written elsewhere.
    """
    import json
    import os

    payload = dict(config)
    if policy_arch is not None:
        from source.algorithms.networks import POLICY_ARCH
        arch = POLICY_ARCH[policy_arch]
        payload.setdefault('activation', arch['activation'])
        payload.setdefault('policy_hidden_dims', list(arch['hidden_dims']))
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"  (could not write config.json: {type(exc).__name__}: {exc})")
