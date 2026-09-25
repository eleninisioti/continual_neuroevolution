"""The Kinetix study: NE and RL over Kinetix's hand-designed levels.

Lived inside the vendored Kinetix checkout, at what is now
`third_party/kinetix/experiments/`,
until 2026-09-08, which made our experiments look like part of a third-party
package and meant they could only be run from one working directory -- the
three continual NE trainers imported `ne_continual_io` as a bare top-level
module. The checkout itself moved the same day, out of `source/` entirely and
into `third_party/kinetix/`, which pyproject installs editable from there.

A sub-task here is WHICH LEVEL, the same shape as the minigrid suite (which
environment). `source/envs/kinetix.py` is that suite.
"""
