"""Kinetix package.

The kinetix tree is rooted at `source/kinetix`, so `import kinetix` works with
that directory on sys.path and the repo root does not have to be. Some modules
here now import the study's shared code (`source.algorithms.rl.redo`, so every suite
scores a dormant neuron the same way), which needs the repo root as well. The
experiment scripts already put it on the path before importing kinetix; this
makes the library modules importable on their own too, e.g. from a test or a
REPL started inside source/kinetix.
"""

import os as _os
import sys as _sys

# This file is <repo>/source/kinetix/kinetix/__init__.py, so the repo root is
# four levels up. Appended rather than inserted: <repo>/source is itself a
# package directory, and putting any part of it ahead of site-packages makes
# `source/gymnax` shadow the pip `gymnax` distribution that kinetix imports.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
if _REPO_ROOT not in _sys.path:
    _sys.path.append(_REPO_ROOT)
