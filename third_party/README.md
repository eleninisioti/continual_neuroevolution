# third_party

Vendored third-party source trees, kept in-repo because we have local changes
to them and they are installed editable from here.

    kinetix/   The Kinetix benchmark (FlairOX/Kinetix), with our own additions
               under `kinetix/kinetix/models/` (`redo.py`, `cchain.py`). It has
               its own LICENSE and pyproject and is installed by the root
               pyproject as `kinetix-env = { path = "third_party/kinetix" }`,
               which is why moving it needs a reinstall, not just a `git mv`.

Our Kinetix *experiments* are not here -- they are a study, and they live at
`source/studies/kinetix/`. This tree held them until 2026-09-08, which made our
code look like part of a third-party package.
