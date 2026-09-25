"""Do this repo's cross-module references still point at something?

A refactor that moves code between modules breaks in three ways that neither
`python -c "import x"` nor a parse check will notice, because all three live
inside function bodies that no smoke test executes:

  1. `from source.a.b import c` where `c` no longer exists in `source/a/b.py`.
  2. `from source.a import b` followed by `b.c(...)` where `b` has no `c`.
  3. a free name a function uses that nothing in the module binds -- what you
     get when a function is lifted into a module that does not import what it
     needed (`flatten_util`, say).

The second and third are the ones that bite. A path-scoped `sed` over a rename
can rewrite a module alias in a call site it was never meant to touch, and
moving a function between modules silently leaves its imports behind; either
way the result is a NameError thousands of generations into a run rather than
at import. This walks every `source/` and `scripts/` file statically and
reports all three, with no imports and no JAX.

It is deliberately conservative: it only checks names it can resolve to a file
under `source/`, only top-level definitions, and skips modules it cannot parse.
A module that builds its namespace dynamically will produce false positives --
add it to `DYNAMIC` rather than loosening the check.

Run:  .venv/bin/python scripts/check_imports.py
Exit: 0 when clean, 1 when anything is unresolved.
"""

from __future__ import annotations

import ast
import builtins
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Modules whose top-level namespace is not fully visible to a static read.
DYNAMIC: set[str] = set()

# `third_party/kinetix/kinetix` is a vendored third-party package with its own
# layout; it is not ours to keep consistent.
SKIP_PREFIXES = ('third_party/kinetix/kinetix/', 'third_party/kinetix/examples/')


def module_path(dotted: str) -> pathlib.Path | None:
    """`source.a.b` -> the file that defines it, or None if not ours."""
    if not dotted.startswith('source.'):
        return None
    base = REPO / dotted.replace('.', '/')
    for cand in (base.with_suffix('.py'), base / '__init__.py'):
        if cand.exists():
            return cand
    return None


def top_level(path: pathlib.Path) -> set[str] | None:
    """Names a module binds at its top level, or None if it cannot be read."""
    try:
        tree = ast.parse(path.read_text(errors='replace'))
    except SyntaxError:
        return None
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            names |= {(a.asname or a.name.split('.')[0]) for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {(a.asname or a.name) for a in node.names}
        elif isinstance(node, (ast.If, ast.Try)):
            # conditional definitions: collect them, do not require them
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    names |= {t.id for t in sub.targets if isinstance(t, ast.Name)}
    return names


def free_names(tree: ast.Module) -> dict[str, int]:
    """Names read but never bound anywhere in the module, and where first read.

    Deliberately whole-module rather than per-scope: a function that reads a
    module-level constant is fine, and tracking real scopes would only add
    false positives without catching anything this is for.
    """
    bound: set[str] = set(dir(builtins)) | {'__file__', '__name__', '__doc__'}
    used: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound |= {(a.asname or a.name.split('.')[0]) for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            bound |= {(a.asname or a.name) for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Global):
            bound |= set(node.names)
        elif isinstance(node, ast.Nonlocal):
            bound |= set(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            else:
                used.setdefault(node.id, node.lineno)
    return {k: v for k, v in used.items() if k not in bound}


def main() -> int:
    cache: dict[pathlib.Path, set[str] | None] = {}

    def names_of(dotted: str) -> set[str] | None:
        p = module_path(dotted)
        if p is None or dotted in DYNAMIC:
            return None
        if p not in cache:
            cache[p] = top_level(p)
        return cache[p]

    problems: list[str] = []
    files = [p for p in list((REPO / 'source').rglob('*.py'))
             + list((REPO / 'scripts').rglob('*.py'))
             if '__pycache__' not in str(p)
             and not any(str(p.relative_to(REPO)).startswith(s) for s in SKIP_PREFIXES)]

    for path in sorted(files):
        rel = path.relative_to(REPO)
        try:
            tree = ast.parse(path.read_text(errors='replace'))
        except SyntaxError as exc:
            problems.append(f"{rel}:{exc.lineno}: does not parse -- {exc.msg}")
            continue

        # alias -> dotted module, for `from source.pkg import mod` / `import source.pkg.mod as m`
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('source'):
                for a in node.names:
                    if a.name == '*':
                        continue
                    # (1) is the imported name actually there?
                    have = names_of(node.module)
                    if have is not None and a.name not in have:
                        # it may itself be a submodule
                        if module_path(f"{node.module}.{a.name}") is None:
                            problems.append(
                                f"{rel}:{node.lineno}: {node.module} has no {a.name!r}")
                    if module_path(f"{node.module}.{a.name}") is not None:
                        aliases[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith('source.') and a.asname:
                        aliases[a.asname] = a.name

        # (3) free names nothing in the module binds
        for name, lineno in sorted(free_names(tree).items(), key=lambda kv: kv[1]):
            problems.append(f"{rel}:{lineno}: {name!r} is not defined in this module")

        # (2) attribute access through a module alias
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                dotted = aliases.get(node.value.id)
                if not dotted:
                    continue
                have = names_of(dotted)
                if have is not None and node.attr not in have:
                    if module_path(f"{dotted}.{node.attr}") is None:
                        problems.append(
                            f"{rel}:{node.lineno}: {node.value.id}.{node.attr} -- "
                            f"{dotted} defines no {node.attr!r}")

    if problems:
        print(f"{len(problems)} unresolved reference(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    print(f"clean: {len(files)} files, every source.* reference resolves")
    return 0


if __name__ == '__main__':
    sys.exit(main())
