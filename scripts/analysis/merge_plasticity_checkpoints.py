"""Merge per-arm shards of `plasticity_checkpoints.py` into one results file.

    merge_plasticity_checkpoints.py shards/*/plasticity_checkpoints.json \
        --out results/elite/plasticity_checkpoints.json

The dormancy pass jits one probe program per (environment, body) and keeps
every compiled program alive for the run. On the gymnax physics family with
nine arms -- 270 runs, twenty bodies each -- the CPU JIT ran out of address
space and died in LLVM ("Cannot allocate memory", then a segfault) after any
eight of the arms had passed on their own (2026-09-13). One process per arm,
then this merge, is the same split the divergence pass already uses for the
MJX bodies (`behavioural_divergence.py --merge`).

`cells` is a nested dict keyed cell -> method -> ...; shards are disjoint in
method, so the merge is a recursive dict union where a later file wins on a
leaf. `meta` is the first shard's, with the union of the cells.
"""
import argparse
import json
import pathlib


def deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(out[k], v) if k in out else v
        return out
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('shards', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    merged = None
    for path in args.shards:
        blob = json.loads(pathlib.Path(path).read_text())
        if merged is None:
            merged = blob
            continue
        merged['cells'] = deep_merge(merged.get('cells', {}), blob.get('cells', {}))
        meta_cells = list(merged.get('meta', {}).get('cells') or [])
        for c in blob.get('meta', {}).get('cells') or []:
            if c not in meta_cells:
                meta_cells.append(c)
        merged.setdefault('meta', {})['cells'] = meta_cells
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=1))
    n = sum(len(v) for v in merged.get('cells', {}).values() if isinstance(v, dict))
    print(f'merged {len(args.shards)} shard(s): {len(merged.get("cells", {}))} cells, '
          f'{n} (cell, method) entries -> {out}')


if __name__ == '__main__':
    main()
