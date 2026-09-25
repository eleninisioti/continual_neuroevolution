"""How many Kinetix levels the CENTROID (population weight mean; the policy for
RL) and the ELITE solve, from the training records -- stationary block and
chain, and at the FIRST record, which separates learning from what the initial
population already solved (~30% of level-runs for RL and ES).

Read on the machine the runs trained on: the records were scored there, and
re-scoring Kinetix NE agents on other hardware flips marginal solutions
(scripts/analysis/kinetix_rescore_saved_agents.py). A running chain keeps its
records in resume.pkl and is reported as running.

    python scripts/analysis/kinetix_solved_counts.py LABEL=<tree>/kinetix [LABEL=<tree>/kinetix ...]
"""
import json, glob, os, sys, collections
if len(sys.argv) < 2:
    sys.exit(__doc__)
TREES = tuple(tuple(a.split("=", 1)) for a in sys.argv[1:])
ARMS = ("ga", "dns", "dns_gaussian", "es", "nes", "ppo", "trac", "redo", "cchain")
TH = 1.0

def cols(rec):
    """(mean prefix, elite prefix), decided by what the record carries: kinetix_repo
    wrote centroid=elite/popmean=mean, the shared runner incumbent=elite/centroid=mean,
    RL writes the policy under centroid."""
    if "popmean_task0" in rec: return "popmean", "centroid"
    if "incumbent_task0" in rec: return "centroid", "incumbent"
    return "centroid", "centroid"

print("#################### STATIONARY: one level per run, end of training (and at the FIRST record)")
for label, tree in TREES:
    print(f"\n=== {label}")
    print(f"{'arm':13s} {'trials':>6s} | {'centroid solved':>15s} {'at 1st record':>13s} {'levels all trials':>17s} {'levels any trial':>16s} | {'elite solved':>12s}")
    for arm in ARMS:
        per_level = collections.defaultdict(list); n = c_end = c_start = e_end = 0
        for f in sorted(glob.glob(f"{tree}/noncontinual/{arm}/*/trial_*/training_metrics.json")):
            rows = json.load(open(f)); mc, ec = cols(rows[-1]); lvl = f.split("/")[-3]
            end_ok = rows[-1][f"{mc}_task0"] >= TH
            n += 1; c_end += end_ok; c_start += rows[0][f"{mc}_task0"] >= TH; e_end += rows[-1][f"{ec}_task0"] >= TH
            per_level[lvl].append(end_ok)
        if not n: continue
        allt = sum(all(v) for v in per_level.values()); anyt = sum(any(v) for v in per_level.values())
        print(f"{arm:13s} {n:6d} | {c_end:9d}/{n:<5d} {c_start:7d}/{n:<5d} {allt:11d}/{len(per_level):<5d} {anyt:10d}/{len(per_level):<5d} | {e_end:6d}/{n:<5d}")

print("\n#################### CHAIN Kinetix20: 20 levels in sequence (finished runs only)")
for label, tree in TREES:
    print(f"\n=== {label}")
    print(f"{'arm':13s} {'trials':>6s} | {'centroid: online':>16s} {'zero-shot':>9s} {'final net':>9s} | {'elite: online':>13s} {'final net':>9s}   (per-trial mean, of 20)")
    for arm in ARMS:
        fs = sorted(glob.glob(f"{tree}/continual/{arm}/Kinetix20/trial_*/training_metrics.json"))
        if not fs:
            running = len(glob.glob(f"{tree}/continual/{arm}/Kinetix20/trial_*/resume.pkl"))
            if running: print(f"{arm:13s} {'--':>6s} | still running ({running} trials with resume.pkl)")
            continue
        agg = collections.defaultdict(list)
        for f in fs:
            rows = json.load(open(f)); mc, ec = cols(rows[-1])
            ph, s = [], 0
            for i in range(1, len(rows) + 1):
                if i == len(rows) or rows[i]["task"] != rows[s]["task"]:
                    ph.append((int(rows[s]["task"]), s, i - 1)); s = i
            agg["c_on"].append(sum(rows[b][f"{mc}_task{t}"] >= TH for t, a, b in ph))
            agg["c_zs"].append(sum(rows[a][f"{mc}_task{t}"] >= TH for t, a, b in ph))
            agg["e_on"].append(sum(rows[b][f"{ec}_task{t}"] >= TH for t, a, b in ph))
            last = rows[-1]; T = len({t for t, _, _ in ph})
            agg["c_fin"].append(sum(last[f"{mc}_task{t}"] >= TH for t in range(T)))
            agg["e_fin"].append(sum(last[f"{ec}_task{t}"] >= TH for t in range(T)))
        m = {k: sum(v) / len(v) for k, v in agg.items()}
        print(f"{arm:13s} {len(fs):6d} | {m['c_on']:16.1f} {m['c_zs']:9.1f} {m['c_fin']:9.1f} | {m['e_on']:13.1f} {m['e_fin']:9.1f}   centroid online per trial {agg['c_on']}")
