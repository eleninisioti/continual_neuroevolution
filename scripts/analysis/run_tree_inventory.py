"""Every run tree of a suite: cells, arms, trials finished/started, the settings
that tell two trees apart (task mod, target speed, speed targets, sub-tasks,
warm-up, noise width, phases, steps) and when the newest trial finished.

    python scripts/analysis/run_tree_inventory.py <projects/iclr_2027> [--suite mjx] [pattern ...]

Patterns default to the MJX trees (runs_mjx* runs_ant* runs_cheetah*).
"""
import json, glob, os, sys, time, collections
args = sys.argv[1:]
if not args:
    sys.exit(__doc__)
IC = args.pop(0)
SUITE = "mjx"
if args[:1] == ["--suite"]:
    SUITE = args[1]; args = args[2:]
patterns = args or ["runs_mjx*", "runs_ant*", "runs_cheetah*"]
trees = sorted(set(p for pat in patterns for p in glob.glob(f"{IC}/{pat}")))
def cfg_of(tdir):
    for n in ("results.json", "config.json"):
        p = os.path.join(tdir, n)
        if os.path.exists(p):
            b = json.load(open(p)); return {**(b.get("config") or {}), **{k: v for k, v in b.items() if k != "config"}}
    return {}
for tree in trees:
    name = os.path.basename(tree)
    for phase in ("continual", "noncontinual"):
        base = f"{tree}/{SUITE}/{phase}"
        if not os.path.isdir(base):
            continue
        cells = collections.defaultdict(dict)
        newest = 0; sample = {}
        for arm_dir in sorted(glob.glob(f"{base}/*")):
            arm = os.path.basename(arm_dir)
            for cell_dir in sorted(glob.glob(f"{arm_dir}/*")):
                cell = os.path.basename(cell_dir)
                trials = glob.glob(f"{cell_dir}/trial_*")
                fin = [t for t in trials if os.path.exists(f"{t}/training_metrics.json")]
                cells[cell][arm] = f"{len(fin)}/{len(trials)}"
                for t in fin:
                    newest = max(newest, os.path.getmtime(f"{t}/training_metrics.json"))
                if fin and cell not in sample and arm in ("ppo", "es"):
                    c = cfg_of(fin[0]); task = c.get("task") or {}
                    opts = task.get("options") or {}
                    sample[cell] = dict(mod=task.get("task_mod"), tgt=opts.get("target_speed"),
                                        spd=opts.get("speed_targets"), nt=c.get("num_tasks"),
                                        wu=c.get("task_warmup"), nr=c.get("noise_range"),
                                        ph=len(c.get("task_sequence") or []),
                                        steps=c.get("num_timesteps") or c.get("env_steps"))
        for cell, arms in cells.items():
            print(f"{name:30s} {phase[:4]:4s} {cell:26s} {time.strftime('%m-%d %H:%M', time.localtime(newest))} "
                  f"arms={','.join(f'{a}:{v}' for a, v in sorted(arms.items()))}  {sample.get(cell, {})}")
