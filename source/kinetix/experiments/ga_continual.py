"""
Continual GA (SimpleGA) training on 20 medium Kinetix tasks (h0-h19).
Population carries over between tasks - no reset.

This is the entry point called by scripts/kinetix/run_GA_kinetix_continual.sh.
It wraps the working training code from the evosax_wrapper framework.
"""
import sys
import os
import argparse

# ── path setup ──────────────────────────────────────────────────────
# This script lives at source/kinetix/experiments/ga_continual.py
# ROOT = source/kinetix  (the working directory set by the shell script)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Local evosax shadows pip-installed evosax
methods_path = os.path.join(ROOT, "methods", "evosax_wrapper")
if methods_path not in sys.path:
    sys.path.insert(0, methods_path)

# Local Kinetix shadows pip-installed kinetix
kinetix_path = os.path.join(ROOT, "methods", "Kinetix")
if kinetix_path not in sys.path:
    sys.path.insert(0, kinetix_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continual SimpleGA training on 20 medium Kinetix tasks")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--popsize", type=int, default=704,
                        help="Population size (should be divisible by 32)")
    parser.add_argument("--generations_per_task", type=int, default=200,
                        help="Number of GA generations per task")
    parser.add_argument("--sigma_init", type=float, default=0.001,
                        help="Initial mutation sigma")
    parser.add_argument("--crossover_rate", type=float, default=0.0,
                        help="Crossover rate (0.0 = no crossover)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of independent trials")
    parser.add_argument("--wandb_project", type=str,
                        default="Kinetix-continual-ga",
                        help="Weights & Biases project name")
    parser.add_argument("--project_dir", type=str, default=None,
                        help="Directory for saving project outputs")
    parser.add_argument("--eval_reps", type=int, default=20,
                        help="Number of evaluation repetitions per individual")
    parser.add_argument("--evolve_reps", type=int, default=20,
                        help="Number of evolution repetitions (same as eval_reps)")
    parser.add_argument("--episode_length", type=int, default=256,
                        help="Maximum episode length")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── CUDA setup (before JAX import) ──────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── wandb setup ─────────────────────────────────────────────────
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    import wandb
    import envs  # noqa: F401 – sets up ecorobot path

    from scripts.train.evosax.train_utils import EvosaxExperiment as Experiment
    from scripts.train.evosax.hyperparams import hyperparams

    # ── number of tasks ─────────────────────────────────────────────
    num_tasks = 20  # m/h0 through m/h19
    total_generations = args.generations_per_task * num_tasks

    # ── build configs (matching the working inspiration code) ───────
    exp_config = {
        "seed": args.seed,
        "num_trials": args.num_trials,
    }

    env_config = {
        "env_type": "kinetix",
        "env_name": "m/h0_unicycle",  # first task; train_() iterates all 20
        "curriculum": False,
        "env_params": {},
    }

    es_kws = {
        "sigma_init": args.sigma_init,
        "elite_ratio": 0.5,
    }

    optimizer_config = {
        "optimizer_name": "SimpleGA",
        "optimizer_type": "evosax",
        "optimizer_params": {
            "generations": total_generations,
            "strategy": "SimpleGA",
            "popsize": args.popsize,
            "es_kws": es_kws,
        },
    }

    model_config = {
        "network_type": "kinetix",
        "model_params": hyperparams["kinetix"],
        "actor_only": True,
    }

    # ── create & run experiment ─────────────────────────────────────
    exp = Experiment(
        env_config=env_config,
        optimizer_config=optimizer_config,
        model_config=model_config,
        exp_config=exp_config,
    )

    # Override project_dir if provided via CLI
    if args.project_dir is not None:
        exp._cli_project_dir = args.project_dir
    exp._cli_wandb_project = args.wandb_project

    # Monkey-patch setup() to use our project_dir and wandb project
    _orig_setup = exp.setup

    def patched_setup():
        _orig_setup()
        if args.project_dir is not None:
            exp.config["exp_config"]["project_dir"] = args.project_dir
            # Recreate trial directories under the custom project dir
            for trial in range(exp.config["exp_config"]["num_trials"]):
                trial_dir = os.path.join(args.project_dir, f"trial_{trial}")
                os.makedirs(os.path.join(trial_dir, "data/train/checkpoints"), exist_ok=True)
                os.makedirs(os.path.join(trial_dir, "data/eval/trajs"), exist_ok=True)
                os.makedirs(os.path.join(trial_dir, "visuals/eval/trajs"), exist_ok=True)
                os.makedirs(os.path.join(trial_dir, "visuals/train/policy"), exist_ok=True)
                os.makedirs(os.path.join(trial_dir, "visuals/eval/network_activations"), exist_ok=True)

    exp.setup = patched_setup

    # Monkey-patch setup_trial() to use our wandb project name
    _orig_setup_trial = exp.setup_trial

    def patched_setup_trial(trial):
        exp.config["exp_config"]["trial_dir"] = os.path.join(
            exp.config["exp_config"]["project_dir"], f"trial_{trial}")
        exp.config["exp_config"]["experiment"] = f"SimpleGA_trial_{trial}"
        exp.config["exp_config"]["trial_seed"] = trial + args.seed

        wandb.init(
            project=args.wandb_project,
            name=f"SimpleGA_trial_{trial}_continual_pixels",
            tags=[f"trial_{trial}"],
            config=exp.config,
            reinit=True,
        )

        exp.setup_trial_keys()
        exp.init_model()

    exp.setup_trial = patched_setup_trial

    exp.run()


if __name__ == "__main__":
    main()
