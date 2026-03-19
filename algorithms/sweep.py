"""Sweep runner for demodice / bdemodice / cdemodice.

Distributes jobs across N_GPUS GPUs (one job per GPU at a time).

Tasks (matching DemoDICE paper notation):
  M1 : 400 expert traj + 100  random traj
  M2 : 400 expert traj + 400  random traj
  M3 : 400 expert traj + 1600 random traj
  M4 : 100 expert traj + 1600 random traj  (new)

Usage:
  python algorithms/sweep.py                   # all envs, all algos
  python algorithms/sweep.py --envs hopper      # single env
  python algorithms/sweep.py --algos bdemodice cdemodice
  python algorithms/sweep.py --tasks M3 M4
  python algorithms/sweep.py --dry_run          # print commands only
"""

import argparse
import subprocess
import time
from itertools import product

# ------------------------------------------------------------------ #
# Experiment grid                                                     #
# ------------------------------------------------------------------ #

ALL_ENVS  = ["hopper", "walker2d", "halfcheetah", "ant"]
ALL_ALGOS = ["demodice", "bdemodice", "cdemodice"]
ALL_TASKS = {
    "M1": dict(expert_num_traj=400, imperfect_num_trajs=100),
    "M2": dict(expert_num_traj=400, imperfect_num_trajs=400),
    "M3": dict(expert_num_traj=400, imperfect_num_trajs=1600),
    "M4": dict(expert_num_traj=100, imperfect_num_trajs=1600),
}
ALL_SEEDS = [0, 1, 2, 3, 4]

N_GPUS = 4


# ------------------------------------------------------------------ #
# Build command for one run                                           #
# ------------------------------------------------------------------ #

def make_cmd(env, algo, task_name, task_cfg, seed, gpu_id, extra_args,
             wandb_project):
    e = task_cfg["expert_num_traj"]
    i = task_cfg["imperfect_num_trajs"]
    run_name = f"{algo}_{env}_{task_name}_s{seed}"   # e.g. bdemodice_hopper_M3_s2
    cmd = [
        "python", "algorithms/lfd_mujoco.py",
        "--algorithm",            algo,
        "--env",                  env,
        "--task",                 task_name,
        "--expert-dataset",       f"{env}-expert-v2",
        "--imperfect-datasets",   f"{env}-random-v2",
        "--expert-num-traj",      str(e),
        "--imperfect-num-trajs",  str(i),
        "--seed",                 str(seed),
        "--gpu-id",               str(gpu_id),
        "--log",
        "--wandb-project",        wandb_project,
        "--wandb-name",           run_name,
    ] + extra_args
    return cmd


# ------------------------------------------------------------------ #
# GPU-aware job runner                                                #
# ------------------------------------------------------------------ #

def run_sweep(jobs, dry_run=False, poll_interval=30):
    gpu_procs = [None] * N_GPUS  # slot → (Popen | None)

    def first_free_gpu():
        for gid, proc in enumerate(gpu_procs):
            if proc is None or (not isinstance(proc, bool) and proc.poll() is not None):
                return gid
        return None

    job_idx = 0
    total = len(jobs)

    while job_idx < total or any(
        p is not None and p.poll() is None for p in gpu_procs
    ):
        gid = first_free_gpu()

        if gid is not None and job_idx < total:
            env, algo, task_name, seed, cmd_template = jobs[job_idx]
            cmd = [c if c != "__GPU__" else str(gid) for c in cmd_template]
            label = f"[{job_idx+1}/{total}] GPU={gid} {algo} {env} {task_name} seed={seed}"

            if dry_run:
                print(label)
                print("  " + " ".join(cmd))
                gpu_procs[gid] = True   # placeholder so we don't reuse same slot
            else:
                print(label)
                log_path = f"logs/{algo}_{env}_{task_name}_s{seed}.log"
                import os; os.makedirs("logs", exist_ok=True)
                with open(log_path, "w") as logf:
                    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
                gpu_procs[gid] = proc

            job_idx += 1
        else:
            # Nothing free right now — wait
            if dry_run:
                break   # in dry-run all slots are "busy" immediately; just break
            time.sleep(poll_interval)

    if not dry_run:
        # Wait for remaining jobs
        for proc in gpu_procs:
            if proc is not None and proc.poll is not None:
                try:
                    proc.wait()
                except Exception:
                    pass

    print("All jobs submitted.")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",   nargs="+", default=ALL_ENVS,  choices=ALL_ENVS)
    parser.add_argument("--algos",  nargs="+", default=ALL_ALGOS, choices=ALL_ALGOS)
    parser.add_argument("--tasks",  nargs="+", default=list(ALL_TASKS.keys()),
                        choices=list(ALL_TASKS.keys()))
    parser.add_argument("--seeds",  nargs="+", type=int, default=ALL_SEEDS)
    parser.add_argument("--wandb_project", default="DemoDICE")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Extra args forwarded to lfd_mujoco.py")
    args = parser.parse_args()

    tasks = {k: ALL_TASKS[k] for k in args.tasks}

    jobs = []
    for env, algo, (task_name, task_cfg), seed in product(
        args.envs, args.algos, tasks.items(), args.seeds
    ):
        cmd = make_cmd(env, algo, task_name, task_cfg, seed, gpu_id="__GPU__",
                       extra_args=args.extra,
                       wandb_project=args.wandb_project)
        jobs.append((env, algo, task_name, seed, cmd))

    print(f"Total jobs: {len(jobs)}  |  GPUs: {N_GPUS}  |  dry_run={args.dry_run}")
    run_sweep(jobs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
