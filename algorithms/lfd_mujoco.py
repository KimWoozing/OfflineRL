"""Runner for DemoDICE / B-DemoDICE on MuJoCo tasks from offline (d4rl) datasets.

Supports --algorithm demodice | bdemodice.

Typical usage
-------------
# DemoDICE
python algorithms/lfd_mujoco.py --algorithm demodice --gpu_id 1 \
    --expert_dataset hopper-expert-v2  --expert_num_traj 100 \
    --imperfect_datasets hopper-random-v2 --imperfect_num_trajs 500

# B-DemoDICE
python algorithms/lfd_mujoco.py --algorithm bdemodice --gpu_id 0 \
    --expert_dataset hopper-expert-v2  --expert_num_traj 100 \
    --imperfect_datasets hopper-random-v2 --imperfect_num_trajs 500 \
    --alpha 1.

# C-DemoDICE
python algorithms/lfd_mujoco.py --algorithm cdemodice --gpu_id 0 \
    --expert_dataset hopper-expert-v2  --expert_num_traj 100 \
    --imperfect_datasets hopper-random-v2 --imperfect_num_trajs 500 \
    --alpha 1.
"""

from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import warnings

import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import tyro
import wandb

# Transition is the same for both algorithms
Transition = namedtuple("Transition", "obs action reward next_obs done")


# ---------------------------------------------------------------------------
# Arguments  (superset for both algorithms)
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # --- Experiment ---
    gpu_id: int = 0
    task: str = ""   # M1 / M2 / M3 / M4
    env: str = ""    # hopper / walker2d / halfcheetah / ant
    # --- Dataset ---
    expert_dataset: str = "hopper-expert-v2"
    expert_num_traj: int = 5
    imperfect_datasets: tuple[str, ...] = ("hopper-medium-v2",)
    imperfect_num_trajs: tuple[int, ...] = (500,)
    # --- Experiment ---
    seed: int = 0
    algorithm: str = "bdemodice"   # "demodice" | "bdemodice"
    num_updates: int = 2_000_000
    eval_interval: int = 10_000
    eval_workers: int = 8
    eval_final_episodes: int = 100
    # --- Logging ---
    log: bool = False
    wandb_project: str = "DemoDICE"
    wandb_entity: str = "wsk208"
    wandb_name: str = ""  # auto-generated from algo/env/task/seed if empty
    # --- Generic optimisation ---
    lr: float = 3e-5
    q_lr: float = 3e-5
    batch_size: int = 512
    gamma: float = 0.99
    # --- Shared DemoDICE / B-DemoDICE ---
    # demodice  : non_expert_reg = alpha + 1  (paper default: alpha=0.0)
    # bdemodice : temperature α in V loss     (suggested default: alpha=1.0)
    alpha: float = 1.0
    grad_reg_coeff_cost: float = 0.1
    # --- DemoDICE only ---
    grad_reg_coeff_nu: float = 1e-4
    # --- B-DemoDICE only ---
    polyak_step_size: float = 0.005
    exp_adv_clip: float = 100.0


# ---------------------------------------------------------------------------
# Trajectory-level data loading  (same as before)
# ---------------------------------------------------------------------------

def load_trajectories(dataset_name: str, num_traj: int, start_idx: int = 0):
    """Load exactly *num_traj* episode trajectories from a d4rl dataset.

    - Timeout transitions are skipped (mirrors imitation-dice-main/utils.py).
    - Terminal transitions are included.
    - init_obs collects the first observation of each selected episode.
    """
    raw       = d4rl.qlearning_dataset(gym.make(dataset_name))
    obs       = raw["observations"]
    actions   = raw["actions"]
    next_obs  = raw["next_observations"]
    rewards   = raw["rewards"]
    terminals = raw["terminals"].astype(bool)
    timeouts  = raw.get("timeouts", onp.zeros(len(terminals), dtype=bool)).astype(bool)
    episode_ends = terminals | timeouts

    init_obs_list, obs_list, act_list = [], [], []
    nobs_list, rew_list, done_list    = [], [], []

    ep_idx   = 0
    ep_start = 0

    for i in range(len(obs)):
        if ep_idx < start_idx:
            if episode_ends[i]:
                ep_idx  += 1
                ep_start = i + 1
            continue
        if ep_idx >= start_idx + num_traj:
            break
        if i == ep_start:
            init_obs_list.append(obs[i])
        if not timeouts[i]:
            obs_list.append(obs[i])
            act_list.append(actions[i])
            nobs_list.append(next_obs[i])
            rew_list.append(rewards[i])
            done_list.append(terminals[i])
        if episode_ends[i]:
            ep_idx  += 1
            ep_start = i + 1

    def _f32(lst):
        return onp.array(lst, dtype=onp.float32)

    return _f32(init_obs_list), _f32(obs_list), _f32(act_list), \
           _f32(nobs_list), _f32(rew_list), _f32(done_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: Args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    rng = jax.random.PRNGKey(args.seed)

    if args.log:
        run_name = args.wandb_name or (
            f"{args.algorithm}_{args.expert_dataset.split('-')[0]}"
            f"_e{args.expert_num_traj}"
            f"_i{'_'.join(str(n) for n in args.imperfect_num_trajs)}"
            f"_s{args.seed}"
        )
        wandb.init(
            config=asdict(args),
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="train_agent",
        )

    # --- Load expert trajectories ---
    expert_init, expert_obs, expert_act, expert_nobs, expert_rew, expert_done = \
        load_trajectories(args.expert_dataset, args.expert_num_traj, start_idx=0)
    print(f"Expert  : {len(expert_obs):>7} transitions  ({args.expert_dataset}, "
          f"{args.expert_num_traj} traj)")

    # --- Load imperfect trajectories ---
    imp_init_l, imp_obs_l, imp_act_l, imp_nobs_l, imp_rew_l, imp_done_l = \
        [], [], [], [], [], []
    for dname, ntraj in zip(args.imperfect_datasets, args.imperfect_num_trajs):
        start = args.expert_num_traj if dname == args.expert_dataset else 0
        ii, io, ia, ino, ir, id_ = load_trajectories(dname, ntraj, start_idx=start)
        imp_init_l.append(ii);  imp_obs_l.append(io);   imp_act_l.append(ia)
        imp_nobs_l.append(ino); imp_rew_l.append(ir);   imp_done_l.append(id_)
        print(f"Imperfect: {len(io):>7} transitions  ({dname}, {ntraj} traj)")

    imp_init = onp.concatenate(imp_init_l, axis=0)
    imp_obs  = onp.concatenate(imp_obs_l,  axis=0)
    imp_act  = onp.concatenate(imp_act_l,  axis=0)
    imp_nobs = onp.concatenate(imp_nobs_l, axis=0)
    imp_rew  = onp.concatenate(imp_rew_l,  axis=0)
    imp_done = onp.concatenate(imp_done_l, axis=0)

    # Union = imperfect + expert (matching original order)
    union_init = onp.concatenate([imp_init, expert_init], axis=0)
    union_obs  = onp.concatenate([imp_obs,  expert_obs],  axis=0)
    union_act  = onp.concatenate([imp_act,  expert_act],  axis=0)
    union_nobs = onp.concatenate([imp_nobs, expert_nobs], axis=0)
    union_rew  = onp.concatenate([imp_rew,  expert_rew],  axis=0)
    union_done = onp.concatenate([imp_done, expert_done], axis=0)
    print(f"Union   : {len(union_obs):>7} transitions total")

    # Observation statistics from imperfect data (matches original paper)
    obs_mean = jnp.array(imp_obs.mean(0), dtype=jnp.float32)
    obs_std  = jnp.array(imp_obs.std(0),  dtype=jnp.float32)

    def _jnp(arr):
        return jnp.array(arr.astype(onp.float32))

    expert_data = Transition(
        obs=_jnp(expert_obs), action=_jnp(expert_act), reward=_jnp(expert_rew),
        next_obs=_jnp(expert_nobs), done=_jnp(expert_done),
    )
    union_data = Transition(
        obs=_jnp(union_obs), action=_jnp(union_act), reward=_jnp(union_rew),
        next_obs=_jnp(union_nobs), done=_jnp(union_done),
    )

    # --- Environment ---
    env         = gym.vector.make(args.expert_dataset, num_envs=args.eval_workers)
    num_actions = env.single_action_space.shape[0]
    obs_dim     = env.single_observation_space.shape[0]
    dummy_obs    = jnp.zeros(obs_dim)
    dummy_action = jnp.zeros(num_actions)

    # ------------------------------------------------------------------ #
    # Algorithm-specific setup                                            #
    # ------------------------------------------------------------------ #
    if args.algorithm == "demodice":
        from demodice import (
            AgentTrainState, CostNetwork, NuNetwork,
            TanhGaussianActor as Actor,
            create_train_state, eval_agent as eval_fn, make_train_step,
        )

        # init_obs needed for DemoDICE's boundary term (1−γ)·E[ν(s₀)]
        init_obs = _jnp(union_init)

        cost_net  = CostNetwork(obs_mean, obs_std)
        nu_net    = NuNetwork(obs_mean, obs_std)
        actor_net = Actor(num_actions, obs_mean, obs_std)

        rng, rng_cost, rng_nu, rng_actor = jax.random.split(rng, 4)
        agent_state = AgentTrainState(
            cost =create_train_state(args, rng_cost,  cost_net,  [dummy_obs, dummy_action]),
            nu   =create_train_state(args, rng_nu,    nu_net,    [dummy_obs]),
            actor=create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        )
        _train_fn = make_train_step(
            args, cost_net.apply, nu_net.apply, actor_net.apply,
            expert_data, union_data, init_obs,
        )

    elif args.algorithm == "bdemodice":
        from bdemodice import (
            AgentTrainState, CostNetwork, DualQNetwork, StateValueFunction,
            TanhGaussianActor as Actor,
            create_train_state, eval_agent as eval_fn, make_train_step,
        )

        cost_net  = CostNetwork(obs_mean, obs_std)
        q_net     = DualQNetwork(obs_mean, obs_std)
        value_net = StateValueFunction(obs_mean, obs_std)
        actor_net = Actor(num_actions, obs_mean, obs_std)

        rng, rng_cost, rng_q, rng_value, rng_actor = jax.random.split(rng, 5)
        agent_state = AgentTrainState(
            cost          = create_train_state(args, rng_cost,  cost_net,  [dummy_obs, dummy_action]),
            dual_q        = create_train_state(args, rng_q,     q_net,     [dummy_obs, dummy_action], lr=args.q_lr),
            dual_q_target = create_train_state(args, rng_q,     q_net,     [dummy_obs, dummy_action], lr=args.q_lr),
            value         = create_train_state(args, rng_value, value_net, [dummy_obs]),
            actor         = create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        )
        _train_fn = make_train_step(
            args, cost_net.apply, q_net.apply, value_net.apply, actor_net.apply,
            expert_data, union_data,
        )

    elif args.algorithm == "cdemodice":
        from cdemodice import (
            AgentTrainState, CostNetwork, StateCostNetwork, DualQNetwork, StateValueFunction,
            TanhGaussianActor as Actor,
            create_train_state, eval_agent as eval_fn, make_train_step,
        )

        cost_net       = CostNetwork(obs_mean, obs_std)
        state_cost_net = StateCostNetwork(obs_mean, obs_std)
        q_net          = DualQNetwork(obs_mean, obs_std)
        value_net      = StateValueFunction(obs_mean, obs_std)
        actor_net      = Actor(num_actions, obs_mean, obs_std)

        rng, rng_cost, rng_sc, rng_q, rng_value, rng_actor = jax.random.split(rng, 6)
        agent_state = AgentTrainState(
            cost          = create_train_state(args, rng_cost,  cost_net,       [dummy_obs, dummy_action]),
            state_cost    = create_train_state(args, rng_sc,    state_cost_net, [dummy_obs]),
            dual_q        = create_train_state(args, rng_q,     q_net,          [dummy_obs, dummy_action], lr=args.q_lr),
            dual_q_target = create_train_state(args, rng_q,     q_net,          [dummy_obs, dummy_action], lr=args.q_lr),
            value         = create_train_state(args, rng_value, value_net,      [dummy_obs]),
            actor         = create_train_state(args, rng_actor, actor_net,      [dummy_obs]),
        )
        _train_fn = make_train_step(
            args, cost_net.apply, state_cost_net.apply, q_net.apply, value_net.apply, actor_net.apply,
            expert_data, union_data,
        )

    else:
        raise ValueError(f"Unknown algorithm '{args.algorithm}'. "
                         "Choose 'demodice', 'bdemodice', or 'cdemodice'.")

    # ------------------------------------------------------------------ #
    # Training loop  (identical for both algorithms)                      #
    # ------------------------------------------------------------------ #
    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        (rng, agent_state), loss = jax.lax.scan(
            _train_fn, (rng, agent_state), None, args.eval_interval
        )

        rng, rng_eval = jax.random.split(rng)
        returns = eval_fn(args, rng_eval, env, agent_state)
        scores  = d4rl.get_normalized_score(args.expert_dataset, returns) * 100.0

        step = (eval_idx + 1) * args.eval_interval
        loss_str = "  ".join(f"{k}: {float(v[-1]):.4f}" for k, v in loss.items())
        print(f"Step: {step:>8}  |  score: {scores.mean():.2f} ± {scores.std():.2f}"
              f"  |  {loss_str}")
        if args.log:
            wandb.log({
                "return":      returns.mean(),
                "score":       scores.mean(),
                "score_std":   scores.std(),
                "num_updates": step,
                **{k: v[-1] for k, v in loss.items()},
            })

    # --- Final evaluation ---
    if args.eval_final_episodes > 0:
        final_iters = int(onp.ceil(args.eval_final_episodes / args.eval_workers))
        print(f"Final evaluation ({final_iters} iterations)...")
        _rng   = jax.random.split(rng, final_iters)
        rets   = onp.array([eval_fn(args, r, env, agent_state) for r in _rng])
        scores = d4rl.get_normalized_score(args.expert_dataset, rets) * 100.0
        print(f"Final score: {scores.mean():.2f} ± {scores.std():.2f}")

        agg_fn = lambda x, k: {k: x, f"{k}_mean": x.mean(), f"{k}_std": x.std()}
        info   = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")

        os.makedirs("final_returns", exist_ok=True)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{args.algorithm}_{args.expert_dataset}_{time_str}.npz"
        with open(os.path.join("final_returns", filename), "wb") as f:
            onp.savez_compressed(f, **info, args=asdict(args))

        if args.log:
            wandb.save(os.path.join("final_returns", filename))

    env.close()
    if args.log:
        wandb.finish()


if __name__ == "__main__":
    run(tyro.cli(Args))
