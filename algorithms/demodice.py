from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import warnings

import distrax
import d4rl
import flax.linen as nn
from flax.training.train_state import TrainState
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tyro
import wandb

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

EPS = jnp.finfo(jnp.float32).eps
EPS2 = 1e-3


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    gpu_id: int = 0
    expert_dataset: str = "hopper-expert-v2"
    union_dataset: str = "hopper-medium-v2"
    algorithm: str = "demodice"
    num_updates: int = 1_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    # --- Generic optimization ---
    lr: float = 3e-4
    actor_lr: float = 3e-5
    batch_size: int = 256
    gamma: float = 0.99
    # --- DemoDICE ---
    alpha: float = 0.0          # non_expert_regularization = alpha + 1
    grad_reg_coeff_cost: float = 10.0
    grad_reg_coeff_nu: float = 1e-4


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple("AgentTrainState", "cost nu actor")
Transition = namedtuple("Transition", "obs action reward next_obs done")


class CostNetwork(nn.Module):
    """Discriminator-like cost network.

    Takes (obs, action) separately so obs can be normalised internally,
    keeping the interface consistent with the other networks.
    """
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class NuNetwork(nn.Module):
    """State-value (ν) network for the stationary distribution ratio."""
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs):
        x = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class TanhGaussianActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array
    log_std_max: float = 2.0
    log_std_min: float = -5.0

    @nn.compact
    def __call__(self, x):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.num_actions)(x)
        log_std = nn.Dense(self.num_actions)(x)
        std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
        return distrax.Transformed(distrax.Normal(mean, std), distrax.Tanh())


def create_train_state(args, rng, network, dummy_input, lr=None):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
    )


def eval_agent(args, rng, env, agent_state):
    # --- Reset environment ---
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, args.eval_workers)
    obs = env.reset()

    # --- Rollout agent ---
    # obs normalisation is baked into TanhGaussianActor, so raw env obs are fine here
    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action)

    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    while step < max_episode_steps and not returned.all():
        # --- Take step in environment ---
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, args.eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, done, info = env.step(onp.array(action))

        # --- Track cumulative reward ---
        cum_reward += reward * ~returned
        returned |= done

    if step >= max_episode_steps and not returned.all():
        warnings.warn("Maximum steps reached before all episodes terminated")
    return cum_reward


r"""
          __/)
       .-(__(=:
    |\ |    \)
    \ ||
     \||
      \|
    ___|_____
    \       /
     \     /
      \___/     Agent
"""


def make_train_step(
    args, cost_apply_fn, nu_apply_fn, actor_apply_fn,
    expert_data, union_data, init_obs,
):
    """Return a JIT-compatible DemoDICE train step.

    cost_apply_fn : (params, obs, action) -> scalar per sample
    nu_apply_fn   : (params, obs)         -> scalar per sample
    actor_apply_fn: (params, obs)         -> distrax distribution
    """
    non_expert_reg = args.alpha + 1.0
    obs_dim = expert_data.obs.shape[-1]  # static at trace time

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Sample batches ---
        rng, rng_expert, rng_union, rng_init = jax.random.split(rng, 4)
        expert_idx = jax.random.randint(rng_expert, (args.batch_size,), 0, len(expert_data.obs))
        union_idx  = jax.random.randint(rng_union,  (args.batch_size,), 0, len(union_data.obs))
        init_idx   = jax.random.randint(rng_init,   (args.batch_size,), 0, len(init_obs))

        exp  = jax.tree_util.tree_map(lambda x: x[expert_idx], expert_data)
        uni  = jax.tree_util.tree_map(lambda x: x[union_idx],  union_data)
        init_batch = init_obs[init_idx]

        # --- Update cost (discriminator) ---
        rng, rng_gp_alpha, rng_gp_perm = jax.random.split(rng, 3)

        def cost_loss_fn(cost_params):
            expert_cost = cost_apply_fn(cost_params, exp.obs, exp.action)
            union_cost  = cost_apply_fn(cost_params, uni.obs, uni.action)

            # Minimax discriminator loss
            disc_loss = -(
                jnp.mean(jnp.log(jax.nn.sigmoid(expert_cost) + EPS2)) +
                jnp.mean(jnp.log(1.0 - jax.nn.sigmoid(union_cost) + EPS2))
            )

            # Gradient penalty on cost (WGAN-style)
            # Interpolate in raw (obs, action) space; split inside the per-sample fn
            exp_sa   = jnp.concatenate([exp.obs, exp.action], axis=-1)
            uni_sa   = jnp.concatenate([uni.obs, uni.action], axis=-1)
            alpha_gp = jax.random.uniform(rng_gp_alpha, shape=(args.batch_size, 1))
            perm     = jax.random.permutation(rng_gp_perm, args.batch_size)
            mixed1   = alpha_gp * exp_sa + (1.0 - alpha_gp) * uni_sa
            mixed2   = alpha_gp * uni_sa[perm] + (1.0 - alpha_gp) * uni_sa
            mixed    = jnp.concatenate([mixed1, mixed2], axis=0)

            def _cost_transform_single(sa):
                obs_    = sa[:obs_dim]
                action_ = sa[obs_dim:]
                c = cost_apply_fn(cost_params, obs_[None], action_[None]).squeeze()
                return jnp.log(1.0 / (jax.nn.sigmoid(c) + EPS2) - 1.0 + EPS2)

            grads       = jax.vmap(jax.grad(_cost_transform_single))(mixed) + EPS
            grad_penalty = jnp.mean(
                jnp.square(jnp.linalg.norm(grads, axis=-1, keepdims=True) - 1.0)
            )
            return disc_loss + args.grad_reg_coeff_cost * grad_penalty

        cost_loss, cost_grad = jax.value_and_grad(cost_loss_fn)(agent_state.cost.params)
        agent_state = agent_state._replace(
            cost=agent_state.cost.apply_gradients(grads=cost_grad)
        )

        # --- Precompute cost transform with stop_gradient (mirrors original tf.stop_gradient) ---
        union_cost_sg = cost_apply_fn(
            jax.lax.stop_gradient(agent_state.cost.params), uni.obs, uni.action
        )
        union_cost_transformed = jax.lax.stop_gradient(
            jnp.log(1.0 / (jax.nn.sigmoid(union_cost_sg) + EPS2) - 1.0 + EPS2)
        )

        # --- Update nu (stationary distribution ratio) ---
        rng, rng_gp_nu = jax.random.split(rng)

        def nu_loss_fn(nu_params):
            init_nu       = nu_apply_fn(nu_params, init_batch)
            union_nu      = nu_apply_fn(nu_params, uni.obs)
            union_next_nu = nu_apply_fn(nu_params, uni.next_obs)

            union_adv_nu = -union_cost_transformed + args.gamma * union_next_nu - union_nu

            # LogSumExp regularisation
            non_linear = non_expert_reg * jax.scipy.special.logsumexp(
                union_adv_nu / non_expert_reg
            )
            linear = (1.0 - args.gamma) * jnp.mean(init_nu)
            loss   = non_linear + linear

            # Gradient penalty on nu
            alpha_nu      = jax.random.uniform(rng_gp_nu, shape=(args.batch_size, 1))
            nu_inter      = alpha_nu * exp.obs + (1.0 - alpha_nu) * uni.obs
            nu_next_inter = alpha_nu * exp.next_obs + (1.0 - alpha_nu) * uni.next_obs
            nu_inter_all  = jnp.concatenate([uni.obs, nu_inter, nu_next_inter], axis=0)

            def _nu_single(obs):
                return nu_apply_fn(nu_params, obs[None]).squeeze()

            nu_grads     = jax.vmap(jax.grad(_nu_single))(nu_inter_all) + EPS
            nu_grad_penalty = jnp.mean(
                jnp.square(jnp.linalg.norm(nu_grads, axis=-1, keepdims=True))
            )
            return loss + args.grad_reg_coeff_nu * nu_grad_penalty, union_adv_nu

        (nu_loss, union_adv_nu), nu_grad = jax.value_and_grad(nu_loss_fn, has_aux=True)(
            agent_state.nu.params
        )
        agent_state = agent_state._replace(
            nu=agent_state.nu.apply_gradients(grads=nu_grad)
        )

        # --- Update actor (weighted behavioural cloning) ---
        def actor_loss_fn(actor_params):
            adv    = jax.lax.stop_gradient(union_adv_nu)
            weight = jnp.exp((adv - jnp.max(adv)) / non_expert_reg)
            weight = weight / jnp.mean(weight)

            def _log_prob(obs, action):
                # Clip to avoid atanh(±1) = ±inf inside the Tanh bijector
                action = jnp.clip(action, -1.0 + EPS, 1.0 - EPS)
                pi = actor_apply_fn(actor_params, obs)
                return pi.log_prob(action).sum()

            log_probs = jax.vmap(_log_prob)(uni.obs, uni.action)
            return -jnp.mean(weight * log_probs)

        actor_loss, actor_grad = jax.value_and_grad(actor_loss_fn)(agent_state.actor.params)
        agent_state = agent_state._replace(
            actor=agent_state.actor.apply_gradients(grads=actor_grad)
        )

        loss = {
            "cost_loss":  cost_loss,
            "nu_loss":    nu_loss,
            "actor_loss": actor_loss,
        }
        return (rng, agent_state), loss

    return _train_step


if __name__ == "__main__":
    # --- Parse arguments ---
    args = tyro.cli(Args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    rng  = jax.random.PRNGKey(args.seed)

    # --- Initialize logger ---
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type="train_agent",
        )

    # --- Load datasets ---
    expert_raw = d4rl.qlearning_dataset(gym.make(args.expert_dataset))
    union_raw  = d4rl.qlearning_dataset(gym.make(args.union_dataset))

    # Obs statistics from the union (imperfect) dataset — used by networks internally
    obs_mean = jnp.array(union_raw["observations"].mean(0), dtype=jnp.float32)
    obs_std  = jnp.array(union_raw["observations"].std(0),  dtype=jnp.float32)

    def _to_jnp(arr):
        return jnp.array(arr.astype(onp.float32))

    expert_data = Transition(
        obs     = _to_jnp(expert_raw["observations"]),
        action  = _to_jnp(expert_raw["actions"]),
        reward  = _to_jnp(expert_raw["rewards"]),
        next_obs= _to_jnp(expert_raw["next_observations"]),
        done    = _to_jnp(expert_raw["terminals"]),
    )

    # Union = imperfect dataset + expert demonstrations combined
    union_data = Transition(
        obs     = _to_jnp(onp.concatenate([union_raw["observations"],      expert_raw["observations"]],      axis=0)),
        action  = _to_jnp(onp.concatenate([union_raw["actions"],           expert_raw["actions"]],           axis=0)),
        reward  = _to_jnp(onp.concatenate([union_raw["rewards"],           expert_raw["rewards"]],           axis=0)),
        next_obs= _to_jnp(onp.concatenate([union_raw["next_observations"], expert_raw["next_observations"]], axis=0)),
        done    = _to_jnp(onp.concatenate([union_raw["terminals"],         expert_raw["terminals"]],         axis=0)),
    )

    # Episode-initial observations from union (for the (1-γ)·E[ν(s₀)] term)
    all_dones  = onp.concatenate([union_raw["terminals"], expert_raw["terminals"]], axis=0).astype(bool)
    init_mask  = onp.zeros(len(all_dones), dtype=bool)
    init_mask[0]  = True
    init_mask[1:] = all_dones[:-1]
    init_obs = union_data.obs[init_mask]

    print(f"Expert transitions : {len(expert_data.obs)}")
    print(f"Union  transitions : {len(union_data.obs)}")
    print(f"Init   states      : {len(init_obs)}")

    # --- Initialize environment and networks ---
    env         = gym.vector.make(args.union_dataset, num_envs=args.eval_workers)
    num_actions = env.single_action_space.shape[0]
    obs_dim     = env.single_observation_space.shape[0]

    dummy_obs    = jnp.zeros(obs_dim)
    dummy_action = jnp.zeros(num_actions)

    cost_net  = CostNetwork(obs_mean, obs_std)
    nu_net    = NuNetwork(obs_mean, obs_std)
    actor_net = TanhGaussianActor(num_actions, obs_mean, obs_std)

    rng, rng_cost, rng_nu, rng_actor = jax.random.split(rng, 4)
    agent_state = AgentTrainState(
        cost =create_train_state(args, rng_cost,  cost_net,  [dummy_obs, dummy_action]),
        nu   =create_train_state(args, rng_nu,    nu_net,    [dummy_obs]),
        actor=create_train_state(args, rng_actor, actor_net, [dummy_obs], lr=args.actor_lr),
    )

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(
        args, cost_net.apply, nu_net.apply, actor_net.apply,
        expert_data, union_data, init_obs,
    )

    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, agent_state),
            None,
            args.eval_interval,
        )

        # --- Evaluate agent ---
        rng, rng_eval = jax.random.split(rng)
        returns = eval_agent(args, rng_eval, env, agent_state)
        scores  = d4rl.get_normalized_score(args.union_dataset, returns) * 100.0

        # --- Log metrics ---
        step = (eval_idx + 1) * args.eval_interval
        print("Step:", step, f"\t Score: {scores.mean():.2f}")
        if args.log:
            log_dict = {
                "return":    returns.mean(),
                "score":     scores.mean(),
                "score_std": scores.std(),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
            }
            wandb.log(log_dict)

    # --- Evaluate final agent ---
    if args.eval_final_episodes > 0:
        final_iters = int(onp.ceil(args.eval_final_episodes / args.eval_workers))
        print(f"Evaluating final agent for {final_iters} iterations...")
        _rng = jax.random.split(rng, final_iters)
        rets   = onp.array([eval_agent(args, r, env, agent_state) for r in _rng])
        scores = d4rl.get_normalized_score(args.union_dataset, rets) * 100.0
        agg_fn = lambda x, k: {k: x, f"{k}_mean": x.mean(), f"{k}_std": x.std()}
        info   = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")

        os.makedirs("final_returns", exist_ok=True)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{args.algorithm}_{args.union_dataset}_{time_str}.npz"
        with open(os.path.join("final_returns", filename), "wb") as f:
            onp.savez_compressed(f, **info, args=asdict(args))

        if args.log:
            wandb.save(os.path.join("final_returns", filename))

    env.close()
    if args.log:
        wandb.finish()
