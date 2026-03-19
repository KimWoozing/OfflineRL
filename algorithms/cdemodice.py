"""C-DemoDICE: B-DemoDICE with an additional state-marginal discriminator.

Reward decomposition
---------------------
  r(s,a) = -log(1/(c_1(s,a)) - 1)   [state-action discriminator, same as B-DemoDICE]
           +log(1/(c_2(s))   - 1)   [state discriminator, corrects for state visitation]

Intuitively: c_1 estimates d_expert(s,a)/d_union(s,a) and c_2 estimates
d_expert(s)/d_union(s), so r ∝ log[d_expert(a|s)/d_union(a|s)] — how much
more expert-like the *action* is at that state, corrected for state coverage.

Pipeline per update step
------------------------
1a. c_1 discriminator (s,a) : minimax BCE + WGAN-GP
1b. c_2 discriminator (s)   : minimax BCE + WGAN-GP  (obs-only)
2.  Q target                : Polyak update
3.  Q  loss (Bellman)       : (r(s,a) + γ V(s') − Q(s,a))²
4.  V  loss (soft Bellman)  : E[V(s) + exp(Q_target(s,a) − V(s))]
5.  Actor (weighted BC)     : −E[exp(Q_target − V) · log π(a|s)]
"""

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

EPS  = jnp.finfo(jnp.float32).eps
EPS2 = 1e-3


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    gpu_id: int = 0
    expert_dataset: str = "hopper-expert-v2"
    union_dataset: str = "hopper-medium-v2"
    algorithm: str = "cdemodice"
    num_updates: int = 1_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    # --- Generic optimisation ---
    lr: float = 3e-4
    actor_lr: float = 3e-5
    q_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    polyak_step_size: float = 0.005
    # --- C-DemoDICE ---
    alpha: float = 1.0           # temperature α in V loss and actor weights
    exp_adv_clip: float = 100.0  # clip exp(Q−V) in actor loss
    grad_reg_coeff_cost: float = 10.0


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple("AgentTrainState", "cost state_cost dual_q dual_q_target value actor")
Transition = namedtuple("Transition", "obs action reward next_obs done")


class CostNetwork(nn.Module):
    """State-action discriminator c_1(s,a): high output = expert-like (s,a)."""
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


class StateCostNetwork(nn.Module):
    """State-only discriminator c_2(s): high output = expert-like state."""
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs):
        x = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class SoftQNetwork(nn.Module):
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


class DualQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=-1,
            axis_size=2,
        )
        return vmap_critic(self.obs_mean, self.obs_std)(obs, action)


class StateValueFunction(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, x):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class TanhGaussianActor(nn.Module):
    """Actor with deterministic eval mode (matches IQL convention)."""
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array
    log_std_max: float = 2.0
    log_std_min: float = -20.0

    @nn.compact
    def __call__(self, x, eval: bool = False):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        x = nn.tanh(x)
        if eval:
            return distrax.Deterministic(x)
        logstd = self.param(
            "logstd",
            init_fn=lambda key: jnp.zeros(self.num_actions, dtype=jnp.float32),
        )
        std = jnp.exp(jnp.clip(logstd, self.log_std_min, self.log_std_max))
        return distrax.Normal(x, std)


def create_train_state(args, rng, network, dummy_input, lr=None):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
    )


def eval_agent(args, rng, env, agent_state):
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    rng, _ = jax.random.split(rng)
    obs = env.reset()

    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs, eval=True)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action)

    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    while step < max_episode_steps and not returned.all():
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, args.eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, done, info = env.step(onp.array(action))
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
    args, cost_apply_fn, state_cost_apply_fn, q_apply_fn, value_apply_fn, actor_apply_fn,
    expert_data, union_data,
):
    """Return a JIT-compatible C-DemoDICE train step.

    cost_apply_fn       : (params, obs, action) -> scalar per sample
    state_cost_apply_fn : (params, obs)         -> scalar per sample
    q_apply_fn          : (params, obs, action) -> (batch, 2)  [dual Q]
    value_apply_fn      : (params, obs)         -> scalar per sample
    actor_apply_fn      : (params, obs)         -> distrax distribution
    """
    obs_dim = expert_data.obs.shape[-1]

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Sample batches ---
        rng, rng_expert, rng_union = jax.random.split(rng, 3)
        expert_idx = jax.random.randint(rng_expert, (args.batch_size,), 0, len(expert_data.obs))
        union_idx  = jax.random.randint(rng_union,  (args.batch_size,), 0, len(union_data.obs))
        exp = jax.tree_util.tree_map(lambda x: x[expert_idx], expert_data)
        uni = jax.tree_util.tree_map(lambda x: x[union_idx],  union_data)

        # ------------------------------------------------------------------ #
        # 1a. Update c_1 : state-action discriminator (same as B-DemoDICE)   #
        # ------------------------------------------------------------------ #
        rng, rng_gp_alpha, rng_gp_perm = jax.random.split(rng, 3)

        def cost_loss_fn(cost_params):
            expert_cost = cost_apply_fn(cost_params, exp.obs, exp.action)
            union_cost  = cost_apply_fn(cost_params, uni.obs, uni.action)

            disc_loss = -(
                jnp.mean(jnp.log(jax.nn.sigmoid(expert_cost) + EPS2)) +
                jnp.mean(jnp.log(1.0 - jax.nn.sigmoid(union_cost) + EPS2))
            )

            # WGAN gradient penalty over (obs, action) space
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

            grads        = jax.vmap(jax.grad(_cost_transform_single))(mixed) + EPS
            grad_penalty = jnp.mean(
                jnp.square(jnp.linalg.norm(grads, axis=-1, keepdims=True) - 1.0)
            )
            return disc_loss + args.grad_reg_coeff_cost * grad_penalty

        cost_loss, cost_grad = jax.value_and_grad(cost_loss_fn)(agent_state.cost.params)
        agent_state = agent_state._replace(
            cost=agent_state.cost.apply_gradients(grads=cost_grad)
        )

        # ------------------------------------------------------------------ #
        # 1b. Update c_2 : state-only discriminator                          #
        # ------------------------------------------------------------------ #
        rng, rng_sc_alpha, rng_sc_perm = jax.random.split(rng, 3)

        def state_cost_loss_fn(sc_params):
            expert_sc = state_cost_apply_fn(sc_params, exp.obs)
            union_sc  = state_cost_apply_fn(sc_params, uni.obs)

            disc_loss = -(
                jnp.mean(jnp.log(jax.nn.sigmoid(expert_sc) + EPS2)) +
                jnp.mean(jnp.log(1.0 - jax.nn.sigmoid(union_sc) + EPS2))
            )

            # WGAN gradient penalty over obs space only
            alpha_gp = jax.random.uniform(rng_sc_alpha, shape=(args.batch_size, 1))
            perm     = jax.random.permutation(rng_sc_perm, args.batch_size)
            mixed1   = alpha_gp * exp.obs + (1.0 - alpha_gp) * uni.obs
            mixed2   = alpha_gp * uni.obs[perm] + (1.0 - alpha_gp) * uni.obs
            mixed    = jnp.concatenate([mixed1, mixed2], axis=0)

            def _state_cost_transform_single(obs):
                c = state_cost_apply_fn(sc_params, obs[None]).squeeze()
                return jnp.log(1.0 / (jax.nn.sigmoid(c) + EPS2) - 1.0 + EPS2)

            grads        = jax.vmap(jax.grad(_state_cost_transform_single))(mixed) + EPS
            grad_penalty = jnp.mean(
                jnp.square(jnp.linalg.norm(grads, axis=-1, keepdims=True) - 1.0)
            )
            return disc_loss + args.grad_reg_coeff_cost * grad_penalty

        sc_loss, sc_grad = jax.value_and_grad(state_cost_loss_fn)(agent_state.state_cost.params)
        agent_state = agent_state._replace(
            state_cost=agent_state.state_cost.apply_gradients(grads=sc_grad)
        )

        # ------------------------------------------------------------------ #
        # Learned reward  r(s,a) = -log(1/c_1(s,a)-1) + log(1/c_2(s)-1)    #
        # ------------------------------------------------------------------ #
        union_cost_sg = cost_apply_fn(
            jax.lax.stop_gradient(agent_state.cost.params), uni.obs, uni.action
        )
        union_sc_sg = state_cost_apply_fn(
            jax.lax.stop_gradient(agent_state.state_cost.params), uni.obs
        )
        cost_transform_1 = -jnp.log(1.0 / (jax.nn.sigmoid(union_cost_sg) + EPS2) - 1.0 + EPS2)
        cost_transform_2 =  jnp.log(1.0 / (jax.nn.sigmoid(union_sc_sg)   + EPS2) - 1.0 + EPS2)
        r_sa = jax.lax.stop_gradient(cost_transform_1 + cost_transform_2)

        # ------------------------------------------------------------------ #
        # 2. Update Q target network (Polyak)                                #
        # ------------------------------------------------------------------ #
        updated_q_target_params = optax.incremental_update(
            agent_state.dual_q.params,
            agent_state.dual_q_target.params,
            args.polyak_step_size,
        )
        agent_state = agent_state._replace(
            dual_q_target=agent_state.dual_q_target.replace(
                step=agent_state.dual_q_target.step + 1,
                params=updated_q_target_params,
            )
        )

        # ------------------------------------------------------------------ #
        # 3. Update Q  :  (r(s,a) + γ V(s') − Q(s,a))²                     #
        # ------------------------------------------------------------------ #
        next_v = jax.lax.stop_gradient(
            value_apply_fn(agent_state.value.params, uni.next_obs)
        )
        q_target = r_sa + args.gamma * (1.0 - uni.done) * next_v

        def q_loss_fn(q_params):
            q_pred = q_apply_fn(q_params, uni.obs, uni.action)   # (batch, 2)
            return jnp.square(q_pred - jnp.expand_dims(q_target, -1)).mean()

        q_loss, q_grad = jax.value_and_grad(q_loss_fn)(agent_state.dual_q.params)
        agent_state = agent_state._replace(
            dual_q=agent_state.dual_q.apply_gradients(grads=q_grad)
        )

        # ------------------------------------------------------------------ #
        # 4. Update V  :  E[V(s) + exp(Q_target(s,a) − V(s))]              #
        # ------------------------------------------------------------------ #
        v_target = jax.lax.stop_gradient(
            q_apply_fn(agent_state.dual_q_target.params, uni.obs, uni.action).min(-1)
        )

        def value_loss_fn(value_params):
            v = value_apply_fn(value_params, uni.obs)
            adv_v = v_target - v
            # Log-partition (DICE) form: E[V(s)] + log E[exp(Q(s,a)-V(s)+1)]
            # return jnp.mean(v) + jax.nn.logsumexp(adv_v + 1.0) - jnp.log(args.batch_size)
            return jnp.mean(v + jnp.exp(adv_v))

        value_loss, value_grad = jax.value_and_grad(value_loss_fn)(agent_state.value.params)
        agent_state = agent_state._replace(
            value=agent_state.value.apply_gradients(grads=value_grad)
        )

        # ------------------------------------------------------------------ #
        # 5. Update actor  :  −E[exp(Q_target − V) · log π(a|s)]            #
        # ------------------------------------------------------------------ #
        v_sg  = jax.lax.stop_gradient(value_apply_fn(agent_state.value.params, uni.obs))
        adv   = v_target - v_sg   # Q_target(s,a) − V(s), both stop-grad

        def actor_loss_fn(actor_params):
            w = jnp.exp(adv).clip(max=args.exp_adv_clip)

            def _log_prob(obs, action):
                pi = actor_apply_fn(actor_params, obs)
                return pi.log_prob(action).sum()

            log_probs = jax.vmap(_log_prob)(uni.obs, uni.action)
            return -jnp.mean(w * log_probs)

        actor_loss, actor_grad = jax.value_and_grad(actor_loss_fn)(agent_state.actor.params)
        agent_state = agent_state._replace(
            actor=agent_state.actor.apply_gradients(grads=actor_grad)
        )

        loss = {
            "cost_loss":       cost_loss,
            "state_cost_loss": sc_loss,
            "q_loss":          q_loss,
            "value_loss":      value_loss,
            "actor_loss":      actor_loss,
            "reward_max":      r_sa.max(),
            "reward_min":      r_sa.min(),
            "reward_mean":     r_sa.mean(),
        }
        return (rng, agent_state), loss

    return _train_step


if __name__ == "__main__":
    args = tyro.cli(Args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    rng  = jax.random.PRNGKey(args.seed)

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

    obs_mean = jnp.array(union_raw["observations"].mean(0), dtype=jnp.float32)
    obs_std  = jnp.array(union_raw["observations"].std(0),  dtype=jnp.float32)

    def _jnp(arr):
        return jnp.array(arr.astype(onp.float32))

    expert_data = Transition(
        obs     = _jnp(expert_raw["observations"]),
        action  = _jnp(expert_raw["actions"]),
        reward  = _jnp(expert_raw["rewards"]),
        next_obs= _jnp(expert_raw["next_observations"]),
        done    = _jnp(expert_raw["terminals"]),
    )
    union_data = Transition(
        obs     = _jnp(onp.concatenate([union_raw["observations"],      expert_raw["observations"]],      axis=0)),
        action  = _jnp(onp.concatenate([union_raw["actions"],           expert_raw["actions"]],           axis=0)),
        reward  = _jnp(onp.concatenate([union_raw["rewards"],           expert_raw["rewards"]],           axis=0)),
        next_obs= _jnp(onp.concatenate([union_raw["next_observations"], expert_raw["next_observations"]], axis=0)),
        done    = _jnp(onp.concatenate([union_raw["terminals"],         expert_raw["terminals"]],         axis=0)),
    )

    print(f"Expert transitions : {len(expert_data.obs)}")
    print(f"Union  transitions : {len(union_data.obs)}")

    # --- Initialize environment and networks ---
    env         = gym.vector.make(args.union_dataset, num_envs=args.eval_workers)
    num_actions = env.single_action_space.shape[0]
    obs_dim     = env.single_observation_space.shape[0]

    dummy_obs    = jnp.zeros(obs_dim)
    dummy_action = jnp.zeros(num_actions)

    cost_net       = CostNetwork(obs_mean, obs_std)
    state_cost_net = StateCostNetwork(obs_mean, obs_std)
    q_net          = DualQNetwork(obs_mean, obs_std)
    value_net      = StateValueFunction(obs_mean, obs_std)
    actor_net      = TanhGaussianActor(num_actions, obs_mean, obs_std)

    rng, rng_cost, rng_sc, rng_q, rng_value, rng_actor = jax.random.split(rng, 6)
    agent_state = AgentTrainState(
        cost          = create_train_state(args, rng_cost,  cost_net,       [dummy_obs, dummy_action]),
        state_cost    = create_train_state(args, rng_sc,    state_cost_net, [dummy_obs]),
        dual_q        = create_train_state(args, rng_q,     q_net,          [dummy_obs, dummy_action], lr=args.q_lr),
        dual_q_target = create_train_state(args, rng_q,     q_net,          [dummy_obs, dummy_action], lr=args.q_lr),
        value         = create_train_state(args, rng_value, value_net,      [dummy_obs]),
        actor         = create_train_state(args, rng_actor, actor_net,      [dummy_obs], lr=args.actor_lr),
    )

    _agent_train_step_fn = make_train_step(
        args, cost_net.apply, state_cost_net.apply, q_net.apply, value_net.apply, actor_net.apply,
        expert_data, union_data,
    )

    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, agent_state),
            None,
            args.eval_interval,
        )

        rng, rng_eval = jax.random.split(rng)
        returns = eval_agent(args, rng_eval, env, agent_state)
        scores  = d4rl.get_normalized_score(args.union_dataset, returns) * 100.0

        step = (eval_idx + 1) * args.eval_interval
        print(
            f"Step: {step}"
            f"\t Score: {scores.mean():.2f}"
            f"\t Q: {loss['q_loss'][-1]:.3f}"
            f"\t V: {loss['value_loss'][-1]:.3f}"
            f"\t r=[{loss['reward_min'][-1]:.2f}, {loss['reward_max'][-1]:.2f}]"
            f"\t r_mean: {loss['reward_mean'][-1]:.3f}"
        )
        if args.log:
            wandb.log({
                "return":      returns.mean(),
                "score":       scores.mean(),
                "score_std":   scores.std(),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
            })

    # --- Evaluate final agent ---
    if args.eval_final_episodes > 0:
        final_iters = int(onp.ceil(args.eval_final_episodes / args.eval_workers))
        print(f"Evaluating final agent for {final_iters} iterations...")
        _rng  = jax.random.split(rng, final_iters)
        rets  = onp.array([eval_agent(args, r, env, agent_state) for r in _rng])
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
