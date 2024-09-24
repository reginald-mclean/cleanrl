# ruff: noqa: E402
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool  # type: ignore
from functools import partial
from pathlib import Path
from typing import Deque, NamedTuple, Optional, Tuple, Union

import distrax  # type: ignore
import flax
import flax.linen as nn
import gymnasium as gym  # type: ignore
import jax
import jax.numpy as jnp
import metaworld  # type: ignore
import numpy as np
import numpy.typing as npt
import optax  # type: ignore
import orbax.checkpoint as ocp  # type: ignore
from flax.training.train_state import TrainState
from jax.typing import ArrayLike
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers_metaworld import MultiTaskReplayBuffer
from cleanrl_utils.env_setup_metaworld import (
    checkpoint_envs,
    load_env_checkpoints,
    make_envs,
)
from cleanrl_utils.evals.metaworld_jax_eval import evaluation


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Meta-World Benchmarking (Updated)",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="reggies-phd-research",
        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the checkpoint directory")
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="whether to resume the experiment")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs"), help="the checkpoint directory")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MT10", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(2e7),
        help="total timesteps of the experiments *across all tasks*, the timesteps per task are this value / num_tasks")
    parser.add_argument("--max-episode-steps", type=int, default=None,
        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=1280,
                        help="the total size of the batch to sample from the replay memory. Must be divisible by number of tasks")
    parser.add_argument("--learning-starts", type=int, default=4e3, help="timestep to start learning")
    parser.add_argument("--evaluation-frequency", type=int, default=200_000 / 500,
        help="every how many episodes to evaluate the agent. Evaluation is disabled if 0.")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes to run per evaluation")
    # SAC
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0,
        help="the value to clip the gradient norm to. Disabled if 0. Not applied to alpha gradients.")
    parser.add_argument("--actor-network", type=str, default="400,400,400", help="The architecture of the actor network")
    parser.add_argument("--critic-network", type=str, default="400,400,400", help="The architecture of the critic network")
    args = parser.parse_args()
    # fmt: on
    return args


def split_obs_task_id(obs: Union[jax.Array, npt.NDArray], num_tasks: int) -> Tuple[ArrayLike, ArrayLike]:
    return obs[..., :-num_tasks], obs[..., -num_tasks:]


class Batch(NamedTuple):
    observations: ArrayLike
    actions: ArrayLike
    rewards: ArrayLike
    next_observations: ArrayLike
    dones: ArrayLike
    task_ids: ArrayLike


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

    return _init


class Actor(nn.Module):
    num_actions: int
    num_tasks: int
    hidden_dims: int = "400,400,400"

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    @nn.compact
    def __call__(self, x: jax.Array, task_idx):
        hidden_lst = [int(dim) for dim in self.hidden_dims.split(",")]
        for i, h_size in enumerate(hidden_lst):
            x = nn.Dense(
                h_size * self.num_tasks if i == len(hidden_lst) - 1 else h_size,
                kernel_init=nn.initializers.he_uniform(),
                bias_init=nn.initializers.constant(0.1),
            )(x)
            x = nn.relu(x)

        # extract the task ids from the one-hot encodings of the observations
        indices = jnp.arange(hidden_lst[-1])[None, :] + (task_idx.argmax(1) * hidden_lst[-1])[..., None]
        x = jnp.take_along_axis(x, indices, axis=1)

        log_sigma = nn.Dense(
            self.num_actions,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)
        mu = nn.Dense(
            self.num_actions,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)
        log_sigma = jnp.clip(log_sigma, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return distrax.Transformed(
            distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_sigma)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )


@jax.jit
def sample_action(
    actor: TrainState,
    obs: ArrayLike,
    task_ids: ArrayLike,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, obs, task_ids)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def sample_and_log_prob(
    actor: TrainState,
    actor_params: flax.core.FrozenDict,
    obs: ArrayLike,
    task_ids: ArrayLike,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor_params, obs, task_ids)
    action, log_prob = dist.sample_and_log_prob(seed=action_key)
    return action, log_prob, key


@jax.jit
def get_deterministic_action(
    actor: TrainState,
    obs: ArrayLike,
    task_ids: ArrayLike,
) -> jax.Array:
    dist = actor.apply_fn(actor.params, obs, task_ids)
    return jnp.tanh(dist.distribution.loc)


class Critic(nn.Module):
    hidden_dims: int = "400,400"
    num_tasks: int = 1

    @nn.compact
    def __call__(self, state, action, task_idx):
        x = jnp.hstack([state, action])
        hidden_lst = [int(dim) for dim in self.hidden_dims.split(",")]
        for i, h_size in enumerate(hidden_lst):
            x = nn.Dense(
                h_size * self.num_tasks if i == len(hidden_lst) - 1 else h_size,
                kernel_init=nn.initializers.he_uniform(),
                bias_init=nn.initializers.constant(0.1),
            )(x)
            x = nn.relu(x)
        # extract the task ids from the one-hot encodings of the observations
        indices = jnp.arange(hidden_lst[-1])[None, :] + (task_idx.argmax(1) * hidden_lst[-1])[..., None]
        x = jnp.take_along_axis(x, indices, axis=1)

        return nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))(x)


class VectorCritic(nn.Module):
    n_critics: int = 2
    num_tasks: int = 1
    hidden_dims: int = "400,400,400"

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array, task_idx) -> jax.Array:
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        return vmap_critic(self.hidden_dims, self.num_tasks)(state, action, task_idx)


class CriticTrainState(TrainState):
    target_params: Optional[flax.core.FrozenDict] = None


@jax.jit
def get_alpha(log_alpha: jax.Array, task_ids: jax.Array) -> jax.Array:
    return jnp.exp(task_ids @ log_alpha.reshape(-1, 1))


class Agent:
    actor: TrainState
    critic: CriticTrainState
    alpha_train_state: TrainState
    target_entropy: float

    def __init__(
        self,
        init_obs: jax.Array,
        num_tasks: int,
        action_space: gym.spaces.Box,
        policy_lr: float,
        q_lr: float,
        gamma: float,
        clip_grad_norm: float,
        init_key: jax.Array,
    ):
        self._action_space = action_space
        self._num_tasks = num_tasks
        self._gamma = gamma

        just_obs, task_id = jax.device_put(split_obs_task_id(init_obs, num_tasks))
        random_action = jnp.array([self._action_space.sample() for _ in range(init_obs.shape[0])])

        def _make_optimizer(lr: float, max_grad_norm: float = 0.0):
            optim = optax.adam(learning_rate=lr)
            if max_grad_norm != 0:
                optim = optax.chain(
                    optax.clip_by_global_norm(max_grad_norm),
                    optim,
                )
            return optim

        actor_network = Actor(
            num_actions=int(np.prod(self._action_space.shape)),
            num_tasks=num_tasks,
            hidden_dims=args.actor_network,
        )
        key, actor_init_key = jax.random.split(init_key)
        self.actor = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network.init(actor_init_key, just_obs, task_id),
            tx=_make_optimizer(policy_lr, clip_grad_norm),
        )

        _, qf_init_key = jax.random.split(key, 2)
        vector_critic_net = VectorCritic(num_tasks=num_tasks, hidden_dims=args.critic_network)
        self.critic = CriticTrainState.create(
            apply_fn=vector_critic_net.apply,
            params=vector_critic_net.init(qf_init_key, just_obs, random_action, task_id),
            target_params=vector_critic_net.init(qf_init_key, just_obs, random_action, task_id),
            tx=_make_optimizer(q_lr, clip_grad_norm),
        )

        self.alpha_train_state = TrainState.create(
            apply_fn=get_alpha,
            params=jnp.zeros(NUM_TASKS),  # Log alpha
            tx=_make_optimizer(q_lr, max_grad_norm=0.0),
        )
        self.target_entropy = -np.prod(self._action_space.shape).item()

    def get_action_train(self, obs: npt.NDArray[np.float64], key: jax.Array) -> Tuple[npt.NDArray[np.float64], jax.Array]:
        state, task_id = split_obs_task_id(obs, self._num_tasks)
        actions, key = sample_action(self.actor, state, task_id, key)
        return jax.device_get(actions), key

    def get_action_eval(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        state, task_id = split_obs_task_id(obs, self._num_tasks)
        actions = get_deterministic_action(self.actor, state, task_id)
        return jax.device_get(actions)

    @staticmethod
    @jax.jit
    def soft_update(tau: float, critic_state: CriticTrainState) -> CriticTrainState:
        qf_state = critic_state.replace(
            target_params=optax.incremental_update(critic_state.params, critic_state.target_params, tau)
        )
        return qf_state

    def soft_update_target_networks(self, tau: float):
        self.critic = self.soft_update(tau, self.critic)

    def checkpoint(self) -> dict:
        return {
            "actor": self.actor,
            "critic": self.critic,
            "alpha": self.alpha_train_state,
            "target_entropy": self.target_entropy,
        }

    def load_checkpoint(self, ckpt: dict) -> None:
        self.actor = ckpt["actor"]
        self.critic = ckpt["critic"]
        self.alpha_train_state = ckpt["alpha"]
        self.target_entropy = ckpt["target_entropy"]


@partial(jax.jit, static_argnames=("gamma", "target_entropy"))
def update(
    actor: TrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    batch: Batch,
    target_entropy: float,
    gamma: float,
    key: jax.Array,
) -> Tuple[Tuple[TrainState, CriticTrainState, TrainState], dict, jax.Array]:
    next_actions, next_action_log_probs, key = sample_and_log_prob(
        actor, actor.params, batch.next_observations, batch.task_ids, key
    )
    q_values = critic.apply_fn(critic.target_params, batch.next_observations, next_actions, batch.task_ids)

    def critic_loss(params: flax.core.FrozenDict, alpha_val: jax.Array):
        min_qf_next_target = jnp.min(q_values, axis=0) - alpha_val * next_action_log_probs.reshape(-1, 1)
        next_q_value = jax.lax.stop_gradient(batch.rewards + (1 - batch.dones) * gamma * min_qf_next_target)

        q_pred = critic.apply_fn(params, batch.observations, batch.actions, batch.task_ids)
        return 0.5 * ((q_pred - next_q_value) ** 2).mean(axis=1).sum(), q_pred.mean()

    def update_critic(_critic: CriticTrainState, alpha_val: jax.Array) -> Tuple[CriticTrainState, dict]:
        (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(critic_loss, has_aux=True)(_critic.params, alpha_val)
        _critic = _critic.apply_gradients(grads=critic_grads)
        return _critic, {
            "losses/qf_values": qf_values,
            "losses/qf_loss": critic_loss_value,
        }

    def alpha_loss(params: jax.Array, log_probs: jax.Array):
        log_alpha = batch.task_ids @ params.reshape(-1, 1)
        return (-log_alpha * (log_probs.reshape(-1, 1) + target_entropy)).mean()

    def update_alpha(_alpha: TrainState, log_probs: jax.Array) -> Tuple[TrainState, jax.Array, dict]:
        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(_alpha.params, log_probs)
        _alpha = _alpha.apply_gradients(grads=alpha_grads)
        alpha_vals = _alpha.apply_fn(_alpha.params, batch.task_ids)
        return (
            _alpha,
            alpha_vals,
            {
                "losses/alpha_loss": alpha_loss_value,
                "alpha": jnp.exp(_alpha.params).sum(),
            },  # type: ignore
        )

    key, actor_loss_key = jax.random.split(key)

    def actor_loss(params: flax.core.FrozenDict):
        action_samples, log_probs, _ = sample_and_log_prob(actor, params, batch.observations, batch.task_ids, actor_loss_key)
        _alpha, _alpha_val, alpha_logs = update_alpha(alpha, log_probs)
        _alpha_val = jax.lax.stop_gradient(_alpha_val)
        _critic, critic_logs = update_critic(critic, _alpha_val)
        logs = {**alpha_logs, **critic_logs}

        q_values = _critic.apply_fn(_critic.params, batch.observations, action_samples, batch.task_ids)
        min_qf_values = jnp.min(q_values, axis=0)
        return (_alpha_val * log_probs.reshape(-1, 1) - min_qf_values).mean(), (
            _alpha,
            _critic,
            logs,
        )

    (actor_loss_value, (alpha, critic, logs)), actor_grads = jax.value_and_grad(actor_loss, has_aux=True)(actor.params)
    actor = actor.apply_gradients(grads=actor_grads)

    return (actor, critic, alpha), {**logs, "losses/actor_loss": actor_loss_value}, key


# Checkpointing utils
def get_ckpt_save_args(agent, buffer, envs, key, total_steps, global_step, episodes_ended):
    rb_ckpt = buffer.checkpoint()
    return ocp.args.Composite(
        agent=ocp.args.PyTreeSave(agent.checkpoint()),
        buffer=ocp.args.Composite(
            data=ocp.args.PyTreeSave(rb_ckpt["data"]),
            rng_state=ocp.args.JsonSave(rb_ckpt["rng_state"]),
        ),
        env_states=ocp.args.JsonSave(checkpoint_envs(envs)),
        rngs=ocp.args.Composite(
            rng_key=ocp.args.JaxRandomKeySave(key),
            python_rng_state=ocp.args.PyTreeSave(random.getstate()),
            global_numpy_rng_state=ocp.args.NumpyRandomKeySave(np.random.get_state()),
        ),
        metadata=ocp.args.JsonSave(
            {
                "total_steps": total_steps,
                "global_step": global_step,
                "episodes_ended": episodes_ended,
            }
        ),
    )


def get_ckpt_restore_args(agent, buffer):
    rb_ckpt = buffer.checkpoint()
    return ocp.args.Composite(
        agent=ocp.args.PyTreeRestore(agent.checkpoint()),
        buffer=ocp.args.Composite(
            data=ocp.args.PyTreeRestore(rb_ckpt["data"]),
            rng_state=ocp.args.JsonRestore(),
        ),
        env_states=ocp.args.JsonRestore(),
        rngs=ocp.args.Composite(
            rng_key=ocp.args.JaxRandomKeyRestore(),
            python_rng_state=ocp.args.PyTreeRestore(random.getstate()),
            global_numpy_rng_state=ocp.args.NumpyRandomKeyRestore(),
        ),
        metadata=ocp.args.JsonRestore(),
    )


# Training loop
if __name__ == "__main__":
    if jax.device_count("gpu") < 1:
        raise RuntimeError("No GPUs found, aborting. Deviecs: %s" % jax.devices())

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            id=run_name,
            monitor_gym=True,
            save_code=True,
            resume="allow",
        )
    writer = SummaryWriter(args.checkpoint_dir / f"{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # env setup
    if args.env_id == "MT10":
        benchmark = metaworld.MT10(seed=args.seed)
    elif args.env_id == "MT50":
        benchmark = metaworld.MT50(seed=args.seed)
    else:
        benchmark = metaworld.MT1(args.env_id, seed=args.seed)

    use_one_hot_wrapper = True if "MT10" in args.env_id or "MT50" in args.env_id else False

    envs = make_envs(
        benchmark,
        args.seed,
        args.max_episode_steps,
        use_one_hot=use_one_hot_wrapper,
    )

    NUM_TASKS = len(benchmark.train_classes)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # agent setup
    rb = MultiTaskReplayBuffer(
        total_capacity=args.buffer_size,
        num_tasks=NUM_TASKS,
        envs=envs,
        use_torch=False,
        seed=args.seed,
    )

    global_episodic_return: Deque[float] = deque([], maxlen=20 * NUM_TASKS)
    global_episodic_length: Deque[int] = deque([], maxlen=20 * NUM_TASKS)

    obs, _ = envs.reset()
    has_autoreset = np.full((envs.num_envs,), False)
    start_step, episodes_ended = 0, 0

    key, agent_init_key = jax.random.split(key)
    agent = Agent(
        init_obs=obs,
        num_tasks=NUM_TASKS,
        action_space=envs.single_action_space,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        gamma=args.gamma,
        clip_grad_norm=args.clip_grad_norm,
        init_key=key,
    )

    if args.save_model:  # Orbax checkpoints
        ckpt_manager = ocp.CheckpointManager(
            Path(args.checkpoint_dir / f"{run_name}/checkpoints").absolute(),
            item_names=(
                "agent",
                "buffer",
                "env_states",
                "rngs",
                "metadata",
            ),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=5,
                create=True,
                best_fn=lambda x: x["charts/mean_success_rate"],
            ),
        )

        if args.resume and ckpt_manager.latest_step() is not None:
            ckpt = ckpt_manager.restore(ckpt_manager.latest_step(), args=get_ckpt_restore_args(agent, rb))

            agent.load_checkpoint(ckpt["agent"])
            rb.load_checkpoint(ckpt["buffer"])
            load_env_checkpoints(envs, ckpt["env_states"])

            key = ckpt["rngs"]["rng_key"]
            random.setstate(ckpt["rngs"]["python_rng_state"])
            np.random.set_state(ckpt["rngs"]["global_numpy_rng_state"])

            start_step = ckpt["metadata"]["global_step"]
            episodes_ended = ckpt["metadata"]["episodes_ended"]

            print(f"Loaded checkpoint at step {start_step}")

    env_names = list(benchmark.train_classes.keys())

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    for global_step in range(start_step, args.total_timesteps // envs.num_envs):
        total_steps = global_step * envs.num_envs

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, key = agent.get_action_train(obs, key)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Store data in the buffer
        if not has_autoreset.any():
            rb.add(obs, next_obs, actions, rewards, terminations)
        elif has_autoreset.any() and not has_autoreset.all():
            # TODO handle the case where only some envs have autoreset
            raise NotImplementedError("Only some envs resetting isn't implemented at the moment.")

        has_autoreset = np.logical_or(terminations, truncations)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for i, env_ended in enumerate(has_autoreset):
            if env_ended:
                global_episodic_return.append(infos["episode"]["r"][i])
                global_episodic_length.append(infos["episode"]["l"][i])
                episodes_ended += 1

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 500 == 0 and global_episodic_return:
            print(f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}")
            writer.add_scalar(
                "charts/mean_episodic_return",
                np.mean(list(global_episodic_return)),
                total_steps,
            )
            writer.add_scalar(
                "charts/mean_episodic_length",
                np.mean(list(global_episodic_length)),
                total_steps,
            )

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # Sample a batch from replay buffer
            data = rb.sample(args.batch_size)
            observations, task_ids = split_obs_task_id(data.observations, NUM_TASKS)
            next_observations, _ = split_obs_task_id(data.next_observations, NUM_TASKS)
            batch = Batch(
                observations,
                data.actions,
                data.rewards,
                next_observations,
                data.dones,
                task_ids,
            )

            # Update agent
            (agent.actor, agent.critic, agent.alpha_train_state), logs, key = update(
                agent.actor,
                agent.critic,
                agent.alpha_train_state,
                batch,
                agent.target_entropy,
                args.gamma,
                key,
            )
            logs = jax.device_get(logs)

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                agent.soft_update_target_networks(args.tau)

            # Logging
            if global_step % 100 == 0:
                sps_steps = (global_step - start_step) * envs.num_envs
                for _key, value in logs.items():
                    writer.add_scalar(_key, value, sps_steps)
                print("SPS:", int(sps_steps / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(sps_steps / (time.time() - start_time)),
                    total_steps,
                )

            # Evaluation
            if (
                args.evaluation_frequency > 0
                and episodes_ended % args.evaluation_frequency == 0
                and has_autoreset.any()
                and global_step > 0
            ):
                envs.call("toggle_terminate_on_success", True)
                (eval_success_rate, eval_returns, eval_success_per_task,) = evaluation(
                    agent=agent,
                    eval_envs=envs,
                    num_episodes=args.evaluation_num_episodes,
                )
                envs.call("toggle_terminate_on_success", False)
                eval_metrics = {
                    "charts/mean_success_rate": float(eval_success_rate),
                    "charts/mean_evaluation_return": float(eval_returns),
                } | {
                    f"charts/{env_name}_success_rate": float(eval_success_per_task[i])
                    for i, (env_name, _) in enumerate(benchmark.train_classes.items())
                }

                for k, v in eval_metrics.items():
                    writer.add_scalar(k, v, total_steps)
                print(
                    f"total_steps={total_steps}, mean evaluation success rate: {eval_success_rate:.4f}"
                    + f" return: {eval_returns:.4f}"
                )

                # Reset envs again to exit eval mode
                obs, _ = envs.reset()

                # Checkpointing
                if args.save_model:
                    if not has_autoreset.all():
                        raise NotImplementedError(
                            "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                        )

                    ckpt_manager.save(
                        total_steps,
                        args=get_ckpt_save_args(
                            agent,
                            rb,
                            envs,
                            key,
                            total_steps,
                            global_step,
                            episodes_ended,
                        ),
                        metrics=eval_metrics,
                    )

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()
    if args.save_model:
        ckpt_manager.wait_until_finished()
        ckpt_manager.close()
