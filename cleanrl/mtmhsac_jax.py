# ruff: noqa: E402
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from functools import partial
from typing import Deque, NamedTuple, Optional, Tuple, Union
import sys
sys.path.append('/mnt/nvme/cleanrl')

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.1"

import distrax
import flax
import flax.linen as nn
import gymnasium as gym  # type: ignore
import jax
import jax.numpy as jnp
import metaworld  # type: ignore
import numpy as np
import numpy.typing as npt
import optax  # type: ignore
import orbax.checkpoint  # type: ignore
from cleanrl_utils.buffers_metaworld import MultiTaskReplayBuffer
from cleanrl_utils.evals.metaworld_jax_eval import evaluation
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax.typing import ArrayLike
from cleanrl_utils.env_setup_metaworld import make_envs, make_eval_envs
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Reward Smoothing",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="reggies-phd-research",
        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MT10", help="the id of the environment")
    parser.add_argument("--reward-version", type=str, default="v1", help="the reward function of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(2e7),
        help="total timesteps of the experiments *across all tasks*, the timesteps per task are this value / num_tasks")
    parser.add_argument("--max-episode-steps", type=int, default=500,
        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=1280,
                        help="the total size of the batch to sample from the replay memory. Must be divisible by number of tasks")
    parser.add_argument("--learning-starts", type=int, default=4e3, help="timestep to start learning")
    parser.add_argument("--evaluation-frequency", type=int, default=200_000,
        help="every how many timesteps to evaluate the agent. Evaluation is disabled if 0.")
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
    parser.add_argument('--gradient-steps', type=int, default=500)

    # reward smoothing
    parser.add_argument("--reward-filter", type=str, default=None)
    parser.add_argument('--filter-mode', type=str, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--kernel-type', type=str, default=None)

    # reward normalization
    parser.add_argument('--normalize-rewards', type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    # fmt: on
    return args


def split_obs_task_id(
    obs: Union[jax.Array, npt.NDArray], num_tasks: int
) -> Tuple[ArrayLike, ArrayLike]:
    return obs[..., :-num_tasks], obs[..., -num_tasks:]


class Batch(NamedTuple):
    observations: ArrayLike
    actions: ArrayLike
    rewards: ArrayLike
    next_observations: ArrayLike
    dones: ArrayLike
    task_ids: ArrayLike


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

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
        indices = (
            jnp.arange(hidden_lst[-1])[None, :]
            + (task_idx.argmax(1) * hidden_lst[-1])[..., None]
        )
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
        return TanhNormal(mu, jnp.exp(log_sigma))


@jax.jit
def sample_action(
    actor: TrainState,
    obs: ArrayLike,
    task_ids: ArrayLike,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.random.PRNGKeyArray]:
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
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.Array, jax.random.PRNGKeyArray]:
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
    return dist.mean()


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
        indices = (
            jnp.arange(hidden_lst[-1])[None, :]
            + (task_idx.argmax(1) * hidden_lst[-1])[..., None]
        )
        x = jnp.take_along_axis(x, indices, axis=1)

        return nn.Dense(
            1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3)
        )(x)


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
        init_key: jax.random.PRNGKeyArray,
    ):
        self._action_space = action_space
        self._num_tasks = num_tasks
        self._gamma = gamma

        just_obs, task_id = jax.device_put(split_obs_task_id(init_obs, num_tasks))
        random_action = jnp.array(
            [self._action_space.sample() for _ in range(init_obs.shape[0])]
        )

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
        vector_critic_net = VectorCritic(
            num_tasks=num_tasks, hidden_dims=args.critic_network
        )
        self.critic = CriticTrainState.create(
            apply_fn=vector_critic_net.apply,
            params=vector_critic_net.init(
                qf_init_key, just_obs, random_action, task_id
            ),
            target_params=vector_critic_net.init(
                qf_init_key, just_obs, random_action, task_id
            ),
            tx=_make_optimizer(q_lr, clip_grad_norm),
        )

        self.alpha_train_state = TrainState.create(
            apply_fn=get_alpha,
            params=jnp.zeros(NUM_TASKS),  # Log alpha
            tx=_make_optimizer(q_lr, max_grad_norm=0.0),
        )
        self.target_entropy = -np.prod(self._action_space.shape).item()

    def get_action_train(
        self, obs: np.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[np.ndarray, jax.random.PRNGKeyArray]:
        state, task_id = split_obs_task_id(obs, self._num_tasks)
        actions, key = sample_action(self.actor, state, task_id, key)
        return jax.device_get(actions), key

    def get_action_eval(self, obs: np.ndarray) -> np.ndarray:
        state, task_id = split_obs_task_id(obs, self._num_tasks)
        actions = get_deterministic_action(self.actor, state, task_id)
        return jax.device_get(actions)

    @staticmethod
    @jax.jit
    def soft_update(tau: float, critic_state: CriticTrainState) -> CriticTrainState:
        qf_state = critic_state.replace(
            target_params=optax.incremental_update(
                critic_state.params, critic_state.target_params, tau
            )
        )
        return qf_state

    def soft_update_target_networks(self, tau: float):
        self.critic = self.soft_update(tau, self.critic)

    def get_ckpt(self) -> dict:
        return {
            "actor": self.actor,
            "critic": self.critic,
            "alpha": self.alpha_train_state,
            "target_entropy": self.target_entropy,
        }


@partial(jax.jit, static_argnames=("gamma", "target_entropy"))
def update(
    actor: TrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    batch: Batch,
    target_entropy: float,
    gamma: float,
    key: jax.random.PRNGKeyArray,
) -> Tuple[
    Tuple[TrainState, CriticTrainState, TrainState], dict, jax.random.PRNGKeyArray
]:
    next_actions, next_action_log_probs, key = sample_and_log_prob(
        actor, actor.params, batch.next_observations, batch.task_ids, key
    )
    q_values = critic.apply_fn(
        critic.target_params, batch.next_observations, next_actions, batch.task_ids
    )

    def critic_loss(params: flax.core.FrozenDict, alpha_val: jax.Array):
        min_qf_next_target = jnp.min(q_values, axis=0).reshape(
            -1, 1
        ) - alpha_val * next_action_log_probs.sum(-1).reshape(-1, 1)
        next_q_value = jax.lax.stop_gradient(
            batch.rewards + (1 - batch.dones) * gamma * min_qf_next_target
        )

        q_pred = critic.apply_fn(
            params, batch.observations, batch.actions, batch.task_ids
        )
        return 0.5 * ((q_pred - next_q_value) ** 2).mean(axis=1).sum(), q_pred.mean()

    def update_critic(
        _critic: CriticTrainState, alpha_val: jax.Array
    ) -> Tuple[CriticTrainState, dict]:
        (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
            critic_loss, has_aux=True
        )(_critic.params, alpha_val)
        _critic = _critic.apply_gradients(grads=critic_grads)
        return _critic, {
            "losses/qf_values": qf_values,
            "losses/qf_loss": critic_loss_value,
        }

    def alpha_loss(params: jax.Array, log_probs: jax.Array):
        log_alpha = batch.task_ids @ params.reshape(-1, 1)
        return (-log_alpha * (log_probs.sum(-1).reshape(-1, 1) + target_entropy)).mean()

    def update_alpha(
        _alpha: TrainState, log_probs: jax.Array
    ) -> Tuple[TrainState, jax.Array, jax.Array, dict]:
        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(
            _alpha.params, log_probs
        )
        _alpha = _alpha.apply_gradients(grads=alpha_grads)
        alpha_vals = _alpha.apply_fn(_alpha.params, batch.task_ids)
        return (
            _alpha,
            alpha_vals,
            {"losses/alpha_loss": alpha_loss_value, "alpha": jnp.exp(_alpha.params).sum()},  # type: ignore
        )

    key, actor_loss_key = jax.random.split(key)

    def actor_loss(params: flax.core.FrozenDict):
        action_samples, log_probs, _ = sample_and_log_prob(
            actor, params, batch.observations, batch.task_ids, actor_loss_key
        )
        _alpha, _alpha_val, alpha_logs = update_alpha(alpha, log_probs)
        _alpha_val = jax.lax.stop_gradient(_alpha_val)
        _critic, critic_logs = update_critic(critic, _alpha_val)
        logs = {**alpha_logs, **critic_logs}

        q_values = _critic.apply_fn(
            _critic.params, batch.observations, action_samples, batch.task_ids
        )
        min_qf_values = jnp.min(q_values, axis=0)
        return (_alpha_val * log_probs.sum(-1).reshape(-1, 1) - min_qf_values).mean(), (
            _alpha,
            _critic,
            logs,
        )

    (actor_loss_value, (alpha, critic, logs)), actor_grads = jax.value_and_grad(
        actor_loss, has_aux=True
    )(actor.params)
    actor = actor.apply_gradients(grads=actor_grads)

    return (actor, critic, alpha), {**logs, "losses/actor_loss": actor_loss_value}, key

@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    dones: jnp.array
    rewards: jnp.array
    next_obs: jnp.array

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

# Training loop
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"

    if args.reward_filter == 'gaussian':
        run_name += f"_gaussian_filtering__{args.filter_mode}"
    elif args.reward_filter == 'exponential':
        assert args.alpha > 0, "make sure EMA constant > 0"
        assert args.alpha < 1, "make sure EMA constant < 1"
        run_name += f"__exponential_filtering__alpha_{args.alpha}"
    elif args.reward_filter == 'uniform':
        assert args.delta > 0, 'delta must be greater than 0'
        run_name += f'__uniform_filtering_{args.kernel_type}_{args.delta}_{args.filter_mode}'
    elif args.reward_filter:
        raise NotImplementedError(f"Filtering not implemented for {args.reward_filter}")

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    run_name += f"_{args.seed}_{time.time()}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.save_model:  # Orbax checkpoints
        ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=5, create=True, best_fn=lambda x: x["charts/mean_success_rate"]
        )
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt_manager = orbax.checkpoint.CheckpointManager(
            f"runs/{run_name}/checkpoints", checkpointer, options=ckpt_options
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

    use_one_hot_wrapper = (
        True if "MT10" in args.env_id or "MT50" in args.env_id else False
    )

    envs = make_envs(
        benchmark, args.seed, args.max_episode_steps, use_one_hot=use_one_hot_wrapper, reward_func_version=args.reward_version
    )

    NUM_TASKS = len(benchmark.train_classes)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

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
    env_names = list(benchmark.train_classes.keys())

    start_time = time.time()
    derivatives = np.asarray([0. for _ in range(NUM_TASKS)])
    last_rewards = None 
    last_actions = None

    episodic_storage = Storage(
        obs=jnp.zeros((args.max_episode_steps, envs.num_envs) + envs.single_observation_space.shape, dtype=jnp.float32),
        actions=jnp.zeros((args.max_episode_steps, envs.num_envs) + envs.single_action_space.shape, dtype=jnp.float32),
        dones=jnp.zeros((args.max_episode_steps, envs.num_envs), dtype=jnp.float32),
        rewards=jnp.zeros((args.max_episode_steps, envs.num_envs), dtype=jnp.float32),
        next_obs=jnp.zeros((args.max_episode_steps, envs.num_envs) + envs.single_observation_space.shape, dtype=jnp.float32),
    )

    gamma = args.gamma
    epsilon = 1e-8
    returns = jnp.zeros(envs.num_envs)
    return_rms = RunningMeanStd(shape=(envs.num_envs, ))

    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps // NUM_TASKS):
        for k in agent.actor.params['params']:
            if 'Dense' in k:
                writer.add_scalar(f'actor_{k}', np.float64(jnp.linalg.norm(agent.actor.params['params'][k]['kernel'])), global_step)
        for k in agent.critic.params['params']['VmapCritic_0']:
            if 'Dense' in k:
                writer.add_scalar(f'critic_{k}', np.float64(jnp.linalg.norm(agent.critic.params['params']['VmapCritic_0'][k]['kernel'])), global_step)

        if global_step % args.max_episode_steps == 0:
             if global_step > args.learning_starts:
                 for i in range(NUM_TASKS):
                     writer.add_scalar(
                        f"charts/{env_names[i]}_avg_derivative_per_episode",
                        derivatives[i]/args.max_episode_steps,
                        global_step,
                     )
             derivatives = np.asarray([0. for _ in range(NUM_TASKS)])

        total_steps = global_step * NUM_TASKS

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(NUM_TASKS)]
            )
        else:
            actions, key = agent.get_action_train(obs, key)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if last_rewards is None and global_step % args.max_episode_steps != 0:
            last_rewards = rewards.copy()
            last_actions = np.asarray(actions).copy()[:, :3]
        elif global_step % args.max_episode_steps == 0:
            last_rewards = None
            last_actions = None
        else: 
            dist = np.linalg.norm(last_actions - np.asarray(actions)[:, :3])
            derivatives += (rewards - last_rewards) / dist
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                global_episodic_return.append(info["episode"]["r"])
                global_episodic_length.append(info["episode"]["l"])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncations):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Store data in the episodic buffer

        next_done = np.logical_or(truncations, terminations)
        episodic_storage = episodic_storage.replace(
            obs=episodic_storage.obs.at[global_step % args.max_episode_steps].set(obs),
            dones=episodic_storage.dones.at[global_step % args.max_episode_steps].set(next_done),
            actions=episodic_storage.actions.at[global_step % args.max_episode_steps].set(actions),
            rewards=episodic_storage.rewards.at[global_step % args.max_episode_steps].set(rewards),
            next_obs=episodic_storage.next_obs.at[global_step % args.max_episode_steps].set(real_next_obs)
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 500 == 0 and global_episodic_return:
            # Sample a batch from replay buffer
            before_rewards = np.mean(np.array(episodic_storage.rewards), axis=0)
            for i in range(envs.num_envs):
                writer.add_scalar(f'Mean before smoothing', before_rewards[i])
            if args.reward_filter == 'gaussian':
                rewards = gaussian_filter1d(np.array(episodic_storage.rewards), args.sigma, mode=args.filter_mode,
                                            axis=0)
            elif args.reward_filter == 'exponential':
                raise NotImplementedError("Reggie look into this one")
                rewards_raw = np.array(episodic_storage.rewards)
                rewards = np.zeros_like(rewards_raw)
                rewards[-1] = rewards_raw[0]
                beta = 1 - args.alpha
                for i, rew_raw in enumerate(rewards_raw):
                    rewards = args.alpha * rewards[i - 1] + beta * rew_raw
            elif args.reward_filter == 'uniform':
                if args.kernel_type == 'uniform':
                    filter = (1.0 / args.delta) * np.array([1] * args.delta)
                elif args.kernel_type == 'uniform_before':
                    filter = (1.0 / args.delta) * np.array([1] * args.delta + [0] * (args.delta - 1))
                elif args.kernel_type == 'uniform_after':
                    filter = (1.0 / args.delta) * np.array([0] * (args.delta - 1) + [1] * args.delta)
                else:
                    raise NotImplementedError('Invalid kernel type for uniform smoothing')
                rewards = convolve1d(np.array(episodic_storage.rewards), filter, mode=args.filter_mode, axis=0)

            if args.normalize_rewards:
                terminated = 1 - terminations
                returns = returns * gamma * (1 - terminated) + rewards
                return_rms.update(returns)
                rewards = rewards / jnp.sqrt(return_rms.var + epsilon)
                rewards = np.asarray(rewards)

            episodic_storage = episodic_storage.replace(rewards=episodic_storage.rewards.at[:].set(rewards))
            after_rewards = np.mean(np.array(episodic_storage.rewards), axis=0)
            for i in range(envs.num_envs):
                writer.add_scalar(f'Mean after smoothing', after_rewards[i])
            for i in range(args.max_episode_steps):
                rb.add(
                    episodic_storage.obs[i, :],
                    episodic_storage.next_obs[i, :],
                    episodic_storage.actions[i, :],
                    episodic_storage.rewards[i, :],
                    episodic_storage.dones[i, :]
                )
            print(
                f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
            )
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

        from scipy.ndimage import convolve1d, gaussian_filter1d

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_episodic_return:

            for i in range(args.gradient_steps):
                if i % 100 == 0 and i > 0:
                    print(f'gradient step {i}')
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
                for _key, value in logs.items():
                    writer.add_scalar(_key, value, total_steps)
                print("SPS:", int(total_steps / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(total_steps / (time.time() - start_time)),
                    total_steps,
                )

            # Evaluation
            if total_steps % args.evaluation_frequency == 0 and global_step > 0:
                envs.set_attr('terminate_on_success', True)
                (
                    eval_success_rate,
                    eval_returns,
                    eval_success_per_task,
                ) = evaluation(
                    agent=agent,
                    eval_envs=envs,
                    num_episodes=args.evaluation_num_episodes,
                )
                envs.set_attr('terminate_on_success', False)
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

                # Checkpointing
                if args.save_model:
                    ckpt = agent.get_ckpt()
                    ckpt["rng_key"] = key
                    ckpt["global_step"] = global_step
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    ckpt_manager.save(
                        step=global_step,
                        items=ckpt,
                        save_kwargs={"save_args": save_args},
                        metrics=eval_metrics,
                    )
                    print(f"model saved to {ckpt_manager.directory}")

    envs.close()
    writer.close()
