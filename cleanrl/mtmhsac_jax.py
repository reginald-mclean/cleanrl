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
] = "0.05"

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
from scipy.ndimage import gaussian_filter1d, convolve1d
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS, MT10_V2
from metaworld import Benchmark, Task
import pickle 
import torch

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
    parser.add_argument("--reward-version", nargs="*", default="v1", help="the reward function of the environment")
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
    parser.add_argument('--gradient-steps', type=int, default=1)

    # reward smoothing
    parser.add_argument("--reward-filter", type=str, default=None)
    parser.add_argument('--filter-mode', type=str, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--kernel-type', type=str, default=None)

    # reward normalization
    parser.add_argument('--normalize-rewards', type=lambda x: bool(strtobool(x)), default=False, help='normalize after smoothing')
    parser.add_argument('--normalize-rewards-env', type=lambda x: bool(strtobool(x)), default=False, help='use the normalization wrapper around each env')


    parser.add_argument('--model-type', type=str, default=None, help='what vlm reward model to use')


    args = parser.parse_args()

    if len(args.reward_version) == 1:
        args.reward_version = args.reward_version[0]

    # fmt: on
    print(args)
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

_N_GOALS = 50

def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, kwargs_override, seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    tasks = []
    for (env_name, args) in args_kwargs.items():
        assert len(args['args']) == 0
        env_cls = classes[env_name]
        env = env_cls()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args['kwargs'].copy()
        del kwargs['task_id']
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS

        env.close()
        for rand_vec in rand_vecs:
            kwargs = args['kwargs'].copy()
            del kwargs['task_id']
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
    if seed is not None:
        np.random.set_state(st0)
    return tasks



class Modified_MT10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()

        env_names = ['button-press-v2', 'door-close-v2', 'door-unlock-v2', 'drawer-close-v2', 'drawer-open-v2', 'handle-press-side-v2', 'handle-press-v2', 'sweep-into-v2', 'window-close-v2', 'window-open-v2']
        self._train_classes = {name: ALL_V2_ENVIRONMENTS[name] for name in env_names}
        self._test_classes = {}

        train_kwargs = {
              key: dict(args=[],
              kwargs={'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
              for key in env_names
        }

        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        dict(partially_observable=False),
                                        seed=seed)
        self._test_tasks = []


def preprocess_metaworld(frames, shorten=True):
    #(10, 480, 480, 3)
    center = 240, 320
    h, w = (250, 250)
    x = int(center[1] - w/2)
    y = int(center[0] - h/2)
    frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
    a = frames
    #if shorten:
    #    frames = frames[:, :,::4,:,:]
    # frames = frames/255
    return frames # torch.from_numpy(frames).double()

instruction_mapping = {
    "window-open-v2": "Push and open a sliding window by its handle.",
    "window-close-v2": "Push and close a sliding window by its handle.",
    "door-open-v2": "Open a door with a revolving joint by the pulling door's handle.",
    "drawer-open-v2": "Open a drawer by its handle by pulling on it.",
    "drawer-close-v2": "Close a drawer by its handle by pushing on it.",
    "door-unlock-v2": "Unlock the door by rotating the lock counter-clockwise.",
    "sweep-into-v2": " Sweep a puck from the initial position into a hole.",
    "button-press-v2": "Press a button in y coordination.",
    "handle-press-v2": "Press a handle down.",
    "handle-press-side-v2": "Press a handle down sideways.",
    "reach-v2": "Reach towards a goal position.",
    "button-press-topdown-v2": "Press a button that is on top of a box.",
    "peg-insert-side-v2": "Move towards a stick on the ground, grasp it in the middle, and insert it into the goal",
    "push-v2": "Move towards the object, and push the object towards a goal",
    "pick-place-v2": "Move towards an object, grasp the object, and move it to the goal",
    "bottom burner": "Turn the oven knob that activates the bottom burner",
    "top burner": "Turn the oven knob that activates the top burner",
    "light switch": "Slide the light switch",
    "slide cabinet": "Push and open a sliding cabinet by its handle.",
    "hinge cabinet": "Grasp the handle of the left hinge cabinet and open",
    "microwave": "Grasp the handle of the microwave and open",
    "kettle": "pick up the kettle by the handle and move it to the top left burner"
}


# Training loop
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"

    if args.reward_filter == 'gaussian':
        assert args.sigma > 0, args.sigma
        assert args.filter_mode, args.filter_mode
        run_name += f"_gaussian_filtering_{args.sigma}_{args.filter_mode}"
    elif args.reward_filter == 'exponential':
        assert args.alpha > 0, "make sure EMA constant > 0"
        assert args.alpha < 1, "make sure EMA constant < 1"
        run_name += f"__exponential_filtering__alpha_{args.alpha}"
    elif args.reward_filter == 'uniform':
        assert args.delta > 0, 'delta must be greater than 0'
        args.delta = int(args.delta)
        run_name += f'__uniform_filtering_{args.kernel_type}_{args.delta}_{args.filter_mode}'
    elif args.reward_filter:
        raise NotImplementedError(f"Filtering not implemented for {args.reward_filter}")

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            #sync_tensorboard=True,
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
    elif args.env_id == "Modified_MT10":
        benchmark = Modified_MT10(seed=args.seed)
    else:
        benchmark = metaworld.MT1(args.env_id, seed=args.seed)

    use_one_hot_wrapper = (
        True if "MT10" in args.env_id or "MT50" in args.env_id else False
    )

    envs = make_envs(
        benchmark, args.seed, args.max_episode_steps, use_one_hot=use_one_hot_wrapper, reward_func_version=args.reward_version, normalize_rewards=False #args.normalize_rewards_env # normalize-rewards-env
    )
    eval_envs = make_eval_envs(
        benchmark, args.seed, args.max_episode_steps, use_one_hot=use_one_hot_wrapper, reward_func_version=args.reward_version, normalize_rewards=False
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

    obs_buffer = np.zeros((args.max_episode_steps, envs.num_envs) + envs.single_observation_space.shape, dtype=np.float32)
    actions_buffer = np.zeros((args.max_episode_steps, envs.num_envs) + envs.single_action_space.shape, dtype=np.float32)
    dones_buffer = np.zeros((args.max_episode_steps, envs.num_envs), dtype=np.float32)
    rewards_buffer = np.zeros((args.max_episode_steps, envs.num_envs), dtype=np.float32)
    next_obs_buffer = np.zeros((args.max_episode_steps, envs.num_envs) + envs.single_observation_space.shape, dtype=np.float32)

    gamma = args.gamma
    epsilon = 1e-8
    returns = jnp.zeros(envs.num_envs)
    return_rms = RunningMeanStd(shape=(envs.num_envs, ))

    mt10_descs = [instruction_mapping[name] for name in benchmark.train_classes]

    if args.model_type == 'S3D':
        from s3dg import S3D
        S3D_PATH = '/home/reggiemclean/rf_smoothness/s3d/'
        model = S3D(f'{S3D_PATH}s3d_dict.npy', 512).double()
        model.load_state_dict(torch.load(f'{S3D_PATH}s3d_howto100m.pth'))
        model.eval()
        text_output = model.text_module(mt10_descs)
        text_output = text_output['text_embedding'].to('cuda:0')
        model = model.to('cuda:0')
        frame_history = np.zeros((args.max_episode_steps, envs.num_envs, 250, 250, 3))
    elif args.model_type == 'LIV':
        import clip
        import torch
        import torchvision.transforms as T
        from PIL import Image 

        from liv import load_liv

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = load_liv()
        model.eval()
        transform = T.Compose([T.ToTensor()])
        text_tokens = []
        # pre-process image and text
        for desc in mt10_descs:
            text = clip.tokenize([desc]).to(device)
            text_tokens.append(text)
        text_tokens = torch.stack(text_tokens).to(device).squeeze(1)
        with torch.no_grad():
            text_embedding = model(input=text_tokens, modality="text")
        # compute LIV value

    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps // NUM_TASKS):
        act_params = {}
        cri_params= {}
        for k in agent.actor.params['params']:
            if 'Dense' in k:
                act_params[f'actor_{k}'] = np.float64(jnp.linalg.norm(agent.actor.params['params'][k]['kernel']))
        for k in agent.critic.params['params']['VmapCritic_0']:
            if 'Dense' in k:
                cri_params[f'critic_{k}'] = np.float64(jnp.linalg.norm(agent.critic.params['params']['VmapCritic_0'][k]['kernel']))

        if args.track:
            wandb.log(act_params, commit=False)
            wandb.log(cri_params, commit=False)
        if global_step % args.max_episode_steps == 0:
             if global_step > args.learning_starts:
                 reward_dict = {}
                 for i in range(NUM_TASKS):
                     reward_dict[f"charts/{env_names[i]}_real_reward_change_per_unit_displace"] = derivatives[i]/args.max_episode_steps
             derivatives = np.asarray([0. for _ in range(NUM_TASKS)])

        total_steps = global_step * NUM_TASKS

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(NUM_TASKS)]
            )
        else:
            actions, key = agent.get_action_train(obs, key)
        #if global_step % 1000 == 0:
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)


        if args.model_type:
            frames = envs.call('render')

        if args.model_type == 'S3D':
            frame_history[global_step % args.max_episode_steps] = preprocess_metaworld(list(frames))
            linspace = torch.linspace(0, global_step % args.max_episode_steps, 32, dtype=torch.int).numpy()
            start = time.time()
            video = torch.from_numpy(frame_history[linspace, :, :, :, :].transpose(1, 4, 0, 2, 3))
            og_rewards = rewards.copy()
            with torch.no_grad():
                video_output = model(video.to('cuda:0'))
                rewards = (video_output['video_embedding'] * text_output).sum(dim=1).cpu().numpy()
        elif args.model_type == 'LIV':
            frames = torch.stack([transform(img) for img in frames])
            with torch.no_grad():
                img_embed = model(input=frames, modality='vision')
            rewards = model.module.sim(img_embed, text_embedding).cpu().numpy()


        if args.normalize_rewards_env:
            terminated = 1 - terminations
            returns = returns * gamma * (1 - terminated) + rewards
            return_rms.update(returns)
            rewards = rewards / jnp.sqrt(return_rms.var + epsilon)
            rewards = np.asarray(rewards)



        if global_step % args.max_episode_steps == 0:
            last_rewards = rewards.copy()
            last_actions = np.asarray(actions).copy()[:, :3]
        else:
            dist = np.linalg.norm(last_actions - np.asarray(actions)[:, :3], axis=1)
            derivatives += (rewards - last_rewards) / dist
            last_rewards = rewards.copy()
            last_actions = np.asarray(actions).copy()[:, :3]

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

        if not args.reward_filter:
             rb.add(obs, real_next_obs, actions, rewards, terminations)
        else:
            next_done = np.logical_or(truncations, terminations)

            obs_buffer[global_step % args.max_episode_steps, :] = obs
            dones_buffer[global_step % args.max_episode_steps, :] = next_done
            actions_buffer[global_step % args.max_episode_steps, :] = actions
            rewards_buffer[global_step % args.max_episode_steps, :] = rewards
            next_obs_buffer[global_step % args.max_episode_steps, :] = real_next_obs

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 500 == 0 and global_episodic_return:
           if args.reward_filter:
               # Sample a batch from replay buffer
               before_rewards = np.mean(rewards_buffer, axis=0)
               if args.track:
                   for i in range(envs.num_envs):
                       wandb.log({f'Mean before smoothing env {i}':before_rewards[i]}, commit=False)
               if args.reward_filter == 'gaussian':
                   rewards = gaussian_filter1d(rewards_buffer, args.sigma, mode=args.filter_mode,
                                           axis=0)
               elif args.reward_filter == 'exponential':
                   rewards = np.zeros_like(rewards_buffer)
                   rewards[-1, :] = rewards_buffer[0, :]
                   beta = 1 - args.alpha
                   for i, rew_raw in enumerate(rewards_buffer):
                       rewards[i, :] = args.alpha * rewards[i - 1, :] + beta * rew_raw
               elif args.reward_filter == 'uniform':
                   if args.kernel_type == 'uniform':
                       filter = (1.0 / args.delta) * np.array([1] * args.delta)
                   elif args.kernel_type == 'uniform_before':
                       filter = (1.0/args.delta) * np.array([1] * args.delta + [0] * (args.delta-1))
                   elif args.kernel_type == 'uniform_after':
                       filter = (1.0 / args.delta) * np.array([0] * (args.delta - 1) + [1] * args.delta)
                   else:
                       raise NotImplementedError('Invalid kernel type for uniform smoothing')
                   rewards = convolve1d(rewards_buffer, filter, mode=args.filter_mode, axis=0)

               smoothing_change = np.array([0.0 for _ in range(envs.num_envs)])
               for i in range(args.max_episode_steps):
                   if i == 0:
                       l_rew = np.asarray(rewards_buffer[i])
                       l_act = np.asarray(actions_buffer[i, :, :-1])
                   else:
                       act_dist = np.linalg.norm(l_act - actions_buffer[i, :, :-1], axis=1)
                       smoothing_change += (rewards_buffer[i] - l_rew) / act_dist
                       l_rew = rewards_buffer[i]
                       l_act = actions_buffer[i, :, :-1]

               if args.track:
                   for i in range(envs.num_envs):
                       wandb.log({f"charts/{env_names[i]}_smoothed_reward_change_per_unit_displace":smoothing_change[i] / args.max_episode_steps}, commit=False)

               if args.normalize_rewards:
                   terminated = 1 - terminations
                   returns = returns * gamma * (1 - terminated) + rewards
                   return_rms.update(returns)
                   rewards = rewards / jnp.sqrt(return_rms.var + epsilon)
                   rewards = np.asarray(rewards)

               rewards_buffer = rewards
               after_rewards = np.mean(rewards_buffer, axis=0)
               if args.track:
                   for i in range(envs.num_envs):
                       wandb.log({f'Mean after smoothing env {i}': after_rewards[i]}, commit=False)

               for i in range(args.max_episode_steps):
                   rb.add(
                       obs_buffer[i, :],
                       next_obs_buffer[i, :],
                       actions_buffer[i, :],
                       rewards_buffer[i, :],
                       dones_buffer[i, :]
                   )
               obs_buffer = np.zeros((args.max_episode_steps, envs.num_envs) + envs.single_observation_space.shape, dtype=np.float32)
               actions_buffer = np.zeros((args.max_episode_steps, envs.num_envs) + envs.single_action_space.shape, dtype=np.float32)
               dones_buffer = np.zeros((args.max_episode_steps, envs.num_envs), dtype=np.float32)
               rewards_buffer = np.zeros((args.max_episode_steps, envs.num_envs), dtype=np.float32)
               next_obs_buffer = np.zeros((args.max_episode_steps, envs.num_envs) + envs.single_observation_space.shape, dtype=np.float32)


           print(
               f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
           )
           if args.track:
               wandb.log({"charts/mean_episodic_return": np.mean(list(global_episodic_return))}, commit=False)
               wandb.log({"charts/mean_episodic_length": np.mean(list(global_episodic_length))}, commit=global_step < args.learning_starts)


        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_episodic_return:

            for i in range(args.gradient_steps):
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
                print("SPS:", int(total_steps / (time.time() - start_time)))

            # Evaluation
            if total_steps % args.evaluation_frequency == 0 and global_step > 0:
                (
                    eval_success_rate,
                    eval_returns,
                    eval_success_per_task,
                ) = evaluation(
                    agent=agent,
                    eval_envs=eval_envs,
                    num_episodes=args.evaluation_num_episodes,
                )
                eval_metrics = {
                    "charts/mean_success_rate": float(eval_success_rate),
                    "charts/mean_evaluation_return": float(eval_returns),
                } | {
                    f"charts/{env_name}_success_rate": float(eval_success_per_task[i])
                    for i, (env_name, _) in enumerate(benchmark.train_classes.items())
                }
                if args.track:
                    print(np.array(returns))
                    wandb.log({f'returns env {idx}': np.array(returns)[idx] for idx in range(len(np.array(returns)))}, commit=False)
                    wandb.log({f'rms mean env {idx}': return_rms.mean[idx] for idx in range(len(return_rms.mean))}, commit=False)
                    wandb.log({f'rms mean env {idx}': return_rms.var[idx] for idx in range(len(return_rms.var))}, commit=False)
                    wandb.log({'rms count': return_rms.count}, commit=False)
                    wandb.log(logs, commit=False)
                    wandb.log(eval_metrics)
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

    print(return_rms.var, return_rms.mean, return_rms.count)

    envs.close()
    writer.close()
