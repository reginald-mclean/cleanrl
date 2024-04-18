# ruff: noqa: E402
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import random
import time
import sys
sys.path.append('/home/reggiemclean/multitask_transfer/cleanrl')

from collections import deque
from distutils.util import strtobool
from functools import partial
from typing import Deque, NamedTuple, Optional, Tuple, Union, List, Type
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from cleanrl_utils.wrappers import metaworld_wrappers
from metaworld import Benchmark, Task
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv
from gymnasium.wrappers.time_limit import TimeLimit

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.06"

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
from argparse import Namespace

jax.config.update("jax_enable_x64", True)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="MW-FK-Transfer",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")

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
    parser.add_argument("--batch-size", type=int, default=640,
                        help="the total size of the batch to sample from the replay memory. Must be divisible by number of tasks")
    parser.add_argument("--learning-starts", type=int, default=1e3, help="timestep to start learning")
    parser.add_argument("--evaluation-frequency", type=int, default=200_000,
        help="every how many timesteps to evaluate the agent. Evaluation is disabled if 0.")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes to run per evaluation")
    # SAC
    parser.add_argument("--policy-lr", type=float, default=3e-5,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-5, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--clip-grad-norm", type=float, default=1,
        help="the value to clip the gradient norm to. Disabled if 0. Not applied to alpha gradients.")
    parser.add_argument("--actor-network", type=str, default="400,400,400", help="The architecture of the actor network")
    parser.add_argument("--critic-network", type=str, default="400,400,400", help="The architecture of the critic network")

    # Obs Wrapper
    parser.add_argument("--original", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help='use the original state space')
    parser.add_argument("--only-pad", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help='if using MW, naively pad the state space (True) or align the inputs as well (False)')
    parser.add_argument("--one-hot", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help='add the one-hot task ID vector to every state')
    parser.add_argument("--map", type=lambda x: bool(strtobool(x)), default=False, nargs="?", 
        help="manually map the task ids to match across envs")

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

class TranslationLayer(nn.Module):
    output_size: int = 400
    hidden_dims: int = "400,400"

    @nn.compact
    def __call__(self, x: jax.Array):
        hidden_lst = [int(dim) for dim in self.hidden_dims.split(",")]
        for i, h_size in enumerate(hidden_lst):
            x = nn.Dense(
                h_size, 
                kernel_init=nn.initializers.he_uniform(), 
                bias_init=nn.initializers.constant(0.1)
            )(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_size,
            kernel_init=nn.initializers.he_uniform(), 
            bias_init=nn.initializers.constant(0.1)
        )(x)
        x = nn.relu(x)
        return x


class SEAL(nn.Module):
    output_size: int = 400
    hidden_dims: int = "400,400,400,400,400"

    @nn.compact
    def __call__(self, x: jax.Array):
        hidden_lst = [int(dim) for dim in self.hidden_dims.split(",")]
        for i, h_size in enumerate(hidden_lst):
            x = nn.Dense(
                h_size,
                kernel_init=nn.initializers.he_uniform(),
                bias_init=nn.initializers.constant(0.1),
            )(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_size, kernel_init=nn.initializers.he_uniform(),
                bias_init=nn.initializers.constant(0.1),)(x)
        x = nn.relu(x)
        return x


class Actor(nn.Module):
    num_actions: int
    num_tasks: int
    hidden_dims: int = "400"

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
            kernel_init=uniform_init(3e-3),
            bias_init=nn.initializers.constant(0.1),
        )(x)
        mu = nn.Dense(
            self.num_actions,
            kernel_init=uniform_init(3e-3),
            bias_init=nn.initializers.constant(0.1),
        )(x)
        log_sigma = jnp.clip(log_sigma, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return TanhNormal(mu, jnp.exp(log_sigma))


@jax.jit
def sample_action(
    actor: TrainState,
    translate: TrainState,
    seal: TrainState,
    obs: ArrayLike,
    task_ids: ArrayLike,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    translated = translate.apply_fn(translate.params, obs)
    seal_out = seal.apply_fn(seal.params, translated)
    dist = actor.apply_fn(actor.params, seal_out, task_ids)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def sample_and_log_prob(
    actor: TrainState,
    actor_params: flax.core.FrozenDict,
    translate: TrainState,
    translate_params: flax.core.FrozenDict,
    seal: TrainState,
    seal_params: flax.core.FrozenDict,
    obs: ArrayLike,
    task_ids: ArrayLike,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    translated = translate.apply_fn(translate_params, obs)
    seal_out = seal.apply_fn(seal_params, translated)
    dist = actor.apply_fn(actor_params, seal_out, task_ids)
    action, log_prob = dist.sample_and_log_prob(seed=action_key)
    return action, log_prob, key


@jax.jit
def get_deterministic_action(
    actor: TrainState,
    translate: TrainState,
    seal: TrainState,
    obs: ArrayLike,
    task_ids: ArrayLike,
) -> jax.Array:
    translated = translate.apply_fn(translate.params, obs)
    seal_out = seal.apply_fn(seal.params, translated)
    dist = actor.apply_fn(actor.params, seal_out, task_ids)
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
                kernel_init=uniform_init(3e-3),
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
            1, kernel_init=uniform_init(3e-3), bias_init=nn.initializers.constant(0.1)
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

def _make_optimizer(lr: float, max_grad_norm: float = 0.0):
    optim = optax.adam(learning_rate=lr)
    if max_grad_norm != 0:
        optim = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optim,
        )
    return optim


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

        translate_network = TranslationLayer()
        key, translate_init_key = jax.random.split(init_key)
        self.translate = TrainState.create(
            apply_fn=translate_network.apply,
            params=translate_network.init(translate_init_key, init_obs),
            tx=_make_optimizer(policy_lr, clip_grad_norm),
        )
        
        actor_network = Actor(
            num_actions=int(np.prod(self._action_space.shape)),
            num_tasks=num_tasks,
            hidden_dims=args.actor_network,
        )
        key, actor_init_key = jax.random.split(key)
        self.actor = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network.init(actor_init_key, jax.random.uniform(key, shape=(init_obs.shape[0], 400,)), task_id),
            tx=_make_optimizer(policy_lr, clip_grad_norm),
        )

        _, qf_init_key = jax.random.split(key, 2)
        vector_critic_net = VectorCritic(
            num_tasks=num_tasks, hidden_dims=args.critic_network
        )
        self.critic = CriticTrainState.create(
            apply_fn=vector_critic_net.apply,
            params=vector_critic_net.init(
                qf_init_key, init_obs, random_action, task_id
            ),
            target_params=vector_critic_net.init(
                qf_init_key, init_obs, random_action, task_id
            ),
            tx=_make_optimizer(q_lr, clip_grad_norm),
        )

        self.alpha_train_state = TrainState.create(
            apply_fn=get_alpha,
            params=jnp.zeros(init_obs.shape[0]),  # Log alpha
            tx=_make_optimizer(q_lr, max_grad_norm=0.0),
        )
        self.target_entropy = -np.prod(self._action_space.shape).item()

    def get_action_train(
        self, obs: np.ndarray, seal: TrainState, key: jax.random.PRNGKeyArray
    ) -> Tuple[np.ndarray, jax.random.PRNGKeyArray]:
        state, task_id = split_obs_task_id(obs, self._num_tasks)
        actions, key = sample_action(self.actor, self.translate, seal, obs, task_id, key)
        return jax.device_get(actions), key

    def get_action_eval(self, obs: np.ndarray, seal) -> np.ndarray:
        state, task_id = split_obs_task_id(obs, self._num_tasks)
        actions = get_deterministic_action(self.actor, self.translate, seal, obs, task_id)
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
    translate: TrainState,
    seal: TrainState,
    batch: Batch,
    target_entropy: float,
    gamma: float,
    key: jax.random.PRNGKeyArray,
) -> Tuple[
    Tuple[TrainState, CriticTrainState, TrainState, TrainState, TrainState], dict, jax.random.PRNGKeyArray
]:
    next_actions, next_action_log_probs, key = sample_and_log_prob(
        actor, actor.params, translate, translate.params, seal, seal.params, batch.next_observations, batch.task_ids, key
    )

    '''
    actor: TrainState,
    actor_params: flax.core.FrozenDict,
    translate: TrainState,
    seal: TrainState,
    seal_params: flax.core.FrozenDict,
    obs: ArrayLike,
    task_ids: ArrayLike,
    key: jax.random.PRNGKeyArray

    '''

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
            actor, params[0], translate, params[1], seal, params[2], batch.observations, batch.task_ids, actor_loss_key
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
    )((actor.params, translate.params, seal.params))

    actor = actor.apply_gradients(grads=actor_grads[0])
    translate = translate.apply_gradients(grads=actor_grads[1])
    seal = seal.apply_gradients(grads=actor_grads[2])

    return (actor, critic, alpha, translate, seal), {**logs, "losses/actor_loss": actor_loss_value}, key



def _make_envs_common(
    benchmark: Benchmark,
    seed: int,
    max_episode_steps: Optional[int] = 500,
    use_one_hot: bool = True,
    terminate_on_success: bool = False,
    args : Namespace = None,
    map: dict = None
) -> gym.vector.VectorEnv:
    def init_each_env(env_cls: Union[SawyerXYZEnv, KitchenEnv], name: str, env_id: int) -> gym.Env:
        """
        @type env_cls: Union[SawyerXYZEnv, KitchenEnv]
        """
        if not isinstance(env_cls, TimeLimit):
            env = env_cls(reward_func_version='v2')
            env = gym.wrappers.TimeLimit(env, 500)
        else:
            env = env_cls
        

        if terminate_on_success:
            env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            env = metaworld_wrappers.OneHotWrapper(
                env, env_id, len(benchmark.train_classes)
            )
        env = metaworld_wrappers.ObsModification(env, {'original' : args.original, 'only_pad' : args.only_pad, 'one-hot': args.one_hot, 'map': args.map}, env_id, len(benchmark.train_classes) if not args.map else max(map.values())+1, name, map)
        if isinstance(env.unwrapped, SawyerXYZEnv):
            tasks = [task for task in benchmark.train_tasks if task.env_name == name]
            env = metaworld_wrappers.PseudoRandomTaskSelectWrapper(env, tasks, True)
        env.action_space.seed(seed + env_id)
        return env

    return gym.vector.AsyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id)
            for env_id, (name, env_cls) in enumerate(benchmark.train_classes.items())
        ]
    )


class FK_Benchmark(Benchmark):
    def __init__(self):
        super().__init__()
        tasks = ['microwave', 'kettle', 'right_hinge_cabinet', 'left_hinge_cabinet', 'slide_cabinet', 'light_switch', 'top_left_burner', 'top_right_burner', 'bottom_left_burner', 'bottom_right_burner']
        self._train_classes = {name : gym.make('FrankaKitchen-v1', tasks_to_complete=[name], render_mode='rgb_array', obs_space='original', max_episode_steps=500) for name in tasks} # obs_space='original'


class MW_FK(Benchmark):
    def __init__(self, seed):
        self.mt10 = metaworld.MT10(seed=seed)
        self.fk = FK_Benchmark()
        self.fk.train_classes.update(dict(self.mt10.train_classes))

    @property
    def train_classes(self) -> "Dict[EnvName, Type]":
        return self.fk.train_classes

    @property
    def train_tasks(self) -> List[Task]:
        return self.mt10.train_tasks



# Training loop
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.exp_name,
            monitor_gym=True,
            save_code=True,
        )
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
            f"/home/reggiemclean/multitask_transfer/runs/{run_name}/checkpoints", checkpointer, options=ckpt_options
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # env setup

    make_envs = partial(_make_envs_common, terminate_on_success=False)
    make_eval_envs = partial(_make_envs_common, terminate_on_success=True)
    print('learning starts: ', args.learning_starts)
    print('making envs:', args.env_id)

    if args.env_id == "MT10":
        benchmark = metaworld.MT10(seed=args.seed)
    elif args.env_id == "FK":
        benchmark = FK_Benchmark()

    id_dict = None

    if args.map:
        tasks = ['microwave', 'kettle', 'right_hinge_cabinet', 'left_hinge_cabinet', 'slide_cabinet', 'light_switch', 'top_left_burner', 'top_right_burner', 'bottom_left_burner', 'bottom_right_burner', 
             'reach-v2', 'drawer-open-v2', 'drawer-close-v2', 'pick-place-v2', 'door-open-v2', 'push-v2', 'button-press-topdown-v2', 'window-close-v2', 'window-open-v2', 'peg-insert-side-v2']
        id_dict = {task: 0 for task in tasks}
        id_dict['pick-place-v2'] = 1
        id_dict['kettle'] = 1
        id_dict['door-open-v2'] = 2
        id_dict['microwave'] = 2
        id_dict['right_hinge_cabinet'] = 2
        id_dict['left_hinge_cabinet'] = 2
        id_dict['push-v2'] = 3
        id_dict['light_switch'] = 3 
        id_dict['button-press-topdown-v2'] = 4
        id_dict['window-close-v2'] = 5
        id_dict['window-open-v2'] = 6
        id_dict['peg-insert-side-v2'] = 7
        id_dict['reach-v2'] = 8
        id_dict['drawer-open-v2'] = 9
        id_dict['top_left_burner'] = 10
        id_dict['top_right_burner'] = 10
        id_dict['bottom_left_burner'] = 10
        id_dict['bottom_right_burner'] = 10

    if args.env_id == "MT10" or args.env_id == "FK":
        use_one_hot_wrapper = (
            True if "MT10" in args.env_id or "MT50" in args.env_id else False
        )

        envs = make_envs(
            benchmark, args.seed, args.max_episode_steps, use_one_hot=False, args=args
        )
        eval_envs = make_eval_envs(
            benchmark, args.seed, args.max_episode_steps, use_one_hot=False, args=args
        )
    else:
        benchmark = metaworld.MT10(seed=args.seed)
        mw_envs = make_envs(
            benchmark, args.seed, args.max_episode_steps, use_one_hot=False, args=args, map=id_dict
        )
        mw_eval_envs = make_eval_envs(
            benchmark, args.seed, args.max_episode_steps, use_one_hot=False, args=args, map=id_dict
        )

        fk_benchmark = FK_Benchmark()

        fk_envs = make_envs(
            fk_benchmark, args.seed, args.max_episode_steps, use_one_hot=False, args=args, map=id_dict
        )
        fk_eval_envs = make_eval_envs(
            fk_benchmark, args.seed, args.max_episode_steps, use_one_hot=False, args=args, map=id_dict
        )
        envs_list = [mw_envs, fk_envs]

    NUM_TASKS = mw_envs.num_envs + fk_envs.num_envs

    # agent setup
    rb1 = MultiTaskReplayBuffer(
        total_capacity=args.buffer_size,
        num_tasks=NUM_TASKS,
        envs=mw_envs,
        use_torch=False,
        seed=args.seed,
    )


    rb2 = MultiTaskReplayBuffer(
        total_capacity=args.buffer_size,
        num_tasks=NUM_TASKS,
        envs=fk_envs,
        use_torch=False,
        seed=args.seed,
    )

    global_episodic_return: Deque[float] = deque([], maxlen=20 * NUM_TASKS)
    global_episodic_length: Deque[int] = deque([], maxlen=20 * NUM_TASKS)

    mw_obs, _ = mw_envs.reset()
    fk_obs, _ = fk_envs.reset()

    key, seal_init_key = jax.random.split(key)

    seal = SEAL()

    seal_policy = TrainState.create(
        apply_fn=seal.apply,
        params=seal.init(seal_init_key, jax.random.uniform(key, shape=(mw_envs.num_envs, 400,))),
        tx=_make_optimizer(args.policy_lr, args.clip_grad_norm)
    )

    mw_agent = Agent(
        init_obs=mw_obs,
        num_tasks=mw_envs.num_envs,
        action_space=mw_envs.single_action_space,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        gamma=args.gamma,
        clip_grad_norm=args.clip_grad_norm,
        init_key=key,
    )
    print('mw_agent')
    fk_agent = Agent(
        init_obs=fk_obs,
        num_tasks=fk_envs.num_envs,
        action_space=fk_envs.single_action_space,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        gamma=args.gamma,
        clip_grad_norm=args.clip_grad_norm,
        init_key=key,
    )
    print('fk_agent')


    if args.env_id == 'FK':
        tasks = np.asarray(['microwave', 'kettle', 'right_hinge_cabinet', 'left_hinge_cabinet', 'slide_cabinet', 'light_switch', 'top_left_burner', 'top_right_burner', 'bottom_left_burner', 'bottom_right_burner'])
    elif args.env_id == 'MT10':
        tasks = np.asarray([None for _ in range(10)])
    else:
        mw_tasks = np.asarray([None for _ in range(10)])
        fk_tasks = np.asarray(['microwave', 'kettle', 'right_hinge_cabinet', 'left_hinge_cabinet', 'slide_cabinet', 'light_switch', 'top_left_burner', 'top_right_burner', 'bottom_left_burner', 'bottom_right_burner'])

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps // NUM_TASKS):
        total_steps = global_step * NUM_TASKS

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            mw_actions = np.array(
                [mw_envs.single_action_space.sample() for _ in range(mw_envs.num_envs)]
            )
            fk_actions = np.array(
                [fk_envs.single_action_space.sample() for _ in range(fk_envs.num_envs)]
            )
        else:
            mw_actions, key = mw_agent.get_action_train(mw_obs, seal_policy, key)
            fk_actions, key = fk_agent.get_action_train(fk_obs, seal_policy, key)
        # TRY NOT TO MODIFY: execute the game and log data.
        fk_next_obs, fk_rewards, fk_terminations, fk_truncations, fk_infos = fk_envs.step(fk_actions, fk_tasks)
        mw_next_obs, mw_rewards, mw_terminations, mw_truncations, mw_infos = mw_envs.step(mw_actions, mw_tasks)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in mw_infos:
            for info in mw_infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                global_episodic_return.append(info["episode"]["r"])
                global_episodic_length.append(info["episode"]["l"])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in fk_infos:
            for info in fk_infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                global_episodic_return.append(info["episode"]["r"])
                global_episodic_length.append(info["episode"]["l"])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        mw_real_next_obs = mw_next_obs.copy()
        fk_real_next_obs = fk_next_obs.copy()
 
        for idx, d in enumerate(mw_truncations):
            if d:
                mw_real_next_obs[idx] = mw_infos["final_observation"][idx]
        for idx, d in enumerate(fk_truncations):
            if d:
                fk_real_next_obs[idx] = fk_infos["final_observation"][idx]



        # Store data in the buffer
        rb1.add(mw_obs, mw_real_next_obs, mw_actions, mw_rewards, mw_terminations)
        rb2.add(fk_obs, fk_real_next_obs, fk_actions, fk_rewards, fk_terminations)


        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        mw_obs = mw_next_obs
        fk_obs = fk_next_obs


        if global_step % 500 == 0 and global_episodic_return:
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

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # Sample a batch from replay buffer
            mw_data = rb1.sample(args.batch_size)
            fk_data = rb2.sample(args.batch_size)


            mw_observations, mw_task_ids = split_obs_task_id(mw_data.observations, mw_envs.num_envs)
            mw_next_observations, _ = split_obs_task_id(mw_data.next_observations, mw_envs.num_envs)

            fk_observations, fk_task_ids = split_obs_task_id(fk_data.observations, fk_envs.num_envs)
            fk_next_observations, _ = split_obs_task_id(fk_data.next_observations, fk_envs.num_envs)

            mw_batch = Batch(
                mw_data.observations,
                mw_data.actions,
                mw_data.rewards,
                mw_data.next_observations,
                mw_data.dones,
                mw_task_ids,
            )

            fk_batch = Batch(
                fk_data.observations,
                fk_data.actions,
                fk_data.rewards,
                fk_data.next_observations,
                fk_data.dones,
                fk_task_ids,
            )

            (mw_agent.actor, mw_agent.critic, mw_agent.alpha_train_state, mw_agent.translate, seal_policy), mw_logs, key = update(
                mw_agent.actor,
                mw_agent.critic,
                mw_agent.alpha_train_state,
                mw_agent.translate,
                seal_policy,
                mw_batch,
                mw_agent.target_entropy,
                args.gamma,
                key,
            )


            # Update agent
            (fk_agent.actor, fk_agent.critic, fk_agent.alpha_train_state, fk_agent.translate, seal_policy), fk_logs, key = update(
                fk_agent.actor,
                fk_agent.critic,
                fk_agent.alpha_train_state,
                fk_agent.translate,
                seal_policy,
                fk_batch,
                fk_agent.target_entropy,
                args.gamma,
                key,
            )
            mw_logs = jax.device_get(mw_logs)
            fk_logs = jax.device_get(fk_logs)

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                mw_agent.soft_update_target_networks(args.tau)
                fk_agent.soft_update_target_networks(args.tau)
            # Logging
            if global_step % 100 == 0:
                for _key, value in mw_logs.items():
                    writer.add_scalar(_key, value, total_steps)
                for _key, value in fk_logs.items():
                    writer.add_scalar(_key, value, total_steps)

                print("SPS:", int(total_steps / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(total_steps / (time.time() - start_time)),
                    total_steps,
                )

            # Evaluation
            if total_steps % args.evaluation_frequency == 0: # and global_step > 0:
                (
                    mw_eval_success_rate,
                    mw_eval_returns,
                    mw_eval_success_per_task,
                    mw_successes
                ) = evaluation(
                    agent=mw_agent,
                    seal=seal_policy,
                    eval_envs=mw_eval_envs,
                    num_episodes=args.evaluation_num_episodes,
                    tasks=mw_tasks
                )
                print(mw_eval_success_per_task)
                print(mw_successes)
                (
                    fk_eval_success_rate,
                    fk_eval_returns,
                    fk_eval_success_per_task,
                    fk_successes
                ) = evaluation(
                    agent=fk_agent,
                    seal=seal_policy,
                    eval_envs=fk_eval_envs,
                    num_episodes=args.evaluation_num_episodes,
                    tasks=fk_tasks
                )
                print(fk_successes)
                
                mw_eval_metrics = {
                    "charts/mw_mean_success_rate": float(mw_eval_success_rate),
                    "charts/mw_mean_evaluation_return": float(mw_eval_returns),
                } | {
                    f"charts/{env_name}_success_rate": float(mw_eval_success_per_task[i])
                    for i, (env_name, _) in enumerate(benchmark.train_classes.items())
                }

                for k, v in mw_eval_metrics.items():
                    writer.add_scalar(k, v, total_steps)
                print(
                    f"total_steps={total_steps}, mean evaluation success rate: {mw_eval_success_rate:.4f}"
                    + f" return: {mw_eval_returns:.4f}"
                )
                fk_eval_metrics = {
                    "charts/fk_mean_success_rate": float(fk_eval_success_rate),
                    "charts/fk_mean_evaluation_return": float(fk_eval_returns),
                } | {
                    f"charts/{env_name}_success_rate": float(fk_eval_success_per_task[i])
                    for i, (env_name, _) in enumerate(fk_benchmark.train_classes.items())
                }

                for k, v in fk_eval_metrics.items():
                    writer.add_scalar(k, v, total_steps)
                print(
                    f"total_steps={total_steps}, mean evaluation success rate: {fk_eval_success_rate:.4f}"
                    + f" return: {fk_eval_returns:.4f}"
                )
                writer.add_scalar('charts/total_mean_success', np.concatenate((mw_successes, fk_successes)).sum() / (args.evaluation_num_episodes * NUM_TASKS), total_steps)
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
