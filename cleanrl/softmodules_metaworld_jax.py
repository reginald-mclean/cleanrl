# ruff: noqa: E402
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from functools import partial
from typing import NamedTuple, Optional, Tuple, Type, Union

os.environ[
    "XLA_PYTHON_CLIENT_PREALLOCATE"
] = "false"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import flax
import flax.linen as nn
import gymnasium as gym  # type: ignore
import jax
import jax.numpy as jnp
import metaworld
import numpy as np
import optax  # type: ignore
import orbax.checkpoint
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers_metaworld import MultiTaskReplayBuffer
from cleanrl_utils.evals.metaworld_jax_eval import evaluation_procedure
from cleanrl_utils.wrappers.metaworld_wrappers import OneHotWrapper, RandomTaskSelectWrapper


# Experiment management utils
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument('--save-model-frequency', type=int, default=50_000,
        help="the frequency of saving the model")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MT10", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=15_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--epoch-size", type=int, default=200,
        help="""if > 1, we'll collect epoch_size * num_envs samples, and then run epoch_size gradient steps.\
            Otherwise, a gradient step will happen once every timestep after learning starts.""")
    parser.add_argument("--max-episode-steps", type=int, default=None,
        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory for each task")
    parser.add_argument("--learning-starts", type=int, default=4e3, help="timestep to start learning")
    parser.add_argument("--evaluation-frequency", type=int, default=1_000_000,
        help="how many updates to do before evaluating the agent")
    parser.add_argument("--evaluation-num-workers", type=int, default=10,
        help="the number of evaluation workers")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes per evaluation")
    # SAC
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0,
        help="the value to clip the gradient norm to. Disabled if 0. Not applied to alpha gradients.") 
    # Soft Modules
    parser.add_argument("--num-modules", "-n", type=int, default=2,
        help="the number of modules per layer in the network (n)")
    parser.add_argument("--num-layers", "-L", type=int, default=2,
        help="the number of layers in the network (L)")
    parser.add_argument("--module-dim", "-d", type=int, default=256,
        help="the dimension of each module (d)")
    parser.add_argument("--embedding-dim", "-D", type=int, default=400,
        help="the dimension of the task embedding (D)")
    args = parser.parse_args()
    # fmt: on
    return args


def make_envs(
    benchmark: metaworld.Benchmark, seed: int, max_episode_steps: Optional[int] = None, use_one_hot: bool = True
) -> gym.Env:
    def init_each_env(env_cls: Type[gym.Env], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            env = OneHotWrapper(env, env_id, len(benchmark.train_classes))
        env = RandomTaskSelectWrapper(env, [task for task in benchmark.train_tasks if task.env_name == name])
        env.action_space.seed(seed)
        return env

    return gym.vector.AsyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id)
            for env_id, (name, env_cls) in enumerate(benchmark.train_classes.items())
        ]
    )


# Utils
JaxOrNPArray: Type = Union[jnp.ndarray, np.ndarray]


def split_obs_task_id(obs: JaxOrNPArray, num_tasks: int) -> Tuple[JaxOrNPArray, JaxOrNPArray]:
    return obs[..., :-num_tasks], obs[..., -num_tasks:]


def scale_action(action_space: gym.spaces.Box, action: np.ndarray) -> np.ndarray:
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space: gym.spaces.Box, scaled_action: np.ndarray) -> np.ndarray:
    low, high = action_space.low, action_space.high
    scaled_action = np.clip(scaled_action, -1, 1)
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


# Networks

# NOTE the paper is missing quite a lot of details that are in the official code
#
# 1) there is an extra embedding layer for the task embedding after z and f have been combined
# 2) the obs embedding is activated before it's passed into the layers
# 3) p_l+1 is not dependent on just p_l but on all p_<l with skip connections
# 4) ReLU is applied after the weighted sum in forward computation, not before as in Eq. 8 in the paper
# 5) there is an extra p_L+1 that is applied as a dot product over the final module outputs
#
# These are marked with "NOTE <number>"


class MLPTorso(nn.Module):
    """A Flax Module to represent an MLP feature extractor.
    Will be used to implement f(s_t) and h(z_Tau)."""

    num_hidden_layers: int  # 1 for f(s_t), 0 for h(z_Tau)
    output_dim: int  # D
    hidden_dim: int = 400
    activate_last: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim, name=f"layer_{i}")(x)  # type: ignore
            x = nn.relu(x)
        x = nn.Dense(self.output_dim, name=f"layer_{self.num_hidden_layers}")(x)  # type: ignore
        return x


class BasePolicyNetworkLayer(nn.Module):
    """A Flax Module to represent a single layer of modules of the Base Policy Network"""

    num_modules: int  # n
    module_dim: int  # d

    def setup(self):
        self.modules = [nn.Dense(self.module_dim) for _ in range(self.num_modules)]

    def __call__(self, x: jax.Array) -> jax.Array:
        # Assuming x to be of shape [B, n, D]
        # Output will be [B, n, d]
        # NOTE 4, relu *should* be here according to the paper, but it's after the weighted sum
        return jnp.stack([module(x[..., j, :]) for j, module in enumerate(self.modules)], axis=-2)


class RoutingNetworkLayer(nn.Module):
    """A Flax Module to represent a single layer of the Routing Network"""

    embedding_dim: int  # D
    num_modules: int
    last: bool = False  # NOTE 5

    def setup(self):
        self.prob_embedding_fc = nn.Dense(self.embedding_dim)  # W_u^l
        # NOTE 5
        prob_output_dim = self.num_modules if self.last else self.num_modules * self.num_modules
        self.prob_output_fc = nn.Dense(prob_output_dim)  # W_d^l

    def __call__(self, task_embedding: jax.Array, prev_probs: Optional[jax.Array] = None) -> jax.Array:
        if prev_probs is not None:  # Eq 5-only bit
            task_embedding *= self.prob_embedding_fc(prev_probs)
        x = self.prob_output_fc(nn.relu(task_embedding))
        if not self.last:  # NOTE 5
            x = x.reshape((-1, self.num_modules, self.num_modules))
        x = nn.softmax(x, axis=-1)  # Eq. 7
        return x


class SoftModularizationNetwork(nn.Module):
    """A Flax Module to represent the Base Policy Network and the Routing Network simultaneously,
    since their layers are so intertwined.

    Corresponds to `ModularGatedCascadeCondNet` in the official implementation."""

    embedding_dim: int  # D
    module_dim: int  # d
    num_layers: int
    num_modules: int
    output_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    routing_skip_connections: bool = True  # NOTE 3

    def setup(self) -> None:
        # Base policy network layers
        self.f = MLPTorso(num_hidden_layers=1, output_dim=self.embedding_dim)
        self.layers = [BasePolicyNetworkLayer(self.num_modules, self.module_dim) for _ in range(self.num_layers)]
        self.output_head = nn.Dense(self.output_dim)

        # Routing network layers
        self.z = MLPTorso(num_hidden_layers=0, output_dim=self.embedding_dim)
        self.task_embedding_fc = MLPTorso(num_hidden_layers=1, hidden_dim=256, output_dim=self.embedding_dim)  # NOTE 1
        self.prob_fcs = [
            RoutingNetworkLayer(self.embedding_dim, self.num_modules, last=i == self.num_layers - 1)
            for i in range(self.num_layers)  # NOTE 5
        ]

    def __call__(self, s_t: jax.Array, z_Tau: jax.Array) -> jax.Array:
        # Feature extraction
        obs_embedding = self.f(s_t)
        task_embedding = self.z(z_Tau) * obs_embedding
        task_embedding = self.task_embedding_fc(nn.relu(task_embedding))  # NOTE 1

        # Initial layer inputs
        prev_probs = None
        obs_embedding = nn.relu(obs_embedding)  # NOTE 2
        module_ins = jnp.stack([obs_embedding for _ in range(self.num_modules)], axis=-2)

        if self.routing_skip_connections:  # NOTE 3
            weights = []

        for i in range(self.num_layers - 1):  # Equation 8, holds for all layers except L
            probs = self.prob_fcs[i](task_embedding, prev_probs)
            module_outs = nn.relu(jnp.squeeze(probs @ self.layers[i](module_ins)))  # NOTE 4

            # Post processing
            if self.routing_skip_connections:  # NOTE 3
                weights.append(probs.reshape(-1, self.num_modules * self.num_modules))
                prev_probs = jnp.concatenate(weights, axis=-1)
            else:
                prev_probs = probs
            module_ins = module_outs

        # Last layer L, Equation 9
        module_outs = self.layers[-1](module_ins)
        probs = jnp.expand_dims(self.prob_fcs[-1](task_embedding, prev_probs), axis=-1)  # NOTE 5
        output_embedding = nn.relu(jnp.sum(module_outs * probs, axis=-2))
        return self.output_head(output_embedding)


# RL Algorithm: MT-SAC
class Batch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    dones: jax.Array
    task_ids: jax.Array


class Actor(nn.Module):  # Policy network
    num_actions: int

    embedding_dim: int
    module_dim: int
    num_layers: int
    num_modules: int

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    def setup(self):
        self.net = SoftModularizationNetwork(
            embedding_dim=self.embedding_dim,
            module_dim=self.module_dim,
            num_layers=self.num_layers,
            num_modules=self.num_modules,
            output_dim=2 * self.num_actions,
        )

    def __call__(self, s_t: jax.Array, z_Tau: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = self.net(s_t, z_Tau)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, a_min=self.LOG_STD_MIN, a_max=self.LOG_STD_MAX)
        return mean, log_std


@jax.jit
def sample_action(
    actor: TrainState,
    obs: JaxOrNPArray,
    task_ids: JaxOrNPArray,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    mean, log_std = actor.apply_fn(actor.params, obs, task_ids)
    action = jnp.tanh(
        mean + jnp.exp(log_std) * jax.random.normal(action_key, shape=mean.shape)
    )  # Reparameterization trick
    return action, key


@jax.jit
def sample_and_log_prob(
    actor: TrainState,
    actor_params: flax.core.FrozenDict,  # For actor loss
    obs: JaxOrNPArray,
    task_ids: JaxOrNPArray,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    mean, log_std = actor.apply_fn(actor_params, obs, task_ids)
    std = jnp.exp(log_std)
    gaussian_action = mean + std * jax.random.normal(action_key, shape=mean.shape)
    log_prob = -0.5 * ((gaussian_action - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - log_std
    log_prob = log_prob.sum(axis=1)
    action = jnp.tanh(gaussian_action)
    log_prob -= jnp.sum(jnp.log((1 - action**2) + 1e-6), axis=1)
    return action, log_prob, key


class Critic(nn.Module):  # Q Network
    embedding_dim: int
    module_dim: int
    num_layers: int
    num_modules: int

    def setup(self):
        self.net = SoftModularizationNetwork(
            embedding_dim=self.embedding_dim,
            module_dim=self.module_dim,
            num_layers=self.num_layers,
            num_modules=self.num_modules,
            output_dim=1,
        )

    def __call__(self, s_t: jax.Array, a_t: jax.Array, z_Tau: jax.Array) -> jax.Array:
        return self.net(jnp.concatenate([s_t, a_t], axis=-1), z_Tau)


class VectorCritic(nn.Module):
    embedding_dim: int
    module_dim: int
    num_layers: int
    num_modules: int

    n_critics: int = 2

    @nn.compact
    def __call__(self, s_t: jax.Array, a_t: jax.Array, z_Tau: jax.Array) -> jax.Array:
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            embedding_dim=self.embedding_dim,
            module_dim=self.module_dim,
            num_layers=self.num_layers,
            num_modules=self.num_modules,
        )(s_t, a_t, z_Tau)
        return q_values


class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict


@jax.jit
def get_alpha(log_alpha: jax.Array, task_ids: jax.Array) -> jax.Array:
    return jnp.exp(task_ids @ log_alpha.reshape(-1, 1))


class Agent:  # MT SAC Agent
    def __init__(
        self,
        init_obs: jax.Array,
        num_tasks: int,
        embedding_dim: int,
        module_dim: int,
        num_layers: int,
        num_modules: int,
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

        network_args = {
            "embedding_dim": embedding_dim,
            "module_dim": module_dim,
            "num_layers": num_layers,
            "num_modules": num_modules,
        }
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

        actor_network = Actor(num_actions=np.prod(self._action_space.shape), **network_args)
        key, actor_init_key = jax.random.split(init_key)
        self.actor = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network.init(actor_init_key, just_obs, task_id),
            tx=_make_optimizer(policy_lr, clip_grad_norm),
        )

        _, qf_init_key = jax.random.split(key, 2)
        vector_critic_net = VectorCritic(**network_args)
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

    def get_action(self, obs: np.ndarray, key: jax.random.PRNGKeyArray) -> Tuple[np.ndarray, jax.random.PRNGKeyArray]:
        s_t, z_Tau = split_obs_task_id(obs, self._num_tasks)
        actions, key = jax.device_get(sample_action(self.actor, s_t, z_Tau, key))
        actions = unscale_action(self._action_space, actions)
        return actions, key

    @staticmethod
    @jax.jit
    def soft_update(tau: float, critic_state: CriticTrainState) -> CriticTrainState:
        qf_state = critic_state.replace(
            target_params=optax.incremental_update(critic_state.params, critic_state.target_params, tau)
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


@jax.jit
def extract_task_weights(log_alpha: jax.Array, task_ids: jax.Array) -> jax.Array:  # NOTE specific to Soft Modules
    task_weights = jax.nn.softmax(-log_alpha)  # NOTE Soft Modules official code uses log_alpha here
    task_weights = task_ids @ task_weights.reshape(-1, 1)
    return task_weights


@partial(jax.jit, static_argnames=("gamma", "target_entropy"))
def update(
    actor: TrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    batch: Batch,
    target_entropy: float,
    gamma: float,
    key: jax.random.PRNGKeyArray,
) -> Tuple[Tuple[TrainState, CriticTrainState, TrainState], dict, jax.random.PRNGKeyArray]:
    # --- Critic loss ---
    # Sample a'
    next_actions, next_action_log_probs, key = sample_and_log_prob(
        actor, actor.params, batch.next_observations, batch.task_ids, key
    )
    # Compute target Q values
    q_values = critic.apply_fn(critic.target_params, batch.next_observations, next_actions, batch.task_ids)

    def critic_loss(params: flax.core.FrozenDict, alpha_val: jax.Array, task_weights: jax.Array) -> jnp.float32:
        # next_action_log_probs is (B,) shaped because of the sum(axis=1), while Q values are (B, 1)
        min_qf_next_target = jnp.min(q_values, axis=0) - alpha_val * next_action_log_probs.reshape(-1, 1)
        next_q_value = jax.lax.stop_gradient(batch.rewards + (1 - batch.dones) * gamma * min_qf_next_target)

        q_pred = critic.apply_fn(params, batch.observations, batch.actions, batch.task_ids)
        # NOTE specific to Soft Modules: task weights
        return 0.5 * (task_weights * (q_pred - next_q_value) ** 2).mean(axis=1).sum(), q_pred.mean()

    def update_critic(
        _critic: CriticTrainState, alpha_val: jax.Array, task_weights: jax.Array
    ) -> Tuple[CriticTrainState, dict]:
        (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(critic_loss, has_aux=True)(
            _critic.params, alpha_val, task_weights
        )
        _critic = _critic.apply_gradients(grads=critic_grads)
        return _critic, {"losses/qf_values": qf_values, "losses/qf_loss": critic_loss_value}

    # --- Alpha loss ---
    def alpha_loss(params: jax.Array, log_probs: jax.Array) -> jnp.float32:
        log_alpha = batch.task_ids @ params.reshape(-1, 1)
        return (-log_alpha * (log_probs.reshape(-1, 1) + target_entropy)).mean()

    def update_alpha(_alpha: TrainState, log_probs: jax.Array) -> Tuple[TrainState, jax.Array, jax.Array, dict]:
        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(_alpha.params, log_probs)
        _alpha = _alpha.apply_gradients(grads=alpha_grads)
        alpha_vals = _alpha.apply_fn(_alpha.params, batch.task_ids)
        task_weights = extract_task_weights(_alpha.params, batch.task_ids)
        return (
            _alpha,
            alpha_vals,
            task_weights,
            {"losses/alpha_loss": alpha_loss_value, "alpha": jnp.exp(_alpha.params).sum()},
        )

    # --- Actor loss --- & calls for the other losses
    key, actor_loss_key = jax.random.split(key)

    def actor_loss(params: flax.core.FrozenDict) -> jnp.float32:
        action_samples, log_probs, _ = sample_and_log_prob(
            actor, params, batch.observations, batch.task_ids, actor_loss_key
        )

        # HACK Putting the other losses / grad updates inside this function for performance,
        # so we can reuse the action_samples / log_probs while also doing alpha loss first
        _alpha, _alpha_val, task_weights, alpha_logs = update_alpha(alpha, log_probs)
        _alpha_val = jax.lax.stop_gradient(_alpha_val)
        task_weights = jax.lax.stop_gradient(task_weights)
        _critic, critic_logs = update_critic(critic, _alpha_val, task_weights)
        logs = {**alpha_logs, **critic_logs}

        q_values = _critic.apply_fn(_critic.params, batch.observations, action_samples, batch.task_ids)
        min_qf_values = jnp.min(q_values, axis=0)
        return (task_weights * (_alpha_val * log_probs.reshape(-1, 1) - min_qf_values)).mean(), (_alpha, _critic, logs)

    (actor_loss_value, (alpha, critic, logs)), actor_grads = jax.value_and_grad(actor_loss, has_aux=True)(actor.params)
    actor = actor.apply_gradients(grads=actor_grads)

    return (actor, critic, alpha), {**logs, "losses/actor_loss": actor_loss_value}, key


# Training loop
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.save_model:  # Orbax checkpoints
        ckpt_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
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

    use_one_hot_wrapper = True if "MT10" in args.env_id or "MT50" in args.env_id else False
    envs = make_envs(benchmark, args.seed, args.max_episode_steps, use_one_hot=use_one_hot_wrapper)

    NUM_TASKS = len(benchmark.train_classes)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # agent setup
    envs.single_observation_space.dtype = np.float32
    rb = MultiTaskReplayBuffer(
        capacity=args.buffer_size,
        num_tasks=NUM_TASKS,
        envs=envs,
        use_torch=False,
        seed=args.seed,
    )

    global_episodic_return = deque([], maxlen=20 * envs.num_envs)
    global_episodic_length = deque([], maxlen=20 * envs.num_envs)

    obs, _ = envs.reset()

    key, agent_init_key = jax.random.split(key)
    agent = Agent(
        obs,
        NUM_TASKS,
        args.embedding_dim,
        args.module_dim,
        args.num_layers,
        args.num_modules,
        envs.single_action_space,
        args.policy_lr,
        args.q_lr,
        args.gamma,
        args.clip_grad_norm,
        key,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, key = agent.get_action(obs, key)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                global_episodic_return.append(info["episode"]["r"])
                global_episodic_length.append(info["episode"]["l"])
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncations):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Store scaled actions
        actions = scale_action(envs.single_action_space, actions)
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 500 == 0 and global_episodic_return:
            print(f"global_step={global_step}, mean_episodic_return={np.mean(global_episodic_return)}")
            writer.add_scalar(
                "charts/mean_episodic_return",
                np.mean(global_episodic_return),
                global_step,
            )
            writer.add_scalar(
                "charts/mean_episodic_length",
                np.mean(global_episodic_length),
                global_step,
            )

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.epoch_size == 0:  # torchrl-style training loop
                for epoch_step in range(args.epoch_size):
                    # Sample a batch from replay buffer
                    data = rb.sample(args.batch_size)
                    observations, task_ids = split_obs_task_id(data.observations, NUM_TASKS)
                    next_observations, _ = split_obs_task_id(data.next_observations, NUM_TASKS)
                    batch = Batch(observations, data.actions, data.rewards, next_observations, data.dones, task_ids)
                    logs = {}

                    (agent.actor, agent.critic, agent.alpha_train_state), logs, key = update(
                        agent.actor,
                        agent.critic,
                        agent.alpha_train_state,
                        batch,
                        agent.target_entropy,
                        args.gamma,
                        key,
                    )

                    # update the target networks
                    if (global_step + epoch_step) % args.target_network_frequency == 0:
                        agent.soft_update_target_networks(args.tau)

                    # Logging
                    if (global_step + epoch_step) % 100 == 0:
                        logs = jax.device_get(logs)
                        for _key, value in logs.items():
                            writer.add_scalar(_key, value, global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # Evaluation
                    if (global_step + epoch_step) % args.evaluation_frequency == 0:
                        print("Evaluating...")
                        eval_success_rate = evaluation_procedure(
                            num_envs=envs.num_envs,
                            num_workers=args.evaluation_num_workers,
                            num_episodes=args.evaluation_num_episodes,
                            writer=writer,
                            agent=agent,
                            update=global_step,
                            keys=list(benchmark.train_classes.keys()),
                            classes=benchmark.train_classes,
                            tasks=benchmark.train_tasks,
                        )
                        print(f"Evaluation success_rate: {eval_success_rate:.4f}")

                    # Checkpointing
                    if args.save_model and (global_step + epoch_step) % args.save_model_frequency == 0:
                        ckpt = agent.get_ckpt()
                        ckpt["rng_key"] = key
                        ckpt["global_step"] = global_step
                        save_args = orbax_utils.save_args_from_target(ckpt)
                        ckpt_manager.save(global_step, ckpt, save_kwargs={"save_args": save_args})
                        print(f"model saved to {ckpt_manager.directory}")

    envs.close()
    writer.close()
