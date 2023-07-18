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

import distrax
import flax
import flax.linen as nn
import gymnasium as gym  # type: ignore
import jax
import jax.numpy as jnp
import metaworld
import numpy as np
import optax  # type: ignore
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

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

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MT10", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=1280,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3, help="timestep to start learning")
    parser.add_argument("--evaluation-frequency", type=int, default=1_000_000,
        help="how many updates to before evaluating the agent")
    parser.add_argument("--evaluation-num-workers", type=int, default=10,
        help="the number of evaluation workers")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes per evaluation")
    # SAC
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1, help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0,
        help="the value to clip the gradient norm to. Disabled if 0 (Default). Not applied to alpha gradients.") 
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


def make_envs(benchmark: metaworld.Benchmark, seed: int, use_one_hot: bool) -> gym.Env:
    def init_each_env(env_cls: Type[gym.Env], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, env.max_path_length)
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

    def __call__(self, s_t: jax.Array, z_Tau: jax.Array) -> distrax.Distribution:
        x = self.net(s_t, z_Tau)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, a_min=self.LOG_STD_MIN, a_max=self.LOG_STD_MAX)
        std = jnp.exp(log_std)
        dist = distrax.Independent(  # TanhNormal
            distrax.Transformed(distrax.Normal(loc=mean, scale=std), distrax.Tanh())
        )
        return dist


@partial(jax.jit, static_argnames=("num_tasks"))
def get_action(actor: TrainState, obs: jax.Array, num_tasks: int, key: jax.random.PRNGKeyArray) -> jax.Array:
    s_t, z_Tau = split_obs_task_id(obs, num_tasks)
    action = actor.apply_fn(actor.params, s_t, z_Tau).sample(seed=key)
    return action


class Agent:  # For evaluation only because we want a pickle-able object for multiprocessing
    def __init__(self, actor: TrainState, num_tasks: int):  # Not used anywhere else
        self.actor_params = actor.params
        self.apply_fn = actor.apply_fn
        self.num_tasks = num_tasks

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, obs: jax.Array, key: jax.random.PRNGKeyArray) -> jax.Array:
        s_t, z_Tau = split_obs_task_id(obs, self.num_tasks)
        return self.apply_fn(self.actor_params, s_t, z_Tau).sample(seed=key)


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


class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict


def get_alpha(log_alpha: jax.Array, task_ids: jax.Array) -> jax.Array:
    assert log_alpha.ndim == 1, "log_alpha must be a vector"
    assert log_alpha.shape[0] == task_ids.shape[-1], "log_alpha and task onehots needs to have the same number of tasks"

    return jnp.exp(task_ids @ jnp.expand_dims(log_alpha, 0).transpose())


@partial(jax.jit, static_argnames=("target_entropy"))
def update_alpha(
    actor_state: TrainState,
    alpha_state: TrainState,
    batch: Batch,
    target_entropy: float,
    key: jax.random.PRNGKeyArray,
) -> Tuple[TrainState, dict]:
    _, action_log_probs = actor_state.apply_fn(
        actor_state.params, batch.observations, batch.task_ids
    ).sample_and_log_prob(seed=key)

    def alpha_loss(params: jax.Array) -> jnp.float32:
        log_alpha = batch.task_ids @ jnp.expand_dims(params, 0).transpose()
        return (-log_alpha * (jax.lax.stop_gradient(action_log_probs) + target_entropy)).mean()

    alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(alpha_state.params)
    alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

    return (
        alpha_state,
        {
            "losses/alpha_loss": alpha_loss_value,
            "alpha": jnp.exp(alpha_state.params).sum(),
        },
    )


@jax.jit
def extract_task_weights(log_alpha: jax.Array, task_ids: jax.Array) -> jax.Array:  # NOTE specific to Soft Modules
    task_weights = jax.nn.softmax(-log_alpha)  # NOTE Soft Modules official code uses log_alpha here
    task_weights = task_ids @ jnp.expand_dims(task_weights, 0).transpose()
    return task_weights


@partial(jax.jit, static_argnames=("gamma"))
def update_critic(
    actor_state: TrainState,
    critic_states: Tuple[CriticTrainState, CriticTrainState],
    batch: Batch,
    alpha: jax.Array,
    task_weights: jax.Array,  # NOTE specific to Soft Modules
    gamma: float,
    key: jax.random.PRNGKeyArray,
):
    qf1, qf2 = critic_states

    # Sample a'
    key, next_state_actions_key = jax.random.split(key)
    next_policy_dist = actor_state.apply_fn(actor_state.params, batch.next_observations, batch.task_ids)
    next_actions, next_action_log_probs = next_policy_dist.sample_and_log_prob(seed=next_state_actions_key)

    # Compute target Q values
    qf1_next_target = qf1.apply_fn(qf1.target_params, batch.next_observations, next_actions, batch.task_ids)
    qf2_next_target = qf2.apply_fn(qf2.target_params, batch.next_observations, next_actions, batch.task_ids)
    min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target) - alpha * next_action_log_probs
    next_q_value = batch.rewards + (1 - batch.dones) * gamma * min_qf_next_target

    def critic_mse_loss(critic: CriticTrainState, params: flax.core.FrozenDict) -> jnp.float32:
        q_pred = critic.apply_fn(params, batch.observations, batch.actions, batch.task_ids)
        # NOTE specific to Soft Modules: task weights
        return (task_weights * (q_pred - jax.lax.stop_gradient(next_q_value)) ** 2).mean(), q_pred

    (qf1_loss, qf1_a_values), qf1_grads = jax.value_and_grad(partial(critic_mse_loss, qf1), has_aux=True)(qf1.params)
    qf1 = qf1.apply_gradients(grads=qf1_grads)

    (qf2_loss, qf2_a_values), qf2_grads = jax.value_and_grad(partial(critic_mse_loss, qf2), has_aux=True)(qf2.params)
    qf2 = qf2.apply_gradients(grads=qf2_grads)

    critic_loss = (qf1_loss + qf2_loss) / 2.0  # for logging

    return (qf1, qf2), {
        "losses/qf1_values": qf1_a_values.mean(),
        "losses/qf2_values": qf2_a_values.mean(),
        "losses/qf_loss": critic_loss,
    }


@jax.jit
def update_actor(
    actor_state: TrainState,
    critic_states: Tuple[CriticTrainState, CriticTrainState],
    batch: Batch,
    alpha: jax.Array,
    task_weights: jax.Array,
    key: jax.random.PRNGKeyArray,
):
    key, actor_loss_key = jax.random.split(key)
    qf1, qf2 = critic_states

    def actor_loss(params: flax.core.FrozenDict) -> jnp.float32:
        policy_dist = actor_state.apply_fn(params, batch.observations, batch.task_ids)
        action_samples, log_probs = policy_dist.sample_and_log_prob(seed=actor_loss_key)
        qf1_values = qf1.apply_fn(qf1.params, batch.observations, action_samples, batch.task_ids)
        qf2_values = qf2.apply_fn(qf2.params, batch.observations, action_samples, batch.task_ids)
        min_qf_values = jnp.minimum(qf1_values, qf2_values)
        # NOTE specific to Soft Modules: task weights
        return (task_weights * (alpha * log_probs - min_qf_values)).mean()

    actor_loss_value, actor_grads = jax.value_and_grad(actor_loss)(actor_state.params)  # value for logging
    actor_state = actor_state.apply_gradients(grads=actor_grads)

    return actor_state, {
        "losses/actor_loss": actor_loss_value,
    }


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
    envs = make_envs(benchmark, args.seed, use_one_hot=use_one_hot_wrapper)

    NUM_TASKS = len(benchmark.train_classes)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # agent setup
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
        n_envs=envs.num_envs,
    )

    global_episodic_return = deque([], maxlen=20 * envs.num_envs)
    global_episodic_length = deque([], maxlen=20 * envs.num_envs)

    obs, _ = envs.reset()

    network_args = {
        "embedding_dim": args.embedding_dim,
        "module_dim": args.module_dim,
        "num_layers": args.num_layers,
        "num_modules": args.num_modules,
    }
    just_obs, task_id = jax.device_put(split_obs_task_id(obs, NUM_TASKS))
    random_action = jnp.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

    def _make_optimizer(lr: float, max_grad_norm: float = 0.0):
        optim = optax.adam(learning_rate=lr)
        if max_grad_norm != 0:
            optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optim,
            )
        return optim

    actor_network = Actor(num_actions=np.prod(envs.single_action_space.shape), **network_args)
    key, actor_init_key = jax.random.split(key)
    actor = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network.init(actor_init_key, just_obs, task_id),
        tx=_make_optimizer(args.policy_lr, args.clip_grad_norm),
    )

    q_network = Critic(**network_args)
    key, qf1_init_key, qf2_init_key = jax.random.split(key, 3)
    qf1 = CriticTrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(qf1_init_key, just_obs, random_action, task_id),
        target_params=q_network.init(qf1_init_key, just_obs, random_action, task_id),
        tx=_make_optimizer(args.q_lr, args.clip_grad_norm),
    )
    qf2 = CriticTrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(qf2_init_key, just_obs, random_action, task_id),
        target_params=q_network.init(qf2_init_key, just_obs, random_action, task_id),
        tx=_make_optimizer(args.q_lr, args.clip_grad_norm),
    )

    alpha_train_state = TrainState.create(
        apply_fn=get_alpha,
        params=jnp.zeros(NUM_TASKS),  # Log alpha
        tx=_make_optimizer(args.q_lr, max_grad_norm=0.0),
    )
    target_entropy = -np.prod(envs.single_action_space.shape).item()

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            key, action_key = jax.random.split(key)
            actions = jax.device_get(get_action(actor, jax.device_put(obs), NUM_TASKS, action_key))

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
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

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
            if args.save_model:
                model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
                with open(model_path + "_qf1", "wb") as f:
                    f.write(flax.serialization.to_bytes(qf1.params))
                with open(model_path + "_qf2", "wb") as f:
                    f.write(flax.serialization.to_bytes(qf2.params))
                with open(model_path + "_actor", "wb") as f:
                    f.write(flax.serialization.to_bytes(actor.params))
                print(f"model saved to {model_path}")

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # Sample a batch from replay buffer
            data = rb.sample(args.batch_size)
            data = jax.tree_map(lambda x: jax.device_put(x.numpy()), data)
            observations, task_ids = split_obs_task_id(data.observations, NUM_TASKS)
            next_observations, _ = split_obs_task_id(data.next_observations, NUM_TASKS)
            batch = Batch(observations, data.actions, data.rewards, next_observations, data.dones, task_ids)
            logs = {}

            # Update alpha  # NOTE Soft-modules specific: this goes first
            key, alpha_update_key = jax.random.split(key)
            alpha_train_state, alpha_logs = update_alpha(actor, alpha_train_state, batch, target_entropy, key)
            logs = {**logs, **alpha_logs}

            # Get task weights & alpha
            log_alpha = alpha_train_state.params
            alpha = alpha_train_state.apply_fn(log_alpha, batch.task_ids)
            task_weights = extract_task_weights(log_alpha, batch.task_ids)

            # Update Q networks
            key, q_update_key = jax.random.split(key)
            (qf1, qf2), critic_logs = update_critic(
                actor, (qf1, qf2), batch, alpha, task_weights, args.gamma, q_update_key
            )
            logs = {**logs, **critic_logs}

            # Update policy network
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    key, actor_update_key = jax.random.split(key)
                    actor, actor_logs = update_actor(actor, (qf1, qf2), batch, alpha, task_weights, actor_update_key)
                    logs = {**logs, **actor_logs}

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                qf1 = qf1.replace(target_params=optax.incremental_update(qf1.params, qf1.target_params, args.tau))
                qf2 = qf2.replace(target_params=optax.incremental_update(qf2.params, qf2.target_params, args.tau))

            # Logging
            if global_step % 100 == 0:
                logs = jax.device_get(logs)
                for _key, value in logs.items():
                    writer.add_scalar(_key, value, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # Evaluation
            if global_step % args.evaluation_frequency == 0:
                print("Evaluating...")
                eval_agent = Agent(actor, num_tasks=NUM_TASKS)
                eval_success_rate = evaluation_procedure(
                    num_envs=len(envs.envs),
                    num_workers=args.evaluation_num_workers,
                    num_episodes=args.evaluation_num_episodes,
                    writer=writer,
                    agent=eval_agent,
                    update=global_step,
                    keys=list(benchmark.train_classes.keys()),
                    classes=benchmark.train_classes,
                    tasks=benchmark.train_tasks,
                )
                print(f"Evaluation success_rate: {eval_success_rate:.4f}")

    envs.close()
    writer.close()
