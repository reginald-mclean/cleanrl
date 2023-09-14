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
sys.path.append('/home/reginaldkmclean/cleanrl')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from cleanrl_utils.evals.meta_world_eval_protocol import eval
import json

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Meta-World Benchmarking",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='reggies-phd-research',
        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MT1", help="the id of the environment")
    parser.add_argument("--env-name", type=str, default="peg-insert-side-v2", help="the name of the environment for MT1")
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
    parser.add_argument("--clip-grad-norm", type=float, default=0,
        help="the value to clip the gradient norm to. Disabled if 0. Not applied to alpha gradients.")
    parser.add_argument("--actor-network", type=str, default="256,256", help="The architecture of the actor network")
    parser.add_argument("--critic-network", type=str, default="256,256", help="The architecture of the critic network")
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


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


class Actor(nn.Module):
    num_actions: int
    num_tasks: int
    hidden_dims: int = "256,256"

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
        return distrax.Transformed(
            distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_sigma)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )


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
def get_deterministic_action(
    actor: TrainState,
    obs: ArrayLike,
    task_ids: ArrayLike,
):
    dist = actor.apply_fn(actor.params, obs, task_ids)
    return jnp.tanh(dist.distribution.mean())


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


class Critic(nn.Module):
    hidden_dims: int = "256,256"
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
    hidden_dims: int = "256,256"

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
        min_qf_next_target = jnp.min(
            q_values, axis=0
        ) - alpha_val * next_action_log_probs.reshape(-1, 1)
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
        return (-log_alpha * (log_probs.reshape(-1, 1) + target_entropy)).mean()

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
        return (_alpha_val * log_probs.reshape(-1, 1) - min_qf_values).mean(), (
            _alpha,
            _critic,
            logs,
        )

    (actor_loss_value, (alpha, critic, logs)), actor_grads = jax.value_and_grad(
        actor_loss, has_aux=True
    )(actor.params)
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    #if args.save_model:  # Orbax checkpoints
    ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=5, create=True, best_fn=lambda x: x["charts/mean_success_rate"]
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_manager = orbax.checkpoint.CheckpointManager(
        f"runs/{run_name}/checkpoints", checkpointer, options=ckpt_options
    )

    env_names = ['push-v2', 'door-open-v2', 'peg-insert-side-v2', 'button-press-topdown-v2', 'window-open-v2', 'window-close-v2', 'drawer-close-v2', 'drawer-open-v2'] # , 'pick-place-v2', reach-v2]
    seeds = [432, 8123, 1029385, 765, 345, 234, 654, 986] # , 1, 4212]
    load_checkpoint = dict()
    #load_checkpoint['reach-v2'] =(4800000, "/home/reginaldkmclean/cleanrl/runs/MT1__mtmhsac_jax__1__1694616820/checkpoints") 
    load_checkpoint['push-v2'] = (400000, "/home/reginaldkmclean/cleanrl/runs/MT1__push__432__1694623435/checkpoints")
    load_checkpoint['door-open-v2'] = (200000, "/home/reginaldkmclean/cleanrl/runs/MT1__mtmhsac_jax__8123__1694616784/checkpoints")
    load_checkpoint['peg-insert-side-v2'] = (1400000, "/home/reginaldkmclean/cleanrl/runs/MT1__mtmhsac_jax__1029385__1694481752/checkpoints")
    load_checkpoint['button-press-topdown-v2'] = (200000, "/home/reginaldkmclean/cleanrl/runs/MT1__button-press-topdown__765__1694623425/checkpoints/")
    load_checkpoint['window-open-v2'] = (200000, "/home/reginaldkmclean/cleanrl/runs/MT1__window-open__345__1694623401/checkpoints")
    load_checkpoint['window-close-v2'] = (200000, "/home/reginaldkmclean/cleanrl/runs/MT1__window-close__234__1694623411/checkpoints")
    load_checkpoint['drawer-close-v2'] = (200000, "/home/reginaldkmclean/cleanrl/runs/MT1__drawer-close__654__1694623380/checkpoints")
    load_checkpoint['drawer-open-v2'] = (2200000, "/home/reginaldkmclean/cleanrl/runs/MT1__drawer-open__986__1694623389/checkpoints")

    store_dict = dict()

    #load_checkpoint['pick-place-v2'] = (, "/home/reginaldkmclean/cleanrl/)
    #ckpt = ckpt_manager.restore(step=200000, directory="/home/reginaldkmclean/cleanrl/runs/MT1__button-press-topdown__765__1694623425/checkpoints/")
    #agent.actor = ckpt["actor"]
    #agent.critic = ckpt["critic"]
    #agent.alpha_train_state = ckpt["alpha"]
    for env_name, seed in list(zip(env_names, seeds)):
        benchmark = metaworld.MT1(env_name, seed=seed)
        #benchmark._train_tasks = benchmark.train_tasks[0]
        use_one_hot_wrapper = True
        envs1 = make_envs(
            benchmark, seed, 500, use_one_hot=use_one_hot_wrapper, terminate_on_success=False
        )

        NUM_TASKS = len(benchmark.train_classes)
        #print(NUM_TASKS)
        obs, _ = envs1.reset()
        key = jax.random.PRNGKey(seed)
        key, agent_init_key = jax.random.split(key)
        agent1 = Agent(
            init_obs=obs,
            num_tasks=NUM_TASKS,
            action_space=envs1.single_action_space,
            policy_lr=args.policy_lr,
            q_lr=args.q_lr,
            gamma=args.gamma,
            clip_grad_norm=args.clip_grad_norm,
            init_key=key,
        )
        #print(env_name)
        ckpt = ckpt_manager.restore(step=load_checkpoint[env_name][0], directory=load_checkpoint[env_name][1])
        agent1.actor = agent1.actor.replace(params=ckpt["actor"]['params'])
        agent1.critic = agent1.critic.replace(params=ckpt["critic"]['params'])
        agent1.alpha_train_state = agent1.alpha_train_state.replace(params=ckpt["alpha"]['params'])
        #print(ckpt["actor"])
        for e_name2, seed2 in list(zip(env_names, seeds)):
            if env_name == e_name2:
                continue
            print(env_name, e_name2)
            benchmark2 = metaworld.MT1(e_name2, seed=seed2)
            #benchmark._train_tasks = benchmark.train_tasks[0]
            use_one_hot_wrapper = True
            envs2 = make_envs(
                benchmark2, seed2, 500, use_one_hot=use_one_hot_wrapper, terminate_on_success=False
            )

            NUM_TASKS = len(benchmark.train_classes)
            #print(NUM_TASKS)
            #print(e_name2, seed2)
            obs, _ = envs2.reset()
            key2 = jax.random.PRNGKey(seed2)
            key2, agent_init_key = jax.random.split(key2)
            agent2 = Agent(
                init_obs=obs,
                num_tasks=NUM_TASKS,
                action_space=envs2.single_action_space,
                policy_lr=args.policy_lr,
                q_lr=args.q_lr,
                gamma=args.gamma,
                clip_grad_norm=args.clip_grad_norm,
                init_key=key2,
            )
            ckpt2 = ckpt_manager.restore(step=load_checkpoint[e_name2][0], directory=load_checkpoint[e_name2][1])
            agent2.actor = agent2.actor.replace(params=ckpt2["actor"]['params'])
            agent2.critic = agent2.critic.replace(params=ckpt2["critic"]['params'])
            agent2.alpha_train_state = agent2.alpha_train_state.replace(params=ckpt2["alpha"]['params'])
            store_dict[env_name + ' ' + e_name2] = dict()
            store_dict[env_name + ' ' + e_name2][env_name] = {i: [] for i in range(500)}
            store_dict[env_name + ' ' + e_name2][e_name2] = {i: [] for i in range(500)}
            for _ in range(1):
                done1 = False
                done2 = False
                count = 0
                obs1, _ = envs1.reset()
                obs2, _ = envs2.reset() 
                while (not done1 or not done2) and count < 500:
                    if not done1:
                        a1 = agent1.get_action_eval(obs1)
                        next1, reward1, trunc1, term1, info1 = envs1.step(a1)
                        store_dict[env_name + ' ' + e_name2][env_name][count].append([a1, obs1, next1, reward1, trunc1, term1, info1, int(info1['success'][0])==1])
                        done1 = int(info1['success'][0]) == 1
                        obs1 = next1
                        if done1:
                            print(f'{env_name} done1 at count {count}')
                    if not done2:
                        a2 = agent2.get_action_eval(obs2)
                        next2, reward2, trunc2, term2, info2 = envs2.step(a2)
                        #if 'success' in info2:
                        #print(info2['success'][0], count)
                        store_dict[env_name + ' ' + e_name2][e_name2][count].append([a2, obs2, next2, reward2, trunc2, term2, info2, int(info2['success'][0])==1])
                        done2 = int(info2['success'][0]) == 1
                        obs2 = next2
                        if done2:
                            print(f'{e_name2} done2 at count {count}')
                    count += 1

data = json.dumps(store_dict)
with open('mtmhsac_jax_trajectories.json', 'w') as f:
    f.write(data)

data2 = None
with open('mtmhsac_jax_trajectories.json') as f:
    data2 = f.read()

def findDiff(d1, d2, path=""):
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                findDiff(d1[k],d2[k], "%s -> %s" % (path, k) if path else k)
            if d1[k] != d2[k]:
                result = [ "%s: " % path, " - %s : %s" % (k, d1[k]) , " + %s : %s" % (k, d2[k])]
                print("\n".join(result))
        else:
            print ("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))
findDiff(data, data1)
