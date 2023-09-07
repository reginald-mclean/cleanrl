import argparse
import os
import random
import time
from distutils.util import strtobool

from flax.training.train_state import TrainState

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from cleanrl_utils.wrappers import metaworld_wrappers
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Optional, Type, Tuple, List
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
import metaworld
from collections import deque
import distrax
import optax

gym.logger.set_level(40)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Metaworld-CleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # RL^2 arguments
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--meta-batch-size", type=int, default=2,
                        help="Number of episodes to consider as single trial for gradient update")
    parser.add_argument("--num-parallel-envs", type=int, default=10,
        help="the number of parallel envs used to collect data")
    parser.add_argument("--num-episodes-per-env", type=int, default=2,
        help="the number episodes collected for each env before training")
    # agent parameters
    parser.add_argument("--recurrent-state-size", type=int, default=256)
    parser.add_argument("--recurrent-num-layers", type=int, default=1)
    parser.add_argument("--encoder-hidden-size", type=int, default=256)

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ML10",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2e7,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=15,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=2e-3,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")

    parser.add_argument("--eval-freq", type=int, default=100_000,
        help="how many steps to do before evaluating the agent")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes to run per evaluation")
    parser.add_argument("--num-evaluation-goals", type=int, default=10,
        help="the number of goal positions to evaluate on per test task")

    args = parser.parse_args()
    # fmt: on
    return args


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


class Critic(nn.Module):
    args: dict

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(
            1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3)
        )(x)


class Actor(nn.Module):
    args: dict
    action_dim: int

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        log_sigma = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)
        mu = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)
        log_sigma = jnp.clip(log_sigma, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return distrax.Transformed(
            distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_sigma)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )


class RL2ActorCritic(nn.Module):
    envs: gym.Env
    args: dict

    def initialize_state(self, batch_size: int) -> jax.Array:
        return nn.GRUCell(
            features=self.args.recurrent_state_size, parent=None
        ).initialize_carry(
            jax.random.PRNGKey(0),
            (batch_size, self.args.encoder_hidden_size),
        )

    @nn.compact
    def __call__(self, x, carry) -> Tuple[distrax.Distribution, jax.Array, jax.Array]:
        x = nn.Dense(self.args.encoder_hidden_size)(x)
        x = nn.relu(x)
        carry, out = nn.GRUCell(features=self.args.recurrent_state_size)(carry, x)
        dist = Actor(self.args, np.array(self.envs.single_action_space.shape).prod())(
            jnp.concatenate([out, x], axis=-1)
        )
        value = Critic(self.args)(jnp.concatenate([out, x], axis=-1))
        return dist, value, carry


@jax.jit
def get_action_log_prob_and_value(
    agent_state: TrainState, obs: jax.Array, carry: jax.Array, key
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    action_dist, value, carry = agent_state.apply_fn(agent_state.params, obs, carry)
    action, log_prob = action_dist.sample_and_log_prob(seed=action_key)
    return action, log_prob.reshape(-1, 1), value, carry, key


def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    num_parallel_envs: int,
    max_episode_steps: Optional[int] = None,
    split: str = "train",
    terminate_on_success: bool = False,
    task_select: str = "random",
    total_tasks_per_class: Optional[int] = None,
) -> Tuple[gym.vector.VectorEnv, List[str]]:
    all_classes = (
        benchmark.train_classes if split == "train" else benchmark.test_classes
    )
    all_tasks = benchmark.train_tasks if split == "train" else benchmark.test_tasks
    envs_per_task = len(all_classes)
    assert (
        num_parallel_envs % envs_per_task == 0
    ), f"{num_parallel_envs=} must be divisible by {envs_per_task=}"
    tasks_per_env = num_parallel_envs // len(all_classes)

    def make_env(env_cls: Type[SawyerXYZEnv], tasks: list) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(
            env, max_episode_steps or env.unwrapped.max_path_length
        )
        env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env.toggle_terminate_on_success(terminate_on_success)
        env = metaworld_wrappers.RL2Env(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if task_select != "random":
            env = metaworld_wrappers.PseudoRandomTaskSelectWrapper(env, tasks)
        else:
            env = metaworld_wrappers.RandomTaskSelectWrapper(env, tasks)
        env.unwrapped.seed(seed)
        return env

    env_tuples = []
    task_names = []
    for env_name, env_cls in all_classes.items():
        tasks = [task for task in all_tasks if task.env_name == env_name]
        if total_tasks_per_class is not None:
            tasks = tasks[:total_tasks_per_class]
        subenv_tasks = [tasks[i::tasks_per_env] for i in range(0, tasks_per_env)]
        for tasks_for_subenv in subenv_tasks:
            assert (
                len(tasks_for_subenv) == len(tasks) // tasks_per_env
            ), f"{len(tasks_for_subenv)} {len(tasks)} {tasks_per_env}"
            env_tuples.append((env_cls, tasks_for_subenv))
            task_names.append(env_name)

    return (
        gym.vector.AsyncVectorEnv(
            [
                partial(make_env, env_cls=env_cls, tasks=tasks)
                for env_cls, tasks in env_tuples
            ]
        ),
        task_names,
    )


make_envs = partial(
    _make_envs_common,
    terminate_on_success=False,
    task_select="pseudorandom",
    split="train",
)
make_eval_envs = partial(
    _make_envs_common,
    terminate_on_success=True,
    task_select="pseudorandom",
    split="test",
)


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


def get_storage(envs, max_episode_steps, num_envs):
    return Storage(
        obs=jnp.zeros(
            (
                max_episode_steps,
                num_envs,
                *envs.single_observation_space.shape,
            )
        ),
        actions=jnp.zeros(
            (max_episode_steps, num_envs, *envs.single_action_space.shape)
        ),
        logprobs=jnp.zeros((max_episode_steps, num_envs, 1)),
        dones=jnp.zeros((max_episode_steps, num_envs, 1)),
        values=jnp.zeros((max_episode_steps, num_envs, 1)),
        advantages=jnp.zeros((max_episode_steps, num_envs, 1)),
        returns=jnp.zeros((max_episode_steps, num_envs, 1)),
        rewards=jnp.zeros((max_episode_steps, num_envs, 1)),
    )


def rl2_evaluation(
    args,
    agent,
    agent_state,
    eval_envs: gym.vector.VectorEnv,
    num_episodes: int,
    task_names,
):
    NUM_ENVS = eval_envs.unwrapped.num_envs
    key = jax.random.PRNGKey(args.seed)

    successes = np.zeros(NUM_ENVS)
    episodic_returns = [[] for _ in range(NUM_ENVS)]
    if task_names is not None:
        successes = {task_name: 0 for task_name in set(task_names)}
        episodic_returns = {task_name: [] for task_name in set(task_names)}
        envs_per_task = {
            task_name: task_names.count(task_name) for task_name in set(task_names)
        }
    else:
        successes = np.zeros(eval_envs.num_envs)
        episodic_returns = [[] for _ in range(eval_envs.num_envs)]

    start_time = time.time()
    carry = agent.initialize_state(NUM_ENVS)

    eval_envs.call("toggle_sample_tasks_on_reset", False)
    eval_envs.call("toggle_terminate_on_success", True)
    obs, _ = zip(*eval_envs.call("sample_tasks"))
    obs = np.stack(obs)

    def eval_done(returns):
        if type(returns) is dict:
            return all(
                len(r) >= (num_episodes * envs_per_task[task_name])
                for task_name, r in returns.items()
            )
        else:
            return all(len(r) >= num_episodes for r in returns)

    while not eval_done(episodic_returns):
        action, _, _, carry, key = get_action_log_prob_and_value(
            agent_state, obs, carry, key
        )
        action = jax.device_get(action)
        obs, _, _, _, infos = eval_envs.step(action)
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                # reset states of finished episodes
                carry = carry.at[:, i : i + 1].set(
                    np.random.random(carry[:, i : i + 1].shape)
                )
                if task_names is not None:
                    episodic_returns[task_names[i]].append(
                        float(info["episode"]["r"][0])
                    )
                    if (
                        len(episodic_returns[task_names[i]])
                        <= num_episodes * envs_per_task[task_names[i]]
                    ):
                        successes[task_names[i]] += int(info["success"])
                else:
                    episodic_returns[i].append(float(info["episode"]["r"][0]))
                    if len(episodic_returns[i]) <= num_episodes:
                        successes[i] += int(info["success"])

    if isinstance(episodic_returns, dict):
        episodic_returns = {
            task_name: returns[: (num_episodes * envs_per_task[task_name])]
            for task_name, returns in episodic_returns.items()
        }
    else:
        episodic_returns = [returns[:num_episodes] for returns in episodic_returns]

    if isinstance(successes, dict):
        success_rate_per_task = np.array(
            [
                successes[task_name] / (num_episodes * envs_per_task[task_name])
                for task_name in set(task_names)
            ]
        )
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(list(episodic_returns.values()))
    else:
        success_rate_per_task = successes / num_episodes
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(episodic_returns)

    task_success_rates = {
        task_name: success_rate_per_task[i]
        for i, task_name in enumerate(set(task_names))
    }

    print(f"Evaluation time: {time.time() - start_time:.2f}s")
    return mean_success_rate, mean_returns, task_success_rates


def update_rl2_ppo(
    args,
    agent: RL2ActorCritic,
    agent_state: TrainState,
    storage_list: List[Storage],
    key,
):
    @jax.jit
    def ppo_loss(params, obs, actions, advantages, logprob, returns, subkey):
        # init hidden state for single trial
        carry = agent.initialize_state(1)
        action_dist, newvalue, carry = agent_state.apply_fn(params, obs, carry)
        action, newlog_prob = action_dist.sample_and_log_prob(seed=subkey)
        logratio = newlog_prob.reshape(-1, 1) - logprob
        ratio = jnp.exp(logratio)

        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # TODO: check of advantages
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jax.lax.clamp(
            1 - args.clip_coef, ratio, 1 + args.clip_coef
        )
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * jnp.square(newvalue - returns).mean()
        entropy_loss = action_dist.distribution.entropy().mean()
        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

        approx_kl = ((ratio - 1) - logratio).mean()
        clip_fraction = (jnp.absolute(ratio - 1) > args.clip_coef).mean()

        return loss, {
            "losses/loss": loss,
            "losses/pg_loss": pg_loss,
            "losses/v_loss": v_loss,
            "losses/entropy_loss": entropy_loss,
            "losses/approx_kl": approx_kl,
            "losses/clip_fraction": clip_fraction,
        }

    _obs = jnp.concatenate([strg.obs for strg in storage_list], axis=1)
    _actions = jnp.concatenate([strg.actions for strg in storage_list], axis=1)
    _logprobs = jnp.concatenate([strg.logprobs for strg in storage_list], axis=1)
    _dones = jnp.concatenate([strg.dones for strg in storage_list], axis=1)
    _values = jnp.concatenate([strg.values for strg in storage_list], axis=1)
    _advantages = jnp.concatenate([strg.advantages for strg in storage_list], axis=1)
    _returns = jnp.concatenate([strg.returns for strg in storage_list], axis=1)
    _rewards = jnp.concatenate([strg.rewards for strg in storage_list], axis=1)

    episode_length = _obs.shape[0]
    num_trial_episodes = _obs.shape[1]

    for epoch in range(args.update_epochs):
        key, subkey = jax.random.split(key)
        index_permuation = jax.random.permutation(
            subkey, num_trial_episodes, independent=True
        )
        for start_ind in range(0, num_trial_episodes, args.meta_batch_size):
            offset = start_ind + args.meta_batch_size
            b_inds = index_permuation[start_ind:offset]
            trial_length = episode_length * args.meta_batch_size
            grads, aux_metrics = jax.grad(ppo_loss, has_aux=True)(
                agent_state.params,
                # From (episode_length, meta_batch_size, D)
                # to (episode_length * meta_batch_size, -1)
                _obs[:, b_inds].reshape(trial_length, _obs.shape[-1]),
                _actions[:, b_inds].reshape(trial_length, _actions.shape[-1]),
                _advantages[:, b_inds].reshape(trial_length, 1),
                _logprobs[:, b_inds].reshape(trial_length, 1),
                _returns[:, b_inds].reshape(trial_length, 1),
                subkey,
            )
            agent_state = agent_state.apply_gradients(grads=grads)

    return agent_state, aux_metrics


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, agent_init_key = jax.random.split(key)

    if args.env_id == "ML10":
        benchmark = metaworld.ML10(seed=args.seed)
    elif args.env_id == "ML45":
        benchmark = metaworld.ML45(seed=args.seed)
    else:
        benchmark = metaworld.ML1(args.env_id, seed=args.seed)

    NUM_CLASSES = len(benchmark.train_classes)

    # env setup
    envs, train_task_names = make_envs(
        benchmark,
        num_parallel_envs=args.num_parallel_envs,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )
    eval_envs, eval_task_names = make_eval_envs(
        benchmark=benchmark,
        num_parallel_envs=args.num_parallel_envs,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        total_tasks_per_class=args.num_evaluation_goals,
    )

    envs.single_observation_space.dtype = np.float32
    obs, _ = zip(*envs.call("sample_tasks"))
    obs = np.stack(obs)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    @jax.jit
    def compute_gae(
        next_value: np.ndarray,
        next_done: np.ndarray,
        _storage: Storage,
    ):
        storage = _storage.replace(advantages=_storage.advantages.at[:].set(0.0))
        gae = 0
        for t in reversed(range(args.max_episode_steps)):
            if t == args.max_episode_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                nextvalues = storage.values[t + 1]
            delta = (
                storage.rewards[t]
                + args.gamma * nextvalues * nextnonterminal
                - storage.values[t]
            )
            gae = delta + args.gamma * args.gae_lambda * nextnonterminal * gae
            storage = storage.replace(advantages=storage.advantages.at[t].set(gae))
        return storage.replace(returns=storage.advantages + storage.values)

    agent = RL2ActorCritic(envs, args)
    agent.apply = jax.jit(agent.apply)
    agent_state = TrainState.create(
        apply_fn=agent.apply,
        params=agent.init(
            agent_init_key,
            obs,
            jnp.zeros((args.num_parallel_envs, args.recurrent_state_size)),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=args.learning_rate),
        ),
    )

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()

    global_episodic_return = deque([], maxlen=20 * args.num_parallel_envs)
    global_episodic_length = deque([], maxlen=20 * args.num_parallel_envs)

    total_steps = 0

    for global_step in range(int(args.total_timesteps // args.num_parallel_envs)):
        # collect a trial of size meta_batch_size
        # https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/default_worker.py
        meta_trials: List[Storage] = []
        for _ in range(args.num_episodes_per_env):
            carry = agent.initialize_state(args.num_parallel_envs)
            storage = get_storage(envs, args.max_episode_steps, args.num_parallel_envs)
            for meta_step in range(args.max_episode_steps):
                total_steps += args.num_parallel_envs
                action, logprob, value, carry, key = get_action_log_prob_and_value(
                    agent_state, obs, carry, key
                )
                action = jax.device_get(action)
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, infos = envs.step(action)
                # done = np.logical_or(terminated, truncated).reshape(-1, 1)
                done = np.array(truncated).reshape(-1, 1)
                storage = storage.replace(
                    obs=storage.obs.at[meta_step].set(next_obs),
                    dones=storage.dones.at[meta_step].set(done),
                    actions=storage.actions.at[meta_step].set(action),
                    values=storage.values.at[meta_step].set(value),
                    logprobs=storage.logprobs.at[meta_step].set(logprob),
                    rewards=storage.rewards.at[meta_step].set(reward.reshape(-1, 1)),
                )
                obs = next_obs

                # Only print when at least 1 env is done
                if "final_info" not in infos:
                    continue

                for i, info in enumerate(infos["final_info"]):
                    # Skip the envs that are not done
                    if info is None:
                        continue
                    global_episodic_return.append(info["episode"]["r"])
                    global_episodic_length.append(info["episode"]["l"])

            # Collect episode batch https://github.com/rlworkgroup/garage/blob/master/src/garage/_dtypes.py#L455
            _, _, next_value, carry, key = get_action_log_prob_and_value(
                agent_state, obs, carry, key
            )
            storage = compute_gae(next_value, done, storage)
            meta_trials.append(storage)

        if total_steps % 500 == 0 and global_episodic_return:
            mean_ep_return = np.mean(global_episodic_return)
            print(f"{total_steps=}, {mean_ep_return=}")
            writer.add_scalar(
                "charts/mean_episodic_return",
                mean_ep_return,
                total_steps,
            )
            writer.add_scalar(
                "charts/mean_episodic_length",
                np.mean(global_episodic_length),
                total_steps,
            )

        agent_state, logs = update_rl2_ppo(args, agent, agent_state, meta_trials, key)
        logs = jax.tree_util.tree_map(lambda x: jax.device_get(x).item(), logs)

        if total_steps % args.eval_freq == 0 and global_step > 0:
            num_evals = (
                len(benchmark.test_classes) * args.num_evaluation_goals
            ) // args.num_parallel_envs
            print(f"Evaluating on test tasks at {total_steps=}")
            test_success_rate, eval_returns, eval_success_per_task = rl2_evaluation(
                args,
                agent,
                agent_state,
                eval_envs,
                num_episodes=args.evaluation_num_episodes,
                task_names=eval_task_names,
            )
            logs["charts/mean_success_rate"] = float(test_success_rate)
            logs["charts/mean_evaluation_return"] = float(eval_returns)
            for task_name, success_rate in eval_success_per_task.items():
                logs[f"charts/{task_name}_success_rate"] = float(success_rate)

            print(f"Evaluating on train set at {total_steps=}")
            num_evals = (
                len(benchmark.train_classes) * args.num_evaluation_goals
            ) // args.num_parallel_envs
            _, _, train_success_per_task = rl2_evaluation(
                args,
                agent,
                agent_state,
                envs,
                num_episodes=args.evaluation_num_episodes,
                task_names=train_task_names,
            )
            for task_name, success_rate in train_success_per_task.items():
                logs[f"charts/{task_name}_train_success_rate"] = float(success_rate)
            envs.call("toggle_terminate_on_success", False)

        logs = jax.device_get(logs)
        for k, v in logs.items():
            writer.add_scalar(k, v, total_steps)
        print(f"{total_steps=}", logs)

        print("SPS:", int(total_steps / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS",
            int(total_steps / (time.time() - start_time)),
            total_steps,
        )

        # Set tasks for next iteration
        obs, _ = zip(*envs.call("sample_tasks"))
        obs = np.stack(obs)

    envs.close()
    writer.close()
