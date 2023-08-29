import argparse
import os
import random
import time
from distutils.util import strtobool

from flax.training.train_state import TrainState

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


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Metaworld-CleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # RL^2 arguments
    parser.add_argument("--n-episodes-per-trial", type=int, default=80,
                        help="number of episodes sampled per trial/meta-batch")
    parser.add_argument("--recurrent-state-size", type=int, default=128)
    parser.add_argument("--encoder-hidden-size", type=int, default=128)
    parser.add_argument("--mini-batch-size", type=int, default=6)
    parser.add_argument("--recurrent-num-layers", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="maximum number of timesteps in one episode during training")

    parser.add_argument("--use-gae", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

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
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--eval-freq", type=int, default=100_000,
        help="how many steps to do before evaluating the agent")
    parser.add_argument("--evaluation-num-episodes", type=int, default=10,
        help="the number episodes to run per evaluation")

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
    def __call__(self, x):
        x = nn.Dense(
            self.args.recurrent_state_size,
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.constant(0.1),
        )(x)

        x = nn.relu(x)
        x = nn.Dense(
            400,
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.constant(0.1),
        )(x)
        x = nn.relu(x)
        return nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1),
            bias_init=nn.initializers.constant(0.0),
        )(x)


class Actor(nn.Module):
    args: dict
    action_dim: int

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Dense(
            400,
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.constant(0.1),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            400,
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.constant(0.1),
        )(x)
        x = nn.relu(x)

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

    def initialize_state(self, batch_size):
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
            out
        )
        value = Critic(self.args)(out)
        return dist, value, carry


@jax.jit
def get_action_log_prob_and_value(
    agent_state: TrainState, obs: jax.Array, carry: jax.Array, key
):
    key, action_key = jax.random.split(key)
    action_dist, value, carry = agent_state.apply_fn(agent_state.params, obs, carry)
    action, log_prob = action_dist.sample_and_log_prob(seed=action_key)
    return action, log_prob.reshape(-1, 1), value, key


@jax.jit
def get_deterministic_action(
    agent_state: TrainState, obs: jax.Array, carry: jax.Array
) -> jax.Array:
    action_dist, _, carry = agent_state.apply_fn(agent_state.params, obs, carry)
    return jnp.tanh(action_dist.distribution.mean()), carry


def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    max_episode_steps: Optional[int] = None,
    terminate_on_success: bool = False,
    train=True,
) -> gym.vector.VectorEnv:
    def init_each_env(env_cls: Type[SawyerXYZEnv], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env.toggle_terminate_on_success(terminate_on_success)
        env = metaworld_wrappers.RL2Env(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if train:
            tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        else:
            tasks = [task for task in benchmark.test_tasks if task.env_name == name]
        env = metaworld_wrappers.RandomTaskSelectWrapper(env, tasks)
        env.unwrapped.seed(seed)
        return env

    if train:
        classes = benchmark.train_classes
    else:
        classes = benchmark.test_classes

    return gym.vector.SyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id)
            for env_id, (name, env_cls) in enumerate(classes.items())
        ]
    )


make_envs = partial(_make_envs_common, terminate_on_success=False)
make_eval_envs = partial(_make_envs_common, terminate_on_success=True)


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


def rl2_evaluation(
    args,
    agent,
    agent_state,
    eval_envs: gym.vector.VectorEnv,
    num_episodes: int,
):
    obs, _ = eval_envs.reset()
    NUM_TASKS = eval_envs.num_envs

    successes = np.zeros(NUM_TASKS)
    episodic_returns = [[] for _ in range(NUM_TASKS)]

    start_time = time.time()
    carry = agent.initialize_state(NUM_TASKS)

    while not all(len(returns) >= num_episodes for returns in episodic_returns):
        action, carry = get_deterministic_action(agent_state, obs, carry)
        action = jax.device_get(action)
        obs, reward, _, _, infos = eval_envs.step(action)

        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                # reset states of finished episodes
                carry = carry.at[:, i : i + 1].set(
                    np.random.random(carry[:, i : i + 1].shape)
                )
                episodic_returns[i].append(float(info["episode"]["r"][0]))
                if len(episodic_returns[i]) <= num_episodes:
                    successes[i] += int(info["success"])

    episodic_returns = [returns[:num_episodes] for returns in episodic_returns]

    print(f"Evaluation time: {time.time() - start_time:.2f}s")

    success_rate_per_task = successes / num_episodes
    mean_success_rate = np.mean(success_rate_per_task)
    mean_returns = np.mean(episodic_returns)

    return mean_success_rate, mean_returns, success_rate_per_task


def update_rl2_ppo(
    args,
    agent_state: TrainState,
    storage_list: List[Storage],
    key,
):
    @jax.jit
    def ppo_loss(params, obs, actions, advantages, logprob, returns, subkey):
        carry = jnp.zeros((obs.shape[0], args.recurrent_state_size))
        action_dist, newvalue, carry = agent_state.apply_fn(params, obs, carry)
        action, newlog_prob = action_dist.sample_and_log_prob(seed=subkey)
        logratio = newlog_prob.reshape(-1, 1) - logprob
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = -newlog_prob.mean()

        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
        return loss, {
            "loss": loss,
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": jax.lax.stop_gradient(approx_kl),
        }

    # combine episodes into trials
    _obs = jnp.stack([strg.obs for strg in storage_list])
    _actions = jnp.stack([strg.actions for strg in storage_list])
    _logprobs = jnp.stack([strg.logprobs for strg in storage_list])
    _dones = jnp.stack([strg.dones for strg in storage_list])
    _values = jnp.stack([strg.values for strg in storage_list])
    _advantages = jnp.stack([strg.advantages for strg in storage_list])
    _returns = jnp.stack([strg.returns for strg in storage_list])
    _rewards = jnp.stack([strg.rewards for strg in storage_list])

    obs_batch = _obs.reshape(-1, args.max_episode_steps, _obs.shape[-1])
    action_batch = _actions.reshape(-1, args.max_episode_steps, _actions.shape[-1])
    advantage_batch = _advantages.reshape(-1, args.max_episode_steps, 1)
    logprob_batch = _logprobs.reshape(-1, args.max_episode_steps, 1)
    return_batch = _returns.reshape(-1, args.max_episode_steps, 1)

    for epoch in range(args.update_epochs):
        key, subkey = jax.random.split(key)
        args.n_episodes_per_trial
        b_inds = jax.random.permutation(subkey, args.mini_batch_size, independent=True)
        for start in range(0, args.n_episodes_per_trial, args.mini_batch_size):
            end = start + args.mini_batch_size
            mb_inds = np.array(b_inds[start:end])
            grads, aux_metrics = jax.grad(ppo_loss, has_aux=True)(
                agent_state.params,
                obs_batch[mb_inds].reshape(-1, _obs.shape[-1]),
                action_batch[mb_inds].reshape(-1, _actions.shape[-1]),
                advantage_batch[mb_inds].reshape(-1, 1),
                logprob_batch[mb_inds].reshape(-1, 1),
                return_batch[mb_inds].reshape(-1, 1),
                subkey,
            )

            agent_state = agent_state.apply_gradients(grads=grads)

    return agent_state, aux_metrics


# @jax.jit
def compute_gae(
    agent_state: TrainState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
    carry: jax.Array,
):
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    _, next_value, carry = agent_state.apply_fn(agent_state.params, next_obs, carry)
    lastgaelam = 0
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
        lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))

    storage = storage.replace(returns=storage.advantages + storage.values)
    return storage


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

    # env setup
    envs = make_envs(benchmark, args.seed, args.max_episode_steps, train=True)
    keys = list(benchmark.train_classes.keys())

    eval_train_envs = make_eval_envs(
        benchmark,
        args.seed,
        train=True,
    )

    eval_test_envs = make_eval_envs(
        benchmark, args.seed, train=False
    )

    NUM_TASKS = len(benchmark.train_classes)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    obs, _ = envs.reset()

    agent = RL2ActorCritic(envs, args)
    agent.apply = jax.jit(agent.apply)
    agent_state = TrainState.create(
        apply_fn=agent.apply,
        params=agent.init(
            agent_init_key, obs, jnp.zeros((NUM_TASKS, args.recurrent_state_size))
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=args.learning_rate),
        ),
    )

    num_trial_episodes = args.n_episodes_per_trial // NUM_TASKS
    # trial_length = args.max_episode_steps * num_trial_episodes
    storage = Storage(
        obs=jnp.zeros(
            (args.max_episode_steps, NUM_TASKS, *envs.single_observation_space.shape)
        ),
        actions=jnp.zeros(
            (args.max_episode_steps, NUM_TASKS, *envs.single_action_space.shape)
        ),
        logprobs=jnp.zeros((args.max_episode_steps, NUM_TASKS, 1)),
        dones=jnp.zeros((args.max_episode_steps, NUM_TASKS, 1)),
        values=jnp.zeros((args.max_episode_steps, NUM_TASKS, 1)),
        advantages=jnp.zeros((args.max_episode_steps, NUM_TASKS, 1)),
        returns=jnp.zeros((args.max_episode_steps, NUM_TASKS, 1)),
        rewards=jnp.zeros((args.max_episode_steps, NUM_TASKS, 1)),
    )

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    global_episodic_return = deque([], maxlen=20 * NUM_TASKS)
    global_episodic_length = deque([], maxlen=20 * NUM_TASKS)

    total_steps = 0

    for global_step in range(int(args.total_timesteps // args.n_episodes_per_trial)):
        assert args.n_episodes_per_trial > NUM_TASKS
        # collect a trial of n episodes per task
        # https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/default_worker.py
        meta_trial = list()
        for meta_ep in range(1, num_trial_episodes + 1):
            # RL^2 stuff
            # reset hidden state for each meta trial
            carry = agent.initialize_state(NUM_TASKS)
            for meta_step in range(args.max_episode_steps):
                total_steps += NUM_TASKS
                action, logprob, value, key = get_action_log_prob_and_value(
                    agent_state, obs, carry, key
                )
                action = jax.device_get(action)
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, infos = envs.step(action)
                done = np.logical_or(terminated, truncated).reshape(-1, 1)
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
            storage = compute_gae(
                agent_state,
                next_obs,
                done,
                storage,
                carry,
            )
            meta_trial.append(storage)

        if global_step % 500 == 0 and global_episodic_return:
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

        agent_state, logs = update_rl2_ppo(args, agent_state, meta_trial, key)
        logs = jax.device_get(logs)

        if global_step % 100 == 0:
            for _key, value in logs.items():
                writer.add_scalar(_key, value, total_steps)
            print("SPS:", int(total_steps / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(total_steps / (time.time() - start_time)),
                total_steps,
            )

        if total_steps % args.eval_freq == 0 and total_steps > 0:
            print(f"Evaluating on test set at {total_steps=}")
            test_success_rate, eval_returns, eval_success_per_task = rl2_evaluation(
                args, agent, agent_state, eval_test_envs, args.evaluation_num_episodes
            )
            eval_metrics = {
                "charts/mean_test_success_rate": float(test_success_rate),
                "charts/mean_test_return": float(eval_returns),
                **{
                    f"charts/{env_name}_test_success_rate": float(
                        eval_success_per_task[i]
                    )
                    for i, (env_name, _) in enumerate(benchmark.test_classes.items())
                },
            }
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, total_steps)

            print(f"Evaluating on train set at {total_steps=}")
            train_success_rate, train_returns, train_success_per_task = rl2_evaluation(
                args, agent, agent_state, eval_train_envs, args.evaluation_num_episodes
            )
            train_metrics = {
                "charts/mean_train_success_rate": float(train_success_rate),
                "charts/mean_train_return": float(train_returns),
                **{
                    f"charts/{env_name}_train_success_rate": float(
                        train_success_per_task[i]
                    )
                    for i, (env_name, _) in enumerate(benchmark.test_classes.items())
                },
            }
            for k, v in train_metrics.items():
                writer.add_scalar(k, v, total_steps)

            print(
                f"{total_steps=} {test_success_rate=:.4f} "
                + f"{eval_returns=:.4f} "
                + f"{train_success_rate=:.4f} "
                + f"{train_returns=:.4f}"
            )

    envs.close()
    writer.close()
