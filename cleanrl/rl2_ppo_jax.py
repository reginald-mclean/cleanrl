import argparse
import os
import random
import time
from distutils.util import strtobool

from flax.training.train_state import TrainState

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from typing import Callable, List, Optional, Tuple, Type

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import metaworld
import numpy as np
import optax
import orbax
import orbax.checkpoint
from flax.core.frozen_dict import FrozenDict
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers_metaworld import MultiTaskRolloutBuffer, Rollout
from cleanrl_utils.wrappers import metaworld_wrappers

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
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")

    # RL^2 arguments
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--num-parallel-envs", type=int, default=10,
                        help="the number of parallel envs used to collect data")
    parser.add_argument("--num-episodes-per-trial", type=int, default=10,
                        help="the number episodes collected for each env before training, the trial size")
    parser.add_argument("--trial-batch-size", type=int, default=10)
    parser.add_argument("--recurrent-state-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--recurrent-type", type=str, default="gru")

    parser.add_argument("--reward-version", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop"])

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ML10",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=25e6,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.02,
        help="coefficient of the entropy")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--norm-advantage", type=bool, default=False)

    parser.add_argument("--eval-freq", type=int, default=100_000,
        help="how many steps to do before evaluating the agent")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes to run per evaluation")
    parser.add_argument("--num-evaluation-goals", type=int, default=10,
        help="the number of goal positions to evaluate on per test task")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    args = parser.parse_args()
    # fmt: on
    return args


def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    num_parallel_envs: int,
    max_episode_steps: Optional[int] = None,
    split: str = "train",
    terminate_on_success: bool = False,
    task_select: str = "random",
    total_tasks_per_class: Optional[int] = None,
    reward_version="v1",
) -> Tuple[gym.vector.VectorEnv, List[str]]:
    all_classes = benchmark.train_classes if split == "train" else benchmark.test_classes
    all_tasks = benchmark.train_tasks if split == "train" else benchmark.test_tasks
    envs_per_task = len(all_classes)
    assert num_parallel_envs % envs_per_task == 0, f"{num_parallel_envs=} must be divisible by {envs_per_task=}"
    tasks_per_env = num_parallel_envs // len(all_classes)

    def make_env(env_cls: Type[SawyerXYZEnv], tasks: list) -> gym.Env:
        env = env_cls(reward_func_version=reward_version)
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.unwrapped.max_path_length)
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
        gym.vector.AsyncVectorEnv([partial(make_env, env_cls=env_cls, tasks=tasks) for env_cls, tasks in env_tuples]),
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


def rl2_evaluation(
    agent,
    agent_state,
    eval_envs: gym.vector.VectorEnv,
    num_episodes: int,
    task_names,
):
    NUM_ENVS = eval_envs.unwrapped.num_envs
    successes = np.zeros(NUM_ENVS)
    episodic_returns = [[] for _ in range(NUM_ENVS)]
    if task_names is not None:
        successes = {task_name: 0 for task_name in set(task_names)}
        episodic_returns = {task_name: [] for task_name in set(task_names)}
        envs_per_task = {task_name: task_names.count(task_name) for task_name in set(task_names)}
    else:
        successes = np.zeros(eval_envs.num_envs)
        episodic_returns = [[] for _ in range(eval_envs.num_envs)]

    start_time = time.time()
    carry = agent.initialize_carry((NUM_ENVS,))

    eval_envs.call("toggle_sample_tasks_on_reset", False)
    eval_envs.call("toggle_terminate_on_success", True)
    obs, _ = zip(*eval_envs.call("sample_tasks"))
    obs = np.stack(obs)

    def eval_done(returns):
        if isinstance(returns, dict):
            return all(len(r) >= (num_episodes * envs_per_task[task_name]) for task_name, r in returns.items())
        else:
            return all(len(r) >= num_episodes for r in returns)

    while not eval_done(episodic_returns):
        action, carry = exploit(agent_state, obs, carry)
        action = jax.device_get(action)
        obs, _, _, _, infos = eval_envs.step(action)
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                if task_names is not None:
                    episodic_returns[task_names[i]].append(float(info["episode"]["r"][0]))
                    if len(episodic_returns[task_names[i]]) <= num_episodes * envs_per_task[task_names[i]]:
                        successes[task_names[i]] += int(info["success"])
                else:
                    episodic_returns[i].append(float(info["episode"]["r"][0]))
                    if len(episodic_returns[i]) <= num_episodes:
                        successes[i] += int(info["success"])

    if isinstance(episodic_returns, dict):
        episodic_returns = {
            task_name: returns[: (num_episodes * envs_per_task[task_name])] for task_name, returns in episodic_returns.items()
        }
    else:
        episodic_returns = [returns[:num_episodes] for returns in episodic_returns]

    if isinstance(successes, dict):
        success_rate_per_task = np.array(
            [successes[task_name] / (num_episodes * envs_per_task[task_name]) for task_name in set(task_names)]
        )
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(list(episodic_returns.values()))
    else:
        success_rate_per_task = successes / num_episodes
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(episodic_returns)

    task_success_rates = {task_name: success_rate_per_task[i] for i, task_name in enumerate(set(task_names))}

    print(f"Evaluation time: {time.time() - start_time:.2f}s")
    return mean_success_rate, mean_returns, task_success_rates


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

    return _init


class RL2Policy(nn.Module):
    envs: gym.Env
    args: argparse.Namespace

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    def initialize_carry(self, batch_seq_len: Tuple):
        carry_shape = batch_seq_len + (self.args.recurrent_state_size,)
        if self.args.recurrent_type == "gru":
            return nn.GRUCell(self.args.recurrent_state_size, parent=None).initialize_carry(jax.random.PRNGKey(0), carry_shape)
        elif self.args.recurrent_type == "lstm":
            return nn.OptimizedLSTMCell(self.args.recurrent_state_size, parent=None).initialize_carry(
                jax.random.PRNGKey(0), carry_shape
            )

    @nn.compact
    def __call__(self, x, carry) -> Tuple[distrax.Distribution, jax.Array, jax.Array]:
        if self.args.recurrent_type == "gru":
            carry, x = nn.RNN(nn.GRUCell(features=self.args.recurrent_state_size), return_carry=True)(
                x, initial_carry=carry, time_major=False
            )
        elif self.args.recurrent_type == "lstm":
            carry, x = nn.RNN(
                nn.OptimizedLSTMCell(features=self.args.recurrent_state_size),
                return_carry=True,
            )(x, initial_carry=carry, time_major=False)

        x = nn.Dense(args.hidden_size)(x)
        x = nn.relu(x)

        log_sigma = nn.Dense(
            np.array(self.envs.single_action_space.shape).prod(),
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)

        mu = nn.Dense(
            np.array(self.envs.single_action_space.shape).prod(),
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)
        log_sigma = jax.lax.clamp(self.LOG_STD_MIN, log_sigma, self.LOG_STD_MAX)
        return mu, log_sigma, carry


@jax.jit
def exploit(
    agent_state: TrainState,
    obs: jax.Array,
    carry: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    mu, _, carry = agent_state.apply_fn(agent_state.params, obs, carry)
    # get last output of RNN
    mu = mu[:, -1]
    # action_dist = distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_sigma))
    # return action_dist.mode(), carry
    return mu, carry


@jax.jit
def sample_action_logprob(
    agent_state: TrainState,
    obs: jax.Array,
    carry: jax.Array,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    mu, log_sigma, carry = agent_state.apply_fn(agent_state.params, jnp.expand_dims(obs, axis=1), carry)
    # get last output of RNN
    mu = jnp.squeeze(mu, axis=1)
    log_sigma = jnp.squeeze(log_sigma, axis=1)
    action_dist = distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_sigma))
    action, log_prob = action_dist.sample_and_log_prob(seed=action_key)
    return action, log_prob.reshape(-1, 1), carry, key


def update_rl2_ppo(
    args,
    agent: RL2Policy,
    agent_state: TrainState,
    meta_rollouts: Rollout,
    key: jax.random.PRNGKeyArray,
) -> Tuple[TrainState, dict]:
    @jax.jit
    def ppo_loss(
        params: FrozenDict,
        observations: jax.Array,
        actions: jax.Array,
        advantages: jax.Array,
        log_probs: jax.Array,
        carry: Tuple[jax.Array, jax.Array],
    ) -> Tuple[jax.Array, dict]:
        mu, log_sigma, carry = agent_state.apply_fn(params, observations, carry)
        action_dist = distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_sigma))
        newlog_prob = action_dist.log_prob(actions)
        logratio = newlog_prob[..., None] - log_probs
        ratio = jnp.exp(logratio)

        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1.0) - logratio).mean()
        clip_fraction = (jnp.absolute(ratio - 1.0) > args.clip_coef).mean()

        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * jax.lax.clamp(1.0 - args.clip_coef, ratio, 1.0 + args.clip_coef)
        pg_loss = -jnp.minimum(pg_loss1, pg_loss2).mean()

        entropy_loss = action_dist.entropy().mean()
        loss = pg_loss - args.ent_coef * entropy_loss

        return loss, {
            "losses/loss": loss,
            "losses/pg_loss": pg_loss,
            "losses/entropy_loss": entropy_loss,
            "losses/approx_kl": approx_kl,
            "losses/old_approx_kl": old_approx_kl,
            "losses/clip_fraction": clip_fraction,
        }

    for _ in range(args.update_epochs):
        key, subkey = jax.random.split(key)
        # shuffle batch of the meta trials
        rollout_inds = jax.random.permutation(subkey, len(meta_rollouts), independent=True)
        for i in rollout_inds:
            rollouts = meta_rollouts[i]
            carry = agent.initialize_carry((rollouts.observations.shape[0],))
            grads, aux_metrics = jax.grad(ppo_loss, has_aux=True)(
                agent_state.params,
                rollouts.observations,
                rollouts.actions,
                rollouts.advantages,
                rollouts.log_probs,
                carry,
            )
            agent_state = agent_state.apply_gradients(grads=grads)

        if args.target_kl is not None:
            if aux_metrics["losses/approx_kl"].item() > args.target_kl:
                break
    return agent_state, aux_metrics


class LinearFeatureBaseline:
    @staticmethod
    def _extract_features(obs: np.ndarray):
        obs = np.clip(obs, -10, 10)
        ones = jnp.ones((*obs.shape[:-1], 1))
        time_step = ones * (np.arange(obs.shape[-2]).reshape(-1, 1) / 100.0)
        features = np.concatenate([obs, obs**2, time_step, time_step**2, time_step**3, ones], axis=-1)
        return features

    @classmethod
    def _fit_baseline(cls, obs: np.ndarray, returns: np.ndarray, reg_coeff: float = 1e-5) -> np.ndarray:
        features = cls._extract_features(obs)
        target = returns.reshape(returns.shape[0], -1, 1)

        coeffs = []
        for task in range(obs.shape[0]):
            featmat = features[task]
            _target = target[task]
            for _ in range(5):
                task_coeffs = np.linalg.lstsq(
                    featmat.T @ featmat + reg_coeff * np.identity(featmat.shape[1]),
                    featmat.T @ _target,
                    rcond=-1,
                )[0]
                if not np.any(np.isnan(task_coeffs)):
                    break
                reg_coeff *= 10

            coeffs.append(task_coeffs)

        return np.stack(coeffs)

    @classmethod
    def fit_baseline(cls, rollouts: Rollout) -> Callable[[Rollout], np.ndarray]:
        coeffs = cls._fit_baseline(rollouts.observations, rollouts.returns)

        def baseline(rollouts: Rollout) -> np.ndarray:
            features = cls._extract_features(rollouts.observations)
            return features @ coeffs

        return baseline


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
    key, agent_init_key = jax.random.split(key)

    if args.env_id == "ML10":
        benchmark = metaworld.ML10(seed=args.seed)
    elif args.env_id == "ML45":
        benchmark = metaworld.ML45(seed=args.seed)
    else:
        benchmark = metaworld.ML1(args.env_id, seed=args.seed)

    # env setup
    envs, train_task_names = make_envs(
        benchmark,
        num_parallel_envs=args.num_parallel_envs,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        reward_version=args.reward_version,
    )
    eval_envs, eval_task_names = make_eval_envs(
        benchmark=benchmark,
        num_parallel_envs=args.num_parallel_envs,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        total_tasks_per_class=args.num_evaluation_goals,
        reward_version=args.reward_version,
    )

    envs.single_observation_space.dtype = np.float32
    obs, _ = zip(*envs.call("sample_tasks"))
    obs = np.stack(obs)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = RL2Policy(envs, args)
    fit_baseline = LinearFeatureBaseline.fit_baseline
    dummy_carry = agent.initialize_carry((args.num_parallel_envs,))
    print(
        agent.tabulate(
            agent_init_key,
            obs,
            dummy_carry,
        )
    )
    agent.apply = jax.jit(agent.apply)
    agent_state = TrainState.create(
        apply_fn=agent.apply,
        params=agent.init(agent_init_key, obs, dummy_carry),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            getattr(optax, args.optimizer)(learning_rate=args.learning_rate),
        ),
    )

    if args.save_model:
        ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=5, create=True, best_fn=lambda x: x["charts/mean_success_rate"]
        )
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt_manager = orbax.checkpoint.CheckpointManager(f"runs/{run_name}/checkpoints", checkpointer, options=ckpt_options)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()

    envs.single_observation_space.dtype = np.float32
    buffer = MultiTaskRolloutBuffer(
        num_tasks=args.num_parallel_envs,
        rollouts_per_task=args.num_episodes_per_trial,
        max_episode_steps=args.max_episode_steps,
    )

    buffer_processing_kwargs = {
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "fit_baseline": fit_baseline,
        "normalize_advantages": args.norm_advantage,
        "trial_mode": True,
    }
    total_steps = 0
    steps_per_iter = args.num_parallel_envs * args.num_episodes_per_trial * args.max_episode_steps * args.trial_batch_size
    n_iters = int(args.total_timesteps // steps_per_iter)
    for _iter in range(n_iters):
        print(f"Iteration {_iter} {total_steps=}")
        meta_trial_batch: List[Rollout] = []
        for _ in range(args.trial_batch_size):
            carry = agent.initialize_carry((args.num_parallel_envs,))
            while not buffer.ready:
                action, logprob, carry, key = sample_action_logprob(agent_state, obs, carry, key)
                action = jax.device_get(action)
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, infos = envs.step(action)
                total_steps += args.num_parallel_envs

                buffer.push(obs, action, reward, truncated, log_prob=jax.device_get(logprob))
                obs = next_obs

            # Collect episode batch https://github.com/rlworkgroup/garage/blob/master/src/garage/_dtypes.py#L455
            rollouts = buffer.get(**buffer_processing_kwargs)
            meta_trial_batch.append(rollouts)
            buffer.reset()

        mean_episodic_return = meta_trial_batch[-1].episode_returns.mean()
        writer.add_scalar("charts/mean_episodic_return", mean_episodic_return, total_steps)
        print(f"{total_steps=}, {mean_episodic_return=}")

        agent_state, logs = update_rl2_ppo(args, agent, agent_state, meta_trial_batch, key)
        logs = jax.tree_util.tree_map(lambda x: jax.device_get(x).item(), logs)

        if total_steps % args.eval_freq == 0 and total_steps > 0:
            num_evals = (len(benchmark.test_classes) * args.num_evaluation_goals) // args.num_parallel_envs
            print(f"Evaluating on test tasks at {total_steps=}")
            test_success_rate, eval_returns, eval_success_per_task = rl2_evaluation(
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
            num_evals = (len(benchmark.train_classes) * args.num_evaluation_goals) // args.num_parallel_envs
            _, _, train_success_per_task = rl2_evaluation(
                agent,
                agent_state,
                envs,
                num_episodes=args.evaluation_num_episodes,
                task_names=train_task_names,
            )
            for task_name, success_rate in train_success_per_task.items():
                logs[f"charts/{task_name}_train_success_rate"] = float(success_rate)
            envs.call("toggle_terminate_on_success", False)

            if args.save_model:
                ckpt_manager.save(
                    step=total_steps,
                    items={
                        "agent_state": agent_state,
                        "key": key,
                        "total_steps": total_steps,
                    },
                    metrics=logs,
                )

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
