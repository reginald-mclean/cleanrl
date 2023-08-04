# ruff: noqa: E402
import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import List, Optional, Tuple, Type

os.environ[
    "XLA_PYTHON_CLIENT_PREALLOCATE"
] = "false"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import distrax  # type: ignore
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import metaworld  # type: ignore
import numpy as np
import numpy.typing as npt
import optax  # type: ignore
import orbax.checkpoint  # type: ignore
from cleanrl_utils.buffers_metaworld import MetaLearningReplayBuffer, Trajectory
from cleanrl_utils.evals.metaworld_jax_eval import metalearning_evaluation
from cleanrl_utils.wrappers import metaworld_wrappers
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax.typing import ArrayLike
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--env-id", type=str, default="ML10", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1001,
        help="total number of meta gradient steps")
    parser.add_argument("--max-episode-steps", type=int, default=500,
        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    # ProMP 
    parser.add_argument("--meta-batch-size", type=int, default=10,
        help="the number of tasks to sample and train on in parallel")
    parser.add_argument("--rollouts-per-task", type=int, default=10,
        help="the number of trajectories to collect per task in the meta batch")
    parser.add_argument("--num-layers", type=int, default=2, help="the number of hidden layers in the MLP")
    parser.add_argument("--hidden-dim", type=int, default=64, help="the dimension of each hidden layer in the MLP")
    parser.add_argument("--inner-lr", type=float, default=0.1, help="the inner (adaptation) step size")
    parser.add_argument("--meta-lr", type=float, default=1e-3, help="the meta-policy gradient step size")
    parser.add_argument("--num-promp-steps", type=int, default=5, help="the number of ProMP steps without re-sampling")
    parser.add_argument("--clip-eps", type=float, default=0.3, help="clipping range")
    parser.add_argument("--inner-kl-penalty", type=float, default=5e-4, help="kl penalty parameter eta")
    parser.add_argument("--adaptive-inner-kl-penalty", type=lambda x: bool(strtobool(x)), default=False, nargs="?",
        const=True, help="whether to use an adaptive or fixed KL-penalty coefficient")
    parser.add_argument("--num-inner-gradient-steps", type=int, default=1,
        help="number of inner / adaptation gradient steps")

    args = parser.parse_args()
    # fmt: on
    return args


def make_envs(
    benchmark: metaworld.Benchmark,
    seed: int,
    meta_batch_size: int,
    max_episode_steps: Optional[int] = None,
) -> gym.vector.VectorEnv:
    assert meta_batch_size % len(benchmark.train_classes) == 0, "meta_batch_size must be divisible by envs_per_task"
    tasks_per_env = meta_batch_size // len(benchmark.train_classes)

    def make_env(env_cls: Type[SawyerXYZEnv], tasks: list) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = metaworld_wrappers.PseudoRandomTaskSelectWrapper(env, tasks)
        env.unwrapped.seed(seed)
        return env

    env_tuples = []
    for env_name, env_cls in benchmark.train_classes.items():
        tasks = [task for task in benchmark.train_tasks if task.env_name == env_name]
        subenv_tasks = [tasks[i::tasks_per_env] for i in range(0, tasks_per_env)]
        for tasks_for_subenv in subenv_tasks:
            assert len(tasks_for_subenv) == len(tasks) // tasks_per_env
            env_tuples.append((env_cls, tasks_for_subenv))

    return gym.vector.AsyncVectorEnv([partial(make_env, env_cls=env_cls, tasks=tasks) for env_cls, tasks in env_tuples])


def make_eval_envs(
    benchmark: metaworld.Benchmark,
    seed: int,
    meta_batch_size: int,
    max_episode_steps: Optional[int] = None,
    terminate_on_success: bool = False,
) -> gym.vector.VectorEnv:
    assert meta_batch_size % len(benchmark.test_classes) == 0, "meta_batch_size must be divisible by envs_per_task"
    tasks_per_env = meta_batch_size // len(benchmark.test_classes)

    def make_env(env_cls: Type[SawyerXYZEnv], tasks: list) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = metaworld_wrappers.PseudoRandomTaskSelectWrapper(env, tasks)
        env.unwrapped.seed(seed)
        return env

    env_tuples = []
    for env_name, env_cls in benchmark.test_classes.items():
        tasks = [task for task in benchmark.test_tasks if task.env_name == env_name]
        subenv_tasks = [tasks[i::tasks_per_env] for i in range(0, tasks_per_env)]
        for tasks_for_subenv in subenv_tasks:
            assert len(tasks_for_subenv) == len(tasks) // tasks_per_env
            env_tuples.append((env_cls, tasks_for_subenv))

    return gym.vector.AsyncVectorEnv([partial(make_env, env_cls=env_cls, tasks=tasks) for env_cls, tasks in env_tuples])


# Networks
class MLPTorso(nn.Module):
    """A Flax Module to represent an MLP with tanh activations."""

    num_hidden_layers: int
    output_dim: int
    hidden_dim: int = 64
    activate_last: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim, name=f"layer_{i}")(x)  # type: ignore
            x = nn.tanh(x)
        x = nn.Dense(self.output_dim, name=f"layer_{self.num_hidden_layers}")(x)  # type: ignore
        if self.activate_last:
            x = nn.tanh(x)
        return x


class GaussianPolicy(nn.Module):
    """The Policy network."""

    num_actions: int
    num_layers: int
    hidden_dim: int

    LOG_STD_MIN: float = np.log(1e-6)

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = MLPTorso(
            num_hidden_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            output_dim=2 * self.num_actions,
        )(x)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.maximum(log_std, self.LOG_STD_MIN)
        std = jnp.exp(log_std)
        return mean, std


class MetaVectorPolicy(nn.Module):
    n_tasks: int
    num_actions: int
    num_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        vmap_policy = nn.vmap(
            GaussianPolicy,
            variable_axes={"params": 0},  # parameters not shared between task policies
            in_axes=0,
            out_axes=0,
            axis_size=self.n_tasks,
        )
        mean, std = vmap_policy(num_actions=self.num_actions, num_layers=self.num_layers, hidden_dim=self.hidden_dim)(
            state
        )
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

    @staticmethod
    def init_single(
        num_actions: int,
        num_layers: int,
        hidden_dim: int,
        rng: jax.random.PRNGKeyArray,
        init_args: list,
    ) -> FrozenDict:
        return GaussianPolicy(num_actions=num_actions, num_layers=num_layers, hidden_dim=hidden_dim).init(
            rng, *init_args
        )

    @staticmethod
    def expand_params(params: FrozenDict, axis_size: int) -> FrozenDict:
        inner_params = jax.tree_map(lambda x: jnp.stack([x for _ in range(axis_size)]), params)["params"]
        return FrozenDict({"params": {"VmapGaussianPolicy_0": inner_params}})


@jax.jit
def sample_actions(
    actor: TrainState, obs: ArrayLike, key: jax.random.PRNGKey
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, obs)
    action_samples, action_log_probs = dist.sample_and_log_prob(seed=action_key)
    return action_samples, action_log_probs, dist.loc, dist.scale_diag, key


def get_actions_log_probs_and_dists(
    actor: TrainState, obs: ArrayLike, key: jax.random.PRNGKey
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, jax.random.PRNGKeyArray]:
    actions, log_probs, means, stds, key = sample_actions(actor, obs, key)
    return jax.device_get(actions), jax.device_get(log_probs), jax.device_get(means), jax.device_get(stds), key


class MetaTrainState(TrainState):
    inner_train_state: TrainState


@partial(jax.jit, static_argnames=("return_kl",))
def inner_step(
    policy: TrainState, trajectories: Trajectory, return_kl: bool = False
) -> Tuple[TrainState, Optional[jax.Array]]:
    assert trajectories.log_probs is not None and trajectories.means is not None and trajectories.stds is not None

    def inner_opt_objective(_theta: FrozenDict):  # J^LR, Equation 12
        theta_dist = policy.apply_fn(_theta, trajectories.observations)
        theta_log_probs = theta_dist.log_prob(trajectories.actions)

        if return_kl:
            kl = distrax.MultivariateNormalDiag(loc=trajectories.means, scale_diag=trajectories.stds).kl_divergence(
                theta_dist
            )
        else:
            kl = None

        ratio = jnp.exp(theta_log_probs - trajectories.log_probs)
        return -(ratio * trajectories.advantages).mean(), kl

    grads, kl = jax.grad(inner_opt_objective, has_aux=True)(policy.params)
    updated_policy = policy.apply_gradients(grads=grads)  # Inner gradient step, SGD

    return updated_policy, kl


@partial(jax.jit, static_argnames=("eta", "clip_eps", "num_grad_steps", "num_tasks"))
def outer_step(
    meta_train_state: MetaTrainState,
    all_trajectories: List[Trajectory],
    eta: float,
    clip_eps: float,
    num_grad_steps: int,
    num_tasks: int,
) -> Tuple[MetaTrainState, dict]:
    def promp_loss(theta: FrozenDict):
        vec_theta = MetaVectorPolicy.expand_params(theta, num_tasks)
        inner_train_state = meta_train_state.inner_train_state.replace(params=vec_theta)
        kls = []

        # Adaptation steps using J^LR to go from theta to theta^\prime
        for i in range(len(all_trajectories) - 1):
            trajectories = all_trajectories[i]
            inner_train_state, kl = inner_step(inner_train_state, trajectories, return_kl=True)
            kls.append(kl)

        # Inner Train State now has theta^\prime
        # Compute J^Clip, Equation 11
        trajectories = all_trajectories[-1]
        new_param_dist = inner_train_state.apply_fn(inner_train_state.params, trajectories.observations)
        new_param_log_probs = new_param_dist.log_prob(trajectories.actions)

        likelihood_ratio = jnp.exp(new_param_log_probs - trajectories.log_probs)
        outer_objective = jnp.minimum(
            likelihood_ratio * trajectories.advantages,
            jnp.clip(likelihood_ratio, 1 - clip_eps, 1 + clip_eps) * trajectories.advantages,
        )

        mean_kl = jnp.stack(kls).mean(axis=1).sum()

        return -(outer_objective.mean(axis=1).sum() - eta * mean_kl), mean_kl  # Equation 13

    # Update theta
    loss_before = None
    for i in range(num_grad_steps):
        (loss, inner_kl), grads = jax.value_and_grad(promp_loss)(meta_train_state.params)
        if loss_before is None:
            loss_before = loss
        meta_train_state = meta_train_state.apply_gradients(grads=grads)

    (loss_after, inner_kl) = promp_loss(meta_train_state.params)

    return meta_train_state, {
        "losses/loss_before": jnp.mean(loss_before),
        "losses/loss_after": jnp.mean(loss_after),
        "losses/kl_inner": inner_kl,
    }


# NOTE ProMP Uses a Linear time-dependent return baseline model from
# Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control" as its
# baseline, rather than a neural network used as a learned value function like e.g. PPO.


class LinearFeatureBaseline:
    @staticmethod
    def _extract_features(obs: np.ndarray, reshape=True):
        obs = np.clip(obs, -10, 10)
        ones = jnp.ones((*obs.shape[:-1], 1))
        time_step = ones * (np.arange(obs.shape[-2]).reshape(-1, 1) / 100.0)
        features = np.concatenate([obs, obs**2, time_step, time_step**2, time_step**3, ones], axis=-1)
        if reshape:
            features = features.reshape(features.shape[0], -1, features.shape[-1])
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
    def fit_baseline(cls, trajectories: Trajectory) -> np.ndarray:
        coeffs = cls._fit_baseline(trajectories.observations, trajectories.returns)

        def baseline(obs: np.ndarray) -> np.ndarray:
            features = cls._extract_features(obs, reshape=False)
            return features @ coeffs

        return baseline


class ProMP:
    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        num_layers: int,
        hidden_dim: int,
        init_key: jax.random.PRNGKey,
        init_obs: ArrayLike,
        eta: float,
        clip_eps: float,
        num_grad_steps: int,
    ):
        self.num_tasks = envs.unwrapped.num_envs

        network_args = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_actions": np.prod(envs.single_action_space.shape),
        }

        # Init general parameters theta
        theta = MetaVectorPolicy.init_single(**network_args, rng=init_key, init_args=[init_obs])

        # Init a vectorized policy, running a separate set of parameters for each task
        # The initial parameters for each task are all the same - theta
        policy_network = MetaVectorPolicy(n_tasks=self.num_tasks, **network_args)
        self.policy = TrainState.create(
            apply_fn=policy_network.apply,
            params=MetaVectorPolicy.expand_params(theta, self.num_tasks),
            tx=optax.sgd(learning_rate=args.inner_lr),  # inner optimizer
        )
        self.train_state = MetaTrainState.create(
            apply_fn=lambda x: x,
            inner_train_state=self.policy,
            params=theta,
            tx=optax.adam(args.meta_lr),  # outer optimizer
        )

        self.fit_baseline = LinearFeatureBaseline.fit_baseline

        # Outer step hparams
        self._eta = eta
        self._clip_eps = clip_eps
        self._num_grad_steps = num_grad_steps

    def reset_inner(self):
        self.policy = self.policy.replace(
            params=MetaVectorPolicy.expand_params(self.train_state.params, self.num_tasks)
        )

    def adapt(self, trajectories: Trajectory):
        self.policy, _ = inner_step(self.policy, trajectories)

    def step(self, all_trajectories: Trajectory):
        return outer_step(
            self.meta_train_state, all_trajectories, self._eta, self._clip_eps, self._num_grad_steps, self.num_tasks
        )

    def get_actions_train(self, obs: ArrayLike, key: jax.random.PRNGKey):
        return get_actions_log_probs_and_dists(self.policy, obs, key)

    def get_actions_eval(self, obs: ArrayLike, key: jax.random.PRNGKey):
        actions, _, _, _, key = get_actions_log_probs_and_dists(self.policy, obs, key)
        return actions, key

    def make_checkpoint(self):
        return {
            "train_state": self.train_state,
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
    if args.env_id == "ML10":
        benchmark = metaworld.ML10(seed=args.seed)
    elif args.env_id == "ML45":
        benchmark = metaworld.ML45(seed=args.seed)
    else:
        benchmark = metaworld.ML1(args.env_id, seed=args.seed)
    envs = make_envs(
        benchmark, meta_batch_size=args.meta_batch_size, seed=args.seed, max_episode_steps=args.max_episode_steps
    )
    NUM_TASKS = len(benchmark.train_classes)

    # agent setup
    envs.single_observation_space.dtype = np.float32
    buffer = MetaLearningReplayBuffer(
        num_tasks=args.meta_batch_size,
        trajectories_per_task=args.rollouts_per_task,
        max_episode_steps=args.max_episode_steps,
    )

    obs, _ = zip(*envs.call("sample_tasks"))
    obs = np.stack(obs)

    key, agent_init_key = jax.random.split(key)
    agent = ProMP(
        envs=envs,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        init_key=agent_init_key,
        init_obs=jax.device_put(obs),
        eta=args.inner_kl_penalty,
        clip_eps=args.clip_eps,
        num_grad_steps=args.num_promp_steps,
    )

    network_args = {
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "num_actions": np.prod(envs.single_action_space.shape),
    }

    buffer_processing_kwargs = {
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "fit_baseline": agent.fit_baseline,
        "normalize_advantages": True,
    }

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps):  # Outer step
        agent.reset_inner()
        all_trajectories: List[Trajectory] = []

        # Sampling step
        # Collect num_inner_gradient_steps D datasets + collect 1 D' dataset
        for inner_step in range(args.num_inner_gradient_steps + 1):
            while not buffer.ready:
                action, log_probs, means, stds, key = agent.get_actions_train(obs, key)
                next_obs, reward, _, truncated, _ = envs.step(action)
                buffer.push(obs, action, reward, truncated, log_probs, means, stds)
                obs = next_obs

            trajectories = buffer.get(**buffer_processing_kwargs)
            all_trajectories.append(trajectories)
            buffer.reset()

        writer.add_scalar("charts/mean_episodic_return", all_trajectories[-1].returns.mean(), global_step)

        # Inner policy update for the sake of sampling close to adapted policy during the
        # computation of the objective.
        if inner_step < args.num_inner_gradient_steps:
            agent.adapt(trajectories)

    # Outer policy update
    logs = agent.step(all_trajectories)
    print(f"Step {global_step}: ", logs)

    # Logging
    for k, v in logs.items():
        writer.add_scalar(k, v, global_step)

    # Evaluation
    _make_eval_envs_common = partial(make_eval_envs, benchmark, args.meta_batch_size, args.seed, args.max_episode_steps)
    eval_success_rate, eval_mean_return, key = metalearning_evaluation(
        agent,
        train_envs=_make_eval_envs_common(terminate_on_success=False),
        eval_envs=_make_eval_envs_common(terminate_on_success=True),
        adaptation_steps=args.num_inner_gradient_steps,
        adaptation_episodes=args.rollouts_per_task,
        eval_episodes=50,
        buffer_kwargs=buffer_processing_kwargs,
        key=key,
    )
    writer.add_scalar("charts/mean_success_rate", eval_success_rate, global_step)
    writer.add_scalar("charts/mean_evaluation_return", eval_mean_return, global_step)

    seconds_per_step = (time.time() - start_time) / (global_step + 1)
    writer.add_scalar("charts/time_per_step", seconds_per_step, global_step)
    print("Time per step: ", seconds_per_step)

    # Set tasks for next iteration
    obs, _ = zip(*envs.call("sample_tasks"))
    obs = np.stack(obs)

    # Checkpoint
    if args.save_model:
        ckpt_manager.save(global_step, agent.make_checkpoint() + {"key": key})
        print("model saved")

    envs.close()
    writer.close()
