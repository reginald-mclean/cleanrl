# ruff: noqa: E402
import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Callable, List, Optional, Tuple, Type

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
from cleanrl_utils.buffers_metaworld import MultiTaskRolloutBuffer, Rollout
from cleanrl_utils.evals.metaworld_jax_eval import metalearning_evaluation
from cleanrl_utils.wrappers import metaworld_wrappers
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
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
    parser.add_argument("--evaluation-frequency", type=int, default=1_000_000,
        help="the frequency of evaluating the model (in total timesteps collected from the env)")
    parser.add_argument("--num-evaluation-goals", type=int, default=10,
        help="the number of goal positions to evaluate on per test task")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ML10", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=15_000_000,
        help="total number timesteps to collect from the environment")
    parser.add_argument("--max-episode-steps", type=int, default=500,
        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=1.0,
        help="the lambda for the general advantage estimation")
    # MAML 
    parser.add_argument("--meta-batch-size", type=int, default=20,
        help="the number of tasks to sample and train on in parallel")
    parser.add_argument("--rollouts-per-task", type=int, default=10,
        help="the number of trajectories to collect per task in the meta batch")
    parser.add_argument("--num-layers", type=int, default=2, help="the number of hidden layers in the MLP")
    parser.add_argument("--hidden-dim", type=int, default=512, help="the dimension of each hidden layer in the MLP")
    parser.add_argument("--inner-lr", type=float, default=0.1, help="the inner (adaptation) step size")
    parser.add_argument("--num-inner-gradient-steps", type=int, default=1,
        help="number of inner / adaptation gradient steps")
    # TRPO
    parser.add_argument("--delta", type=float, default=0.01,
        help="the value the KL divergence is constrained to in TRPO")
    parser.add_argument("--cg-iters", type=int, default=10,
        help="the maximum number of iterations of the conjugate gradient algorithm")
    parser.add_argument("--backtrack-ratio", type=float, default=0.8,
        help="the backtrack ratio in the line search for TRPO (exponential decay rate)")
    parser.add_argument("--max-backtrack-iters", type=int, default=15,
        help="the maximum number of iterations in the line search for TRPO")
    args = parser.parse_args()
    # fmt: on
    return args


def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    meta_batch_size: int,
    max_episode_steps: Optional[int] = None,
    split: str = "train",
    terminate_on_success: bool = False,
    task_select: str = "random",
    total_tasks_per_class: Optional[int] = None,
) -> Tuple[gym.vector.VectorEnv, List[str]]:
    all_classes = benchmark.train_classes if split == "train" else benchmark.test_classes
    all_tasks = benchmark.train_tasks if split == "train" else benchmark.test_tasks
    assert meta_batch_size % len(all_classes) == 0, "meta_batch_size must be divisible by envs_per_task"
    tasks_per_env = meta_batch_size // len(all_classes)

    def make_env(env_cls: Type[SawyerXYZEnv], tasks: list) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env.toggle_terminate_on_success(terminate_on_success)
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
            assert len(tasks_for_subenv) == len(tasks) // tasks_per_env
            env_tuples.append((env_cls, tasks_for_subenv))
            task_names.append(env_name)

    return (
        gym.vector.AsyncVectorEnv([partial(make_env, env_cls=env_cls, tasks=tasks) for env_cls, tasks in env_tuples]),
        task_names,
    )


make_envs = partial(_make_envs_common, terminate_on_success=False, task_select="pseudorandom", split="train")
make_eval_envs = partial(_make_envs_common, terminate_on_success=True, task_select="pseudorandom", split="test")


# Networks
class MLPTorso(nn.Module):
    """A Flax Module to represent an MLP with tanh activations."""

    num_hidden_layers: int
    output_dim: int
    hidden_dim: int = 64
    activate_last: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        CustomDense = partial(nn.Dense, kernel_init=self.kernel_init)

        for i in range(self.num_hidden_layers):
            x = CustomDense(self.hidden_dim, name=f"layer_{i}")(x)
            x = nn.tanh(x)
        x = CustomDense(self.output_dim, name=f"layer_{self.num_hidden_layers}")(x)
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
        mean = MLPTorso(
            num_hidden_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_actions,
        )(x)
        # fmt: off
        #                               zeros_init = log(ones_init)
        log_std = self.param("log_std", nn.initializers.zeros_init(), (self.num_actions,))
        log_std = jnp.maximum(log_std, self.LOG_STD_MIN)
        # fmt: on
        log_std = jnp.broadcast_to(log_std, mean.shape)
        return mean, log_std


class MetaVectorPolicy(nn.Module):
    n_tasks: int
    num_actions: int
    num_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state: jax.Array) -> distrax.Distribution:
        vmap_policy = nn.vmap(
            GaussianPolicy,
            variable_axes={"params": 0},  # parameters not shared between task policies
            in_axes=0,
            out_axes=0,
            axis_size=self.n_tasks,
        )
        mean, log_std = vmap_policy(
            num_actions=self.num_actions, num_layers=self.num_layers, hidden_dim=self.hidden_dim
        )(state)
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))

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


@jax.jit
def get_deterministic_actions(actor: TrainState, obs: ArrayLike) -> jax.Array:
    return actor.apply_fn(actor.params, obs).loc


def get_actions_log_probs_and_dists(
    actor: TrainState, obs: ArrayLike, key: jax.random.PRNGKey
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, jax.random.PRNGKeyArray]:
    actions, log_probs, means, stds, key = sample_actions(actor, obs, key)
    return jax.device_get(actions), jax.device_get(log_probs), jax.device_get(means), jax.device_get(stds), key


class MetaTrainState(TrainState):
    inner_train_state: TrainState


# MAMLTRPO
@jax.jit
def inner_step(policy: TrainState, rollouts: Rollout) -> TrainState:
    def inner_opt_objective(_theta: FrozenDict):
        log_probs = jnp.expand_dims(policy.apply_fn(_theta, rollouts.observations).log_prob(rollouts.actions), -1)
        return -(log_probs * rollouts.advantages).mean()

    grads = jax.grad(inner_opt_objective)(policy.params)
    updated_policy = policy.apply_gradients(grads=grads)  # Inner gradient step

    return updated_policy


@partial(jax.jit, static_argnames=("num_tasks", "delta", "cg_iters", "backtrack_ratio", "max_backtrack_iters"))
def outer_step(
    train_state: MetaTrainState,
    all_rollouts: List[Rollout],
    num_tasks: int,
    delta: float,
    cg_iters: int,
    backtrack_ratio: float,
    max_backtrack_iters: int,
) -> Tuple[MetaTrainState, dict]:
    def maml_loss(theta: FrozenDict):
        vec_theta = MetaVectorPolicy.expand_params(theta, num_tasks)
        inner_train_state = train_state.inner_train_state.replace(params=vec_theta)

        # Adaptation steps
        for i in range(len(all_rollouts) - 1):
            rollouts = all_rollouts[i]
            inner_train_state = inner_step(inner_train_state, rollouts)

        # Inner Train State now has theta^\prime
        # Compute MAML objective
        rollouts = all_rollouts[-1]
        new_param_dist = inner_train_state.apply_fn(inner_train_state.params, rollouts.observations)
        new_param_log_probs = jnp.expand_dims(new_param_dist.log_prob(rollouts.actions), -1)

        likelihood_ratio = jnp.exp(new_param_log_probs - rollouts.log_probs)
        outer_objective = likelihood_ratio * rollouts.advantages
        return -outer_objective.mean()

    # TRPO, outer gradient step
    def kl_constraint(params: FrozenDict, inputs: Rollout, targets: distrax.Distribution):
        vec_theta = MetaVectorPolicy.expand_params(params, num_tasks)
        inner_train_state = train_state.inner_train_state.replace(params=vec_theta)

        # Adaptation steps
        for i in range(len(inputs) - 1):
            rollouts = inputs[i]
            inner_train_state = inner_step(inner_train_state, rollouts)

        new_param_dist = inner_train_state.apply_fn(inner_train_state.params, inputs[-1].observations)
        return targets.kl_divergence(new_param_dist).mean()

    target_dist = distrax.MultivariateNormalDiag(all_rollouts[-1].means, all_rollouts[-1].stds)
    kl_before = kl_constraint(train_state.params, all_rollouts, target_dist)

    ## Compute search direction by solving for Ax = g

    def hvp(x):
        hvp_deep = optax.hvp(kl_constraint, v=x, params=train_state.params, inputs=all_rollouts, targets=target_dist)
        hvp_shallow = ravel_pytree(hvp_deep)[0]
        return hvp_shallow + 1e-5 * x  # Ensure positive definite

    loss_before, opt_objective_grads = jax.value_and_grad(maml_loss)(train_state.params)
    g, unravel_params = ravel_pytree(opt_objective_grads)
    s, _ = jax.scipy.sparse.linalg.cg(hvp, g, maxiter=cg_iters)

    ## Compute optimal step beta
    beta = jnp.sqrt(2.0 * delta * (1 / (jnp.dot(s, hvp(s)) + 1e-8)))

    ## Line search
    s = unravel_params(s)

    def _cond_fn(val):
        step, loss, kl, _ = val
        return ((kl > delta) | (loss >= loss_before)) & (step < max_backtrack_iters)

    def _body_fn(val):
        step, loss, kl, _ = val
        new_params = jax.tree_util.tree_map(
            lambda theta_i, s_i: theta_i - (backtrack_ratio**step) * beta * s_i, train_state.params, s
        )
        loss, kl = maml_loss(new_params), kl_constraint(new_params, all_rollouts, target_dist)
        return step + 1, loss, kl, new_params

    step, loss, kl, new_params = jax.lax.while_loop(
        _cond_fn, _body_fn, init_val=(0, loss_before, jnp.finfo(jnp.float32).max, train_state.params)
    )

    # Param updates
    # Reject params if line search failed
    params = jax.lax.cond((loss < loss_before) & (kl <= delta), lambda: new_params, lambda: train_state.params)
    train_state = train_state.replace(params=params)

    return train_state, {
        "losses/loss_before": jnp.mean(loss_before),
        "losses/loss_after": jnp.mean(loss),
        "losses/kl_before": kl_before,
        "losses/kl_after": kl,
        "losses/backtrack_steps": step,
    }


# Baseline
# NOTE The original MAML paper uses a Linear time-dependent return baseline model from
# Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control" as its
# baseline, rather than a neural network used as a learned value function.
#
# The garage MAML implementation for Metaworld did use a Gaussian MLP Baseline, however,
# it does not seem to perform as well as the original LinearFeatureBaseline on the current iteration of
# Metaworld at least and is unable to reproduce the original MAML results, while the LinearFeatureBaseline can.


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

            coeffs.append(np.expand_dims(task_coeffs, axis=0))

        return np.stack(coeffs)

    @classmethod
    def fit_baseline(cls, rollouts: Rollout) -> Callable[[Rollout], np.ndarray]:
        coeffs = cls._fit_baseline(rollouts.observations, rollouts.returns)

        def baseline(rollouts: Rollout) -> np.ndarray:
            features = cls._extract_features(rollouts.observations, reshape=False)
            return features @ coeffs

        return baseline


# Wrapper
class MAMLTRPO:
    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        num_layers: int,
        hidden_dim: int,
        init_key: jax.random.PRNGKey,
        init_obs: ArrayLike,
        inner_lr: float,
        delta: float,
        cg_iters: int,
        backtrack_ratio: float,
        max_backtrack_iters: int,
    ):
        self.num_tasks = envs.unwrapped.num_envs

        self.inner_lr = inner_lr

        self.network_args = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_actions": np.prod(envs.single_action_space.shape),
        }

        # Init general parameters theta
        self.policy = None
        theta = MetaVectorPolicy.init_single(**self.network_args, rng=init_key, init_args=[init_obs])
        self.init_multitask_policy(self.num_tasks, theta)

        self.train_state = MetaTrainState.create(
            apply_fn=lambda x: x,
            inner_train_state=self.policy,
            params=theta,
            tx=optax.identity(),  # TRPO optimiser's in outer_step
        )

        # Baseline
        self.fit_baseline = LinearFeatureBaseline.fit_baseline

        # TRPO
        self.delta = delta
        self.cg_iters = cg_iters
        self.backtrack_ratio = backtrack_ratio
        self.max_backtrack_iters = max_backtrack_iters

    def init_multitask_policy(self, num_tasks: int, params: FrozenDict) -> None:
        self.num_tasks = num_tasks

        # Init a vectorized policy, running a separate set of parameters for each task
        # The initial parameters for each task are all the same
        policy_network = MetaVectorPolicy(n_tasks=num_tasks, **self.network_args)
        if not self.policy:
            self.policy = TrainState.create(
                apply_fn=policy_network.apply,
                params=MetaVectorPolicy.expand_params(params, num_tasks),
                tx=optax.sgd(learning_rate=self.inner_lr),  # inner optimizer
            )
        else:
            self.policy = self.policy.replace(
                apply_fn=policy_network.apply, params=policy_network.expand_params(params, num_tasks)
            )

    def adapt(self, rollouts: Rollout) -> None:
        self.policy = inner_step(self.policy, rollouts)

    def step(self, all_rollouts: Rollout) -> dict:
        self.train_state, logs = outer_step(
            train_state=self.train_state,
            all_rollouts=all_rollouts,
            num_tasks=self.num_tasks,
            delta=self.delta,
            cg_iters=self.cg_iters,
            backtrack_ratio=self.backtrack_ratio,
            max_backtrack_iters=self.max_backtrack_iters,
        )
        return logs

    def get_action_train(self, obs: ArrayLike, key: jax.random.PRNGKey):
        return get_actions_log_probs_and_dists(self.policy, obs, key)

    def get_action_eval(self, obs: ArrayLike) -> np.ndarray:
        actions = get_deterministic_actions(self.policy, obs)
        return jax.device_get(actions)

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
    if args.env_id == "ML10":
        benchmark = metaworld.ML10(seed=args.seed)
    elif args.env_id == "ML45":
        benchmark = metaworld.ML45(seed=args.seed)
    else:
        benchmark = metaworld.ML1(args.env_id, seed=args.seed)
    envs, train_task_names = make_envs(
        benchmark, meta_batch_size=args.meta_batch_size, seed=args.seed, max_episode_steps=args.max_episode_steps
    )
    eval_envs, eval_task_names = make_eval_envs(
        benchmark=benchmark,
        meta_batch_size=args.meta_batch_size,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        total_tasks_per_class=args.num_evaluation_goals,
    )

    # agent setup
    envs.single_observation_space.dtype = np.float32
    buffer = MultiTaskRolloutBuffer(
        num_tasks=args.meta_batch_size,
        rollouts_per_task=args.rollouts_per_task,
        max_episode_steps=args.max_episode_steps,
    )

    obs, _ = zip(*envs.call("sample_tasks"))
    obs = np.stack(obs)

    key, agent_init_key = jax.random.split(key)
    agent = MAMLTRPO(
        envs=envs,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        init_key=agent_init_key,
        init_obs=jax.device_put(obs),
        inner_lr=args.inner_lr,
        delta=args.delta,
        cg_iters=args.cg_iters,
        backtrack_ratio=args.backtrack_ratio,
        max_backtrack_iters=args.max_backtrack_iters,
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
    steps_per_iter = args.meta_batch_size * args.rollouts_per_task * args.max_episode_steps
    n_iters = args.total_timesteps // steps_per_iter
    for _iter in range(n_iters):  # Outer step
        global_step = _iter * steps_per_iter
        print(f"Iteration {_iter}, Global num of steps {global_step}")
        agent.init_multitask_policy(envs.num_envs, agent.train_state.params)
        all_rollouts: List[Rollout] = []

        # Sampling step
        # Collect num_inner_gradient_steps D datasets + collect 1 D' dataset
        for _step in range(args.num_inner_gradient_steps + 1):
            print(f"- Collecting inner step {_step}")
            while not buffer.ready:
                action, log_probs, means, stds, key = agent.get_action_train(obs, key)
                next_obs, reward, _, truncated, _ = envs.step(action)
                buffer.push(obs, action, reward, truncated, log_probs, means, stds)
                obs = next_obs

            rollouts = buffer.get(**buffer_processing_kwargs)
            all_rollouts.append(rollouts)
            buffer.reset()

            # Inner policy update for the sake of sampling close to adapted policy during the
            # computation of the objective.
            if _step < args.num_inner_gradient_steps:
                print(f"- Adaptation step {_step}")
                agent.adapt(rollouts)

        mean_episodic_return = all_rollouts[-1].episode_returns.mean()
        writer.add_scalar("charts/mean_episodic_return", mean_episodic_return, global_step)
        print("- Mean episodic return: ", mean_episodic_return)

        # Outer policy update
        print("- Computing outer step")
        logs = agent.step(all_rollouts)
        logs = jax.tree_util.tree_map(lambda x: jax.device_get(x).item(), logs)

        # Evaluation
        if global_step % args.evaluation_frequency == 0 and global_step > 0:
            print("- Evaluating on test set...")
            num_evals = (len(benchmark.test_classes) * args.num_evaluation_goals) // args.meta_batch_size

            evaluation_kwargs = {
                "agent": agent,
                "adaptation_steps": args.num_inner_gradient_steps,
                "adaptation_episodes": args.rollouts_per_task,
                "max_episode_steps": args.max_episode_steps,
                "eval_episodes": 3,  # How many episodes to evaluate the adapted policy *per sampled task*
                "buffer_kwargs": buffer_processing_kwargs,
            }

            eval_success_rate, eval_mean_return, eval_success_rate_per_task, key = metalearning_evaluation(
                eval_envs=eval_envs,
                num_evals=num_evals,  # How many times to sample new tasks to do meta evaluation on
                key=key,
                task_names=eval_task_names,
                **evaluation_kwargs,
            )

            logs["charts/mean_success_rate"] = float(eval_success_rate)
            logs["charts/mean_evaluation_return"] = float(eval_mean_return)
            for task_name, success_rate in eval_success_rate_per_task.items():
                logs[f"charts/{task_name}_success_rate"] = float(success_rate)

            print("- Evaluating on train set...")
            num_evals = (len(benchmark.train_classes) * args.num_evaluation_goals) // args.meta_batch_size
            _, _, eval_success_rate_per_train_task, key = metalearning_evaluation(
                eval_envs=envs,
                num_evals=num_evals,  # How many times to sample new tasks to do meta evaluation on
                key=key,
                task_names=train_task_names,
                **evaluation_kwargs,
            )
            for task_name, success_rate in eval_success_rate_per_train_task.items():
                logs[f"charts/{task_name}_train_success_rate"] = float(success_rate)

            envs.call("toggle_terminate_on_success", False)

            if args.save_model:  # Checkpoint
                ckpt_manager.save(
                    step=global_step,
                    items=agent.make_checkpoint() | {"key": key, "global_step": global_step},
                    metrics=logs,
                )
                print("- Saved Model")

        # Logging
        logs = jax.device_get(logs)
        for k, v in logs.items():
            writer.add_scalar(k, v, global_step)
        print(logs)

        writer.add_scalar("charts/sps", global_step / (time.time() - start_time), global_step)
        print("- SPS: ", global_step / (time.time() - start_time))

        # Set tasks for next iteration
        obs, _ = zip(*envs.call("sample_tasks"))
        obs = np.stack(obs)

    envs.close()
    writer.close()
