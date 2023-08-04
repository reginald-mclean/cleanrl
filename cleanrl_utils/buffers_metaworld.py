from typing import Callable, List, NamedTuple, Optional, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore
import torch

TorchOrNumpyArray = Union[torch.Tensor, npt.NDArray]


class ReplayBufferSamples(NamedTuple):
    observations: TorchOrNumpyArray
    actions: TorchOrNumpyArray
    next_observations: TorchOrNumpyArray
    dones: TorchOrNumpyArray
    rewards: TorchOrNumpyArray


class Trajectory(NamedTuple):
    # Standard timestep data
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    dones: npt.NDArray

    # Auxiliary policy outputs
    log_probs: Optional[npt.NDArray] = None
    means: Optional[npt.NDArray] = None
    stds: Optional[npt.NDArray] = None

    # Computed statistics about observed rewards
    returns: Optional[npt.NDArray] = None
    advantages: Optional[npt.NDArray] = None


class MultiTaskReplayBuffer:
    """Replay buffer for the multi-task benchmarks (MT1, MT10, MT50).

    Each sampling step, it samples a batch for each tasks, returning a batch of shape (batch_size, num_tasks).
    """

    obs: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    next_obs: npt.NDArray
    dones: npt.NDArray
    pos: int

    def __init__(
        self,
        capacity: int,
        num_tasks: int,
        envs: gym.vector.VectorEnv,
        use_torch: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.num_tasks = num_tasks
        self.use_torch = use_torch
        self.device = device
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(envs.single_observation_space.shape).prod()
        self._action_shape = np.array(envs.single_action_space.shape).prod()

        self.reset()  # Init buffer

    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros((self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.num_tasks, self._action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.pos = 0

    def add(self, obs: npt.NDArray, next_obs: npt.NDArray, action: npt.NDArray, reward: npt.NDArray, done: npt.NDArray):
        """Add a batch of samples to the buffer.

        It is assumed that the observation has a one-hot task embedding as its suffix."""
        task_idx = obs[:, -self.num_tasks :].argmax(1)

        self.obs[self.pos, task_idx] = obs.copy()
        self.actions[self.pos, task_idx] = action.copy()
        self.rewards[self.pos, task_idx] = reward.copy().reshape(-1, 1)
        self.next_obs[self.pos, task_idx] = next_obs.copy()
        self.dones[self.pos, task_idx] = done.copy().reshape(-1, 1)

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, single_task_batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of size `single_task_batch_size` for each task.

        Args:
            single_task_batch_size (int): The batch size for each task.

        Returns:
            ReplayBufferSamples: A batch of samples of batch shape (num_tasks * single_task_batch_size,).
        """
        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos, single_task_batch_size),
            size=(single_task_batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)  # type: ignore

        if self.use_torch:
            batch = map(lambda x: torch.tensor(x).to(self.device), batch)  # type: ignore

        return ReplayBufferSamples(*batch)


class MetaLearningReplayBuffer:
    """A buffer to accumulate trajectories from batches envs for batches of tasks.
    Useful for ML1, ML10, ML45.

    In Metaworld, all episodes are as long as the time limit (typically 500), thus in this buffer we assume
    fixed-length episodes and leverage that for optimisations."""

    trajectories: List[List[Trajectory]]

    def __init__(
        self,
        num_tasks: int,
        trajectories_per_task: int,
        max_episode_steps: int,
        use_torch: bool = False,
        device: Optional[str] = None,
    ):
        self.num_tasks = num_tasks
        self._trajectories_per_task = trajectories_per_task
        self._max_episode_steps = max_episode_steps

        self._use_torch = use_torch
        self._device = device

        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.trajectories = [[] for _ in range(self.num_tasks)]
        self._running_trajectories = [[] for _ in range(self.num_tasks)]

    @property
    def ready(self) -> bool:
        """Returns whether or not a full batch of trajectories for each task has been sampled.

        Note that this is approximate and won't always have <trajectories_per_task> full trajectories ready.
        Mirrors how the official ProMP code samples."""
        return all(len(t) == self._trajectories_per_task for t in self.trajectories)

    def _get_returns(self, rewards: npt.NDArray, discount: float):
        """Discounted cumulative sum.

        See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering"""
        # From garage, modified to work on multi-dimensional arrays
        return scipy.signal.lfilter([1], [1, float(-discount)], rewards[..., ::-1], axis=-1)[..., ::-1]

    def _compute_advantage(self, rewards: npt.NDArray, baselines: npt.NDArray, gamma: float, gae_lambda: float):
        # From ProMP's advantage computation, modified to work on multi-dimensional arrays
        baselines = np.append(baselines, np.zeros((*baselines.shape[:-1], 1)), axis=-1)
        deltas = rewards + gamma * baselines[..., 1:] - baselines[..., :-1]
        return self._get_returns(deltas, gamma * gae_lambda)

    def _to_torch(self, trajectories: Trajectory) -> Trajectory:
        return Trajectory(*map(lambda x: torch.tensor(x).to(self._device), trajectories))  # type: ignore

    def get(
        self,
        as_is: bool = False,
        gamma: Optional[float] = None,
        gae_lambda: Optional[float] = None,
        baseline: Optional[Callable] = None,
        fit_baseline: Optional[Callable] = None,
        normalize_advantages: bool = False,
    ) -> Trajectory:
        """Compute returns and advantages for the collected trajectories.

        Returns a Trajectory tuple where each array has the batch dimensions (Task,Timestep,).
        The timesteps are multiple trajectories flattened into one time dimension."""
        trajectories_per_task = [Trajectory(*map(lambda *xs: np.stack(xs), *t)) for t in self.trajectories]
        all_trajectories = Trajectory(*map(lambda *xs: np.stack(xs), *trajectories_per_task))
        assert all_trajectories.observations.shape[:3] == (
            self.num_tasks,
            self._trajectories_per_task,
            self._max_episode_steps,
        ), "Buffer does not have the expected amount of data before sampling."

        if as_is:
            return self._to_torch(all_trajectories) if self._use_torch else all_trajectories

        assert (
            gamma is not None and gae_lambda is not None
        ), "Gamma and gae_lambda must be provided if GAE computation is not disabled through the `as_is` flag."

        # 1) Get returns
        all_trajectories = all_trajectories._replace(
            returns=self._get_returns(all_trajectories.rewards, gamma)  # type: ignore
        )

        # 2.1) (Optional) Fit baseline
        if fit_baseline is not None:
            baseline = fit_baseline(all_trajectories)

        # 2.2) Apply baseline
        # NOTE baseline is responsible for any data conversions / moving to the GPU
        assert baseline is not None, "You must provide a baseline function, or a fit_baseline that returns one."
        baselines = baseline(all_trajectories.observations)

        # 3) Compute advantages
        advantages = self._compute_advantage(all_trajectories.rewards, baselines, gamma, gae_lambda)  # type: ignore
        all_trajectories = all_trajectories._replace(advantages=advantages)

        # 4) Flatten trajectory and time dimensions
        all_trajectories = Trajectory(*map(lambda x: x.reshape(self.num_tasks, -1, *x.shape[3:]), all_trajectories))

        # 4.1) (Optional) Normalize advantages
        if normalize_advantages:
            advantages = all_trajectories.advantages
            norm_advantages = (advantages - np.mean(advantages, axis=1, keepdims=True)) / (
                np.std(advantages, axis=1, keepdims=True) + 1e-8
            )
            all_trajectories = all_trajectories._replace(advantages=norm_advantages)

        return self._to_torch(all_trajectories) if self._use_torch else all_trajectories

    def push(
        self,
        obs: npt.NDArray,
        action: npt.NDArray,
        reward: npt.NDArray,
        done: npt.NDArray,
        log_prob: Optional[npt.NDArray] = None,
        mean: Optional[npt.NDArray] = None,
        std: Optional[npt.NDArray] = None,
    ):
        """Add a batch of timesteps to the buffer. Multiple batch dims are supported, but they
        need to multiply to the buffer's meta batch size.

        If an episode finishes here for any of the envs, pop the full trajectory into the trajectories buffer."""
        assert np.prod(reward.shape) == self.num_tasks

        obs = obs.copy()
        action = action.copy()
        assert obs.ndim == action.ndim
        if obs.ndim > 2 and action.ndim > 2:  # Flatten outer batch dims only if they exist
            obs = obs.reshape(-1, *obs.shape[2:])
            action = action.reshape(-1, *action.shape[2:])

        reward = reward.reshape(-1, 1).copy()
        done = done.reshape(-1, 1).copy()
        if log_prob is not None:
            log_prob = log_prob.reshape(-1, 1).copy()
        if mean is not None:
            mean = mean.copy()
            if mean.ndim > 2:
                mean = mean.reshape(-1, *mean.shape[2:])
        if std is not None:
            std = std.copy()
            if std.ndim > 2:
                std = std.reshape(-1, *std.shape[2:])

        for i in range(self.num_tasks):
            trajectory_step = (obs[i], action[i], reward[i], done[i])
            if log_prob is not None:
                trajectory_step += (log_prob[i],)
            if mean is not None:
                trajectory_step += (mean[i],)
            if std is not None:
                trajectory_step += (std[i],)
            self._running_trajectories[i].append(trajectory_step)

            if done[i]:  # pop full trajectories into the trajectories buffer
                trajectory = Trajectory(*map(lambda *xs: np.stack(xs), *self._running_trajectories[i]))
                self.trajectories[i].append(trajectory)
                self._running_trajectories[i] = []
