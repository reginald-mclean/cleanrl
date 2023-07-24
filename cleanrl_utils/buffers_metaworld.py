from typing import NamedTuple, Optional, Type, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch

TorchOrNumpyArray: Type = Union[torch.Tensor, npt.NDArray]


class ReplayBufferSamples(NamedTuple):
    observations: TorchOrNumpyArray
    actions: TorchOrNumpyArray
    next_observations: TorchOrNumpyArray
    dones: TorchOrNumpyArray
    rewards: TorchOrNumpyArray


class MultiTaskReplayBuffer:
    """Replay buffer for the multi-task benchmarks (MT1, MT10, MT50).

    Each sampling step, it samples a batch for each tasks, returning a batch of shape (batch_size, num_tasks).
    """

    def __init__(
        self,
        capacity: int,
        num_tasks: int,
        envs: gym.Env,
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
        if self.use_torch:
            batch = map(lambda x: torch.tensor(x).to(self.device), batch)

        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = map(lambda x: x.reshape(mt_batch_size, -1), batch)

        return ReplayBufferSamples(*batch)


class MetaLearningReplayBuffer:
    """A buffer to accumulate trajectories from batches envs for batches of tasks."""

    def __init__(self, num_tasks: int, num_envs_per_task: int, trajectories_per_task: int, max_trajectory_length: int):
        self.num_tasks = num_tasks
        self.num_envs_per_task = num_envs_per_task
        self._combined_batch_size = num_tasks * num_envs_per_task
        self._target_samples = num_tasks * trajectories_per_task * max_trajectory_length

        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.trajectories = [[] for _ in range(self.num_tasks)]
        self._running_trajectories = [[] for _ in range(self._combined_batch_size)]
        self._total_samples = 0

    @property
    def ready(self) -> bool:
        """Returns whether or not a full batch of trajectories for each task has been sampled.

        Note that this is approximate and won't always have <trajectories_per_task> full trajectories ready.
        Mirrors how the official ProMP code samples."""
        return self._total_samples >= self._target_samples

    def sample(self) -> ReplayBufferSamples:
        """Return the collected trajectories, padded and packed and passed through some extra post processing.

        Mirrors how the official ProMP code prepares data."""
        # TODO implement all the relevant processing like fitting & applying a baseline, advantage norm etc
        raise NotImplementedError

    def push(
        self, obs: npt.NDArray, next_obs: npt.NDArray, action: npt.NDArray, reward: npt.NDArray, done: npt.NDArray
    ):
        """Add a 2D batch (meta_batch, batch_per_task, ...) of timesteps to the buffer.

        If an episode finishes here for any of the envs, pop the full trajectory into the trajectories buffer."""

        obs = obs.reshape(self._combined_batch_size, -1).copy()
        next_obs = next_obs.reshape(self._combined_batch_size, -1).copy()
        action = action.reshape(self._combined_batch_size, -1).copy()
        reward = reward.reshape(self._combined_batch_size, 1).copy()
        done = done.reshape(self._combined_batch_size, 1).copy()

        for i in range(self._combined_batch_size):
            self._running_trajectories[i].append((obs[i], action[i], next_obs[i], done[i], reward[i]))

            if done[i]:  # pop full trajectories into the trajectories buffer
                trajectory = ReplayBufferSamples(*map(lambda *xs: np.stack(xs), *self._running_trajectories[i]))
                self.trajectories[i // self.num_envs_per_task].append(trajectory)
                self._total_samples += len(trajectory.rewards.shape[0])
                self._running_trajectories[i] = []
