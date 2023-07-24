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

        self.clear()  # Init buffer

    def clear(self):
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

    def _get_buffer_tuple(self):
        """Utility function to lower amount of code necessary to toggle torch usage."""
        return self.obs, self.actions, self.next_obs, self.dones, self.rewards

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

        batch = tuple(tuple_item[sample_idx] for tuple_item in self._get_buffer_tuple())
        if self.use_torch:
            batch = tuple(torch.tensor(item).to(self.device) for item in batch)

        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = tuple(tuple_item.reshape(mt_batch_size, -1) for tuple_item in batch)

        return ReplayBufferSamples(*batch)
