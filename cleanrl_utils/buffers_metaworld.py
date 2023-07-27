from typing import Callable, List, NamedTuple, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import scipy
import torch

TorchOrNumpyArray: Type = Union[torch.Tensor, npt.NDArray]


class ReplayBufferSamples(NamedTuple):
    observations: TorchOrNumpyArray
    actions: TorchOrNumpyArray
    next_observations: TorchOrNumpyArray
    dones: TorchOrNumpyArray
    rewards: TorchOrNumpyArray


class Trajectory(NamedTuple):
    observations: TorchOrNumpyArray
    actions: TorchOrNumpyArray
    rewards: TorchOrNumpyArray
    dones: TorchOrNumpyArray
    returns: Optional[TorchOrNumpyArray] = None
    advantages: Optional[TorchOrNumpyArray] = None


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

        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)

        if self.use_torch:
            batch = map(lambda x: torch.tensor(x).to(self.device), batch)

        return ReplayBufferSamples(*batch)


class MetaLearningReplayBuffer:
    """A buffer to accumulate trajectories from batches envs for batches of tasks.
    Useful for ML1, ML10, ML45."""

    trajectories: List[List[Trajectory]]

    def __init__(self, num_tasks: int, num_envs_per_task: int, trajectories_per_task: int, max_trajectory_length: int):
        self.num_tasks = num_tasks
        self._target_samples = num_tasks * trajectories_per_task * max_trajectory_length

        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.trajectories = [[] for _ in range(self.num_tasks)]
        self._running_trajectories = [[] for _ in range(self.num_tasks)]
        self._total_samples = 0

    @property
    def ready(self) -> bool:
        """Returns whether or not a full batch of trajectories for each task has been sampled.

        Note that this is approximate and won't always have <trajectories_per_task> full trajectories ready.
        Mirrors how the official ProMP code samples."""
        return self._total_samples >= self._target_samples

    def _get_returns(self, rewards: npt.NDArray, discount: float):
        """Discounted cumulative sum.

        See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering"""
        # From garage
        return scipy.signal.lfilter([1], [1, float(-discount)], rewards[::-1], axis=-1)[::-1]

    def _compute_advantage(self, rewards: npt.NDArray, baselines: npt.NDArray, gamma: float, gae_lambda: float):
        # From ProMP's advantage computation
        baselines = np.append(baselines, 0)
        deltas = rewards + gamma * baselines[1:] - baselines[:-1]
        return self._get_returns(deltas, gamma * gae_lambda)

    def _pad_and_stack(self, trajectories: List[Trajectory]) -> Tuple[Trajectory, npt.NDArray]:
        valids = np.array([len(t.rewards) for t in trajectories])
        max_len = max(valids)
        stacked_trajectory = Trajectory(
            *map(lambda *xs: np.stack([np.pad(x, (0, max_len - len(x))) for x in xs]), *trajectories)
        )
        return stacked_trajectory, valids

    def get(self, gamma: float, gae_lambda: float, baseline: Callable) -> List[Trajectory]:
        """Compute returns and advantages for the collected trajectories. Then flatten the trajectories for each task.

        Returns a list of flattened batches of trajectories, one for each task."""
        ret = []

        for i in range(self.num_tasks):
            task_trajectories = self.trajectories[i]

            # 1) Get returns
            returns = [self._get_returns(t.rewards, gamma) for t in task_trajectories]

            # 2) Apply baseline
            # TODO might want some way to fit the baseline on the returns before this is run
            # If the baseline is a neural network though you should be able to do backprop after the
            # buffer return
            # However if the baseline is a linear model like the one used in the ProMP code, then it might make sense
            # to allow for fitting here, rather than outside.
            baselines = [baseline(t.observations) for t in task_trajectories]

            # 3) Compute advantages
            advantages = [
                self._compute_advantage(t.rewards, b, gamma, gae_lambda) for t, b in zip(task_trajectories, baselines)
            ]

            # Pack everything
            task_trajectories = [
                Trajectory(t.observations, t.actions, t.rewards, t.dones, r, a)
                for t, r, a in zip(task_trajectories, returns, advantages)
            ]
            task_trajectories = Trajectory(*map(lambda *xs: np.concatenate(xs), *task_trajectories))  # Flatten

            ret.append(task_trajectories)

        return ret

    def push(self, obs: npt.NDArray, action: npt.NDArray, reward: npt.NDArray, done: npt.NDArray):
        """Add a batch of timesteps to the buffer. Multiple batch dims are supported, but they
        need to multiply to the buffer's meta batch size.

        If an episode finishes here for any of the envs, pop the full trajectory into the trajectories buffer."""
        assert np.prod(reward.shape) == self.num_tasks

        obs = obs.reshape(-1, *obs.shape[2:]).copy()
        action = action.reshape(-1, *action.shape[2:]).copy()
        reward = reward.reshape(-1, 1).copy()
        done = done.reshape(-1, 1).copy()

        for i in range(self._combined_batch_size):
            self._running_trajectories[i].append((obs[i], action[i], reward[i], done[i]))

            if done[i]:  # pop full trajectories into the trajectories buffer
                trajectory = Trajectory(*map(lambda *xs: np.stack(xs), *self._running_trajectories[i]))
                self.trajectories[i // self.num_envs_per_task].append(trajectory)
                self._total_samples += len(trajectory.rewards.shape[0])
                self._running_trajectories[i] = []
