from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from numpy.typing import NDArray


class OneHotWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: Env, task_idx: int, num_tasks: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        env_lb = env.observation_space.low
        env_ub = env.observation_space.high
        one_hot_ub = np.ones(num_tasks)
        one_hot_lb = np.zeros(num_tasks)

        self.one_hot = np.zeros(num_tasks)
        self.one_hot[task_idx] = 1.0

        self._observation_space = gym.spaces.Box(
            np.concatenate([env_lb, one_hot_lb]), np.concatenate([env_ub, one_hot_ub])
        )

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    def observation(self, obs: NDArray) -> NDArray:
        return np.concatenate([obs, self.one_hot])


class RandomTaskSelectWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically set / reset the environment to a random
    task."""

    tasks: List[object]
    current_task: object

    def _set_random_task(self):
        self.current_task = self.np_random.choice(len(self.tasks))
        self.unwrapped.set_task(self.tasks[self.current_task])

    def __init__(self, env: Env, tasks: List[object]):
        super().__init__(env)
        self.tasks = tasks

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self._set_random_task()
        return self.env.reset(seed=seed, options=options)


class PseudoRandomTaskSelectWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically reset the environment to a *pseudo*random task when explicitly called.

    Pseudorandom implies no collisions therefore the next task in the list will be used cyclically.

    Note that the task initialization is not every environment reset, but only when sample_tasks() is explicitly called.

    For use in MetaRL algorithms with the Metaworld ML benchmark classes.
    """

    tasks: List[object]
    current_task: object

    def _set_pseudo_random_task(self):
        self.current_task = (self.current_task + 1) % len(self.tasks)
        self.unwrapped.set_task(self.tasks[self.current_task])

    def __init__(self, env: Env, tasks: List[object]):
        super().__init__(env)
        self.tasks = tasks

    def sample_tasks(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self._set_pseudo_random_task()
        return self.env.reset(seed=seed, options=options)


# ---- Kept for compatibility ----
class OneHotV0(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, task_idx: int, num_envs: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        # self.env = env
        assert task_idx < num_envs, "The task idx of an env cannot be greater than or equal to the number of envs"
        self.one_hot = np.zeros(num_envs)
        self.one_hot[task_idx] = 1

    def step(self, action):
        next_state, reward, terminate, truncate, info = self.env.step(action)
        next_state = np.concatenate([next_state, self.one_hot])
        return next_state, reward, terminate, truncate, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = np.concatenate([obs, self.one_hot])
        return obs, info


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> env.reset(seed=42)
        (array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32), {})
    """

    def __init__(
        self,
        env_fns,
        tasks: List,
        use_one_hot_wrapper: bool = False,
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = []
        self.tasks = dict()
        self.env_names = []
        self.current_tasks = dict()
        for env_name in env_fns:
            env = env_fns[env_name]()
            self.env_names.append(env_name)
            self.tasks[env_name] = [task for task in tasks if task.env_name == env_name]
            self.current_tasks[env_name] = np.random.choice(len(self.tasks[env_name]))
            env.set_task(self.tasks[env_name][self.current_tasks[env_name]])
            if use_one_hot_wrapper:
                env = OneHotV0(env, self.env_names.index(env_name), len(self.env_fns.keys()))
            env = RecordEpisodeStatistics(env)
            self.envs.append(env)
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            if use_one_hot_wrapper:
                low = np.zeros(len(self.envs))
                high = np.ones(len(self.envs))
                observation_space = Box(
                    low=np.hstack([observation_space.low, low]),
                    high=np.hstack([observation_space.high, high]),
                    dtype=np.float64,
                )
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(self.single_observation_space, n=self.num_envs, fn=np.zeros)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminateds[:] = False
        self._truncateds[:] = False
        observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options
            # need to set task first
            env_name = self.env_names[i]
            _, _ = env.reset()
            self.current_tasks[env_name] = (self.current_tasks[env_name] + 1) % len(self.tasks[env_name])
            env.set_task(self.tasks[env_name][self.current_tasks[env_name]])
            observation, info = env.reset(**kwargs)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(self.single_observation_space, observations, self.observations)
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
        self._actions = iterate(self.action_space, actions)

    def step_wait(self) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                # select new task
                env_name = self.env_names[i]
                _, _ = env.reset()
                self.current_tasks[env_name] = np.random.choice(len(self.tasks[env_name]))
                env.set_task(self.tasks[env_name][self.current_tasks[env_name]])
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(self.single_observation_space, observations, self.observations)

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True
