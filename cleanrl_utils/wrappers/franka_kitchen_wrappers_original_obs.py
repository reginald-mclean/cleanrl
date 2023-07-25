import gymnasium as gym
import numpy as np
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

from numpy.typing import NDArray

from gymnasium import Env
from gymnasium.spaces import Space, Box
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers import normalize

class OneHotV0(gym.Wrapper):
    def __init__(self, env: gym.Env, task_idx: int, num_envs: int):
        gym.Wrapper.__init__(self, env)
        #self.env = env
        assert task_idx < num_envs, "The task idx of an env cannot be greater than or equal to the number of envs"
        self.one_hot = np.zeros(num_envs)
        self.one_hot[task_idx] = 1
        print(env, self.one_hot)

    def step(self, action, task):
        next_state, reward, terminate, truncate, info = self.env.step(action, task)
        next_state = np.concatenate([next_state['observation'], self.one_hot])
        return next_state, reward, terminate, truncate, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        new_obs = np.concatenate([obs['observation'], self.one_hot])
        obs['observation'] = new_obs
        return new_obs, info

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
        for i, env_fn in enumerate(env_fns):
            env = env_fn('FrankaKitchen-v1', tasks_to_complete=[tasks[i]], obs_space='original')
            self.env_names.append(tasks[i])
            if use_one_hot_wrapper:
                env = OneHotV0(env, self.env_names.index(tasks[i]), len(tasks))
            env = TimeLimit(env, max_episode_steps=500)
            env = RecordEpisodeStatistics(env)
            self.envs.append(env)
        self.copy = copy
        self.metadata = self.envs[0].metadata

        #temp_e = env_fns[0]('FrankaKitchen-v1', tasks_to_complete=[tasks[0]], obs_space='original')
        #print(temp_e.reset())
        #print(observation_space)
        #print(self.envs[0].observation_space)
        
        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            if use_one_hot_wrapper:
                low = np.zeros(len(self.envs))
                high = np.ones(len(self.envs))
                observation_space = Box(low=np.hstack([observation_space['observation'].low, low]), high=np.hstack([observation_space['observation'].high, high]), dtype=np.float64)
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        print(self.single_observation_space)
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
            observation, info = env.reset(**kwargs)
            #print(observation.shape)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        #print(self.observations.shape, len(observations), len(observations[0]))

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        #print(self.observations.shape, len(observations), len(observations[0]))
        #print("done")
        #exit(0)
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
            ) = env.step(action, self.env_names[i])

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            #print(type(observation))
            #print(f"{env} {observation.shape}")
            infos = self._add_info(infos, info, i)
        #print(self.observations.shape, len(observations), len(observations[0]))
        
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        #print(self.observations.shape, len(observations), len(observations[0]))
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
