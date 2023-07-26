# ruff: noqa: E402
import time
from typing import List, Tuple

import gymnasium as gym
import jax
import numpy as np


def evaluation_procedure(
    agent,
    eval_envs: gym.vector.AsyncVectorEnv,
    num_episodes: int,
    key: jax.random.PRNGKey,
) -> Tuple[float, List[List[float]], jax.random.PRNGKey]:
    obs, _ = eval_envs.reset()
    successes = np.zeros(eval_envs.num_envs)
    episodic_returns = [[] for _ in range(eval_envs.num_envs)]

    start_time = time.time()

    while not all(len(returns) >= num_episodes for returns in episodic_returns):
        actions, key = agent.get_action(obs, key)
        obs, _, _, _, infos = eval_envs.step(actions)
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                episodic_returns[i].append(info["episode"]["r"])
                # Discard any extra episodes from envs that ran ahead
                if len(episodic_returns[i]) <= num_episodes:
                    successes[i] += int(info["success"])

    episodic_returns = [returns[:num_episodes] for returns in episodic_returns]

    print(f"Evaluation time: {time.time() - start_time:.2f}s")

    return (successes / num_episodes).mean(), np.mean(episodic_returns), key
