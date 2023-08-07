# ruff: noqa: E402
import time
from typing import List, Tuple

import gymnasium as gym
import jax
import numpy as np

from cleanrl_utils.buffers_metaworld import MultiTaskRolloutBuffer


def evaluation_procedure(
    agent,
    eval_envs: gym.vector.VectorEnv,
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


def metalearning_evaluation(
    agent,
    train_envs: gym.vector.VectorEnv,
    eval_envs: gym.vector.VectorEnv,
    adaptation_steps: int,
    adaptation_episodes: int,
    eval_episodes: int,
    buffer_kwargs: dict,
    key: jax.random.PRNGKey,
):
    agent.reset_inner()

    # Adaptation
    obs, _ = zip(*train_envs.call("sample_tasks"))
    obs = np.stack(obs)
    eval_buffer = MultiTaskRolloutBuffer(train_envs.num_envs, adaptation_episodes, 500)

    for _ in range(adaptation_steps):
        while not eval_buffer.ready():
            action, log_probs, means, stds, key = agent.get_actions_train(obs, key)
            next_obs, reward, terminated, _, _ = train_envs.step(action)
            eval_buffer.push(obs, action, reward, terminated, log_probs, means, stds)
            obs = next_obs

        rollouts = eval_buffer.get(**buffer_kwargs)
        agent.adapt(rollouts)

    # Evaluation
    obs, _ = eval_envs.reset()
    eval_buffer = MultiTaskRolloutBuffer(eval_envs.num_envs, eval_episodes, 500)

    task_eps = np.zeros(eval_envs.num_envs)
    successes = np.zeros(eval_envs.num_envs)
    while not eval_buffer.ready():
        action, key = agent.get_actions_eval(obs, key)
        next_obs, reward, done, infos = eval_envs.step(action)

        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                task_eps[i] += 1
                # Discard any extra episodes from envs that ran ahead
                if task_eps <= eval_episodes:
                    successes[i] += int(info["success"])

        eval_buffer.push(obs, action, reward, done)
        obs = next_obs

    rollouts = eval_buffer.get(**buffer_kwargs)
    return (successes / eval_episodes).mean(), np.mean(rollouts.returns), key
