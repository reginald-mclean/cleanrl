import multiprocessing as mp
from typing import Callable, Optional, Type

import gymnasium as gym
import jax
import numpy as np

from cleanrl_utils.wrappers.metaworld_wrappers import OneHotV0


def evaluation_procedure(writer, agent, classes, tasks, keys, update, num_envs, add_onehot=True):
    mp.set_start_method("spawn")
    workers = []
    shared_queue = mp.Queue(num_envs)
    num_evals = 50
    eval_rewards = []
    mean_success_rate = 0.0
    task_results = []

    batch_size = 10 if num_envs >= 10 else num_envs
    itrs = int(num_envs / batch_size)
    for i in range(itrs):
        current_keys = keys[i * batch_size : (i + 1) * batch_size]
        for key in current_keys:
            env_cls = classes[key]
            env_tasks = [task for task in tasks if task.env_name == key]
            p = mp.Process(
                target=multiprocess_eval,
                args=(env_cls, env_tasks, key, agent, shared_queue, num_evals, add_onehot, keys.index(key), num_envs),
            )
            p.start()
            workers.append(p)
        for process in workers:
            process.join()
        for _ in range(len(current_keys)):
            worker_result = shared_queue.get()
            if worker_result["eval_rewards"] is not None:
                eval_rewards += worker_result["eval_rewards"]
                mean_success_rate += worker_result["success_rate"]
                task_results.append(
                    (worker_result["task_name"], worker_result["success_rate"], np.mean(worker_result["eval_rewards"]))
                )
                writer.add_scalar(
                    f"charts/{worker_result['task_name']}_success_rate", worker_result["success_rate"] / 50, update - 1
                )
                writer.add_scalar(
                    f"charts/{worker_result['task_name']}_avg_eval_rewards",
                    np.mean(worker_result["eval_rewards"]),
                    update - 1,
                )
    success_rate = float(mean_success_rate) / (num_envs * num_evals)
    writer.add_scalar("charts/mean_success_rate", success_rate, update - 1)
    return success_rate


def multiprocess_eval(
    env_cls: Type[gym.Env],
    env_tasks: list[str],
    env_name: str,
    agent,  # Agent expected to have a get_action function that takes in batches of obs and an rng key
    shared_queue: mp.Queue,
    num_evals: int,
    add_onehot: bool,
    idx: int,
    num_envs: int,
):
    env = env_cls()
    key = jax.random.PRNGKey(0)
    if add_onehot:
        env = OneHotV0(env, num_envs=num_envs, task_idx=idx)
    rewards = []
    success = 0.0
    for _ in range(num_evals):
        env.set_task(env_tasks[np.random.randint(0, len(env_tasks))])
        obs, info = env.reset()
        count = 0
        done = False
        while count < 500 and not done:
            key, action_key = jax.random.split(key)
            obs = jax.device_put(np.expand_dims(obs, axis=0))  # Add batch dim + put obs on GPU
            action = np.squeeze(jax.device_get(agent.get_action(obs, action_key)), axis=0)  # Reverse
            next_obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = truncated or terminated
            obs = next_obs
            count += 1
            if int(info["success"]) == 1:
                success += 1
                done = True
            if done:
                break

    shared_queue.put({"eval_rewards": rewards, "success_rate": success, "task_name": env_name})
