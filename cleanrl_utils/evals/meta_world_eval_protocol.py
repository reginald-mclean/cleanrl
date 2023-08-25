# ruff: noqa: E402
import torch
import torch.multiprocessing as mp

import time
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from cleanrl_utils.buffers_metaworld import MultiTaskRolloutBuffer

from cleanrl_utils.wrappers.metaworld_wrappers import OneHotWrapper, RandomTaskSelectWrapper
#from cleanrl.varibad_ppo import get_latent_for_policy



def evaluation_procedure(writer, agent, classes, tasks, keys, update, num_envs, add_onehot=True, writer_append='train'):
    workers = []
    manager = mp.Manager()
    shared_queue = manager.Queue(num_envs)
    num_evals = 2
    eval_rewards = []
    mean_success_rate = 0.0
    task_results = []
    agent = agent.to('cuda:0')
    batch_size = 10 if num_envs >= 10 else num_envs
    itrs = int(num_envs/batch_size)
    with torch.no_grad():
        for i in range(itrs):
            current_keys = keys[i*batch_size:(i+1)*batch_size]
            for key in current_keys:
                # print(f"process for {key}")
                env_cls = classes[key]
                env_tasks = [task for task in tasks if task.env_name == key]

                p = mp.Process(target=multiprocess_eval, args=(env_cls, env_tasks, key, agent, shared_queue, num_evals, add_onehot, keys.index(key), num_envs, 'cuda:0'))
                p.start()
                workers.append(p)
            for process in workers:
                process.join()
            for _ in range(len(current_keys)):
                worker_result = shared_queue.get()
                if worker_result['eval_rewards'] is not None:
                    eval_rewards += worker_result['eval_rewards']
                    mean_success_rate += worker_result['success_rate']
                    task_results.append((worker_result['task_name'], worker_result['success_rate'], np.mean(worker_result['eval_rewards'])))
                    writer.add_scalar(f"charts/{worker_result['task_name']}_success_rate "+writer_append, worker_result['success_rate']/50, update - 1)
                    writer.add_scalar(f"charts/{worker_result['task_name']}_avg_eval_rewards "+writer_append, np.mean(worker_result['eval_rewards']), update - 1)
        success_rate = float(mean_success_rate) / (num_envs * num_evals)
        writer.add_scalar("charts/mean_success_rate", success_rate, update - 1)
    return success_rate


def multiprocess_eval(env_cls, env_tasks, env_name, agent, shared_queue, num_evals, add_onehot, idx, num_envs, device):
    # print(f"Agent Device for {env_name} {next(agent.parameters()).device}")
    agent.eval()
    env = env_cls()
    if add_onehot:
        env = OneHotWrapper(env, idx, num_envs)
    env = RandomTaskSelectWrapper(env, env_tasks)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    rewards = []
    success = 0.0

    env.sample_tasks()
    obs, info = env.reset()

    for m in range(num_evals):
        print(f"{env_name} {m}")
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                        torch.from_numpy(obs).to(torch.float32).to(device).unsqueeze(0))
            next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
            rewards.append(reward)
            done = truncated or terminated
            obs = next_obs
            if int(info['success']) == 1:
                success += 1
                done = True
            if done:
                break

    shared_queue.put({
        'eval_rewards' : rewards,
        'success_rate' : success,
        'task_name'    : env_name
    })
