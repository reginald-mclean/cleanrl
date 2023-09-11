# ruff: noqa: E402
import torch
import torch.multiprocessing as mp

import gymnasium as gym
import numpy as np
from cleanrl_utils.wrappers.metaworld_wrappers import OneHotWrapper, RandomTaskSelectWrapper


def evaluation_procedure(writer, agent, classes, tasks, keys, update, num_envs, add_onehot=True, writer_append='train', encoder=None):
    workers = []
    manager = mp.Manager()
    shared_queue = manager.Queue(num_envs)
    num_evals = 50
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

                p = mp.Process(target=multiprocess_eval, args=(env_cls, env_tasks, key, agent, shared_queue, num_evals, False, keys.index(key), num_envs, 'cuda:0', encoder))
                p.start()
                workers.append(p)
            for process in workers:
                process.join()
            for _ in range(len(current_keys)):
                worker_result = shared_queue.get()
                if worker_result['eval_rewards'] is not None:
                    eval_rewards += worker_result['eval_rewards']
                    mean_success_rate += worker_result['success_rate']
                    writer.add_scalar(f"charts/{worker_result['task_name']}_success_rate_"+writer_append, worker_result['success_rate']/50, update - 1)
                    writer.add_scalar(f"charts/{worker_result['task_name']}_avg_eval_rewards_"+writer_append, np.mean(worker_result['eval_rewards']), update - 1)
        success_rate = float(mean_success_rate) / (num_envs * num_evals)
        writer.add_scalar("charts/mean_success_rate_"+writer_append, success_rate, update - 1)
        print(success_rate)
    return success_rate


def eval(writer, env, env_name, agent, num_evals,  device, update):
    # print(f"Agent Device for {env_name} {next(agent.parameters()).device}")
    #agent.eval()
    #env = env_cls()
    #if add_onehot:
    #    env = OneHotWrapper(env, idx, num_envs)
    #env = RandomTaskSelectWrapper(env, env_tasks, sample_tasks_on_reset=True)
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    rewards = []
    success = 0.0
    print(env)
    for m in range(num_evals):
        obs, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = agent.get_action_eval(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = truncated or terminated
            obs = next_obs
            if "final_info" in info:
                if int(info['final_info'][0]['success']) == 1:
                    success += 1
                    done = True
            if done:
                break

    print(float(success) / num_evals)
    writer.add_scalar('charts/mean_success_rate', float(success)/num_evals, update - 1)
