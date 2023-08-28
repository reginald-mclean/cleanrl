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

                p = mp.Process(target=multiprocess_eval, args=(env_cls, env_tasks, key, agent, shared_queue, num_evals, add_onehot, keys.index(key), num_envs, 'cuda:0', encoder))
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
    return success_rate


def multiprocess_eval(env_cls, env_tasks, env_name, agent, shared_queue, num_evals, add_onehot, idx, num_envs, device, encoder):
    # print(f"Agent Device for {env_name} {next(agent.parameters()).device}")
    agent.eval()
    env = env_cls()
    if add_onehot:
        env = OneHotWrapper(env, idx, num_envs)
    env = RandomTaskSelectWrapper(env, env_tasks, sample_tasks_on_reset=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    rewards = []
    success = 0.0
    if encoder:
        encoder.to(device)
        encoder.eval()
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(1)
    for m in range(num_evals):
        env.get_wrapper_attr('sample_tasks')()
        for step in range(2):
            print(f"{env_name} {m} {step}")
            obs, info = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    if encoder:
                        obs = torch.from_numpy(obs).unsqueeze(0).to(device)
                        action = agent.act(state=obs.float(), latent=latent_sample.to(device).float(), belief=None, task=None, deterministic=False, only_action=True)
                    else:
                        action, _, _, _ = agent.get_action_and_value(
                            torch.from_numpy(obs).to(torch.float32).to(device).unsqueeze(0))
                next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
                if encoder:
                    with torch.no_grad():
                        latent_sample, latent_mean, latent_logvars, hidden_state = encoder(actions=action.float().unsqueeze(0),
                                                                                       states=torch.from_numpy(next_obs).to(device).unsqueeze(0),
                                                                                       rewards=torch.tensor(reward, dtype=torch.float64).to(device).unsqueeze(0).unsqueeze(0),
                                                                                       hidden_state=hidden_state.cpu(),
                                                                                       return_prior=False)
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
