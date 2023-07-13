import torch
import torch.multiprocessing as mp
import numpy as np
from cleanrl_utils.wrappers.metaworld_wrappers import OneHotV0

def evaluation_procedure(writer, agent, classes, tasks, keys, update, num_envs, add_onehot=True, device=None):
    workers = []
    manager = mp.Manager()
    shared_queue = manager.Queue(num_envs)
    num_evals = 50
    eval_rewards = []
    mean_success_rate = 0.0
    task_results = []

    batch_size = 10 if num_envs >= 10 else num_envs
    itrs = int(num_envs/batch_size)
    for i in range(itrs):
        current_keys = keys[i*batch_size:(i+1)*batch_size]
        for key in current_keys:
            # print(f"process for {key}")
            env_cls = classes[key]
            env_tasks = [task for task in tasks if task.env_name == key]
            p = mp.Process(target=multiprocess_eval, args=(env_cls, env_tasks, key, agent, shared_queue, num_evals, add_onehot, keys.index(key), num_envs, device))
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
                writer.add_scalar(f"charts/{worker_result['task_name']}_success_rate", worker_result['success_rate']/50, update - 1)
                writer.add_scalar(f"charts/{worker_result['task_name']}_avg_eval_rewards", np.mean(worker_result['eval_rewards']), update - 1)
    success_rate = float(mean_success_rate) / (num_envs * num_evals)
    writer.add_scalar("charts/mean_success_rate", success_rate, update - 1)
    return success_rate

def multiprocess_eval(env_cls, env_tasks, env_name, agent, shared_queue, num_evals, add_onehot, idx, num_envs, device):
    # print(f"Agent Device for {env_name} {next(agent.parameters()).device}")
    env = env_cls()
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
            action, _, _, _ = agent.get_action_and_value(
                torch.from_numpy(obs).to(torch.float32).to(device).unsqueeze(0))
            next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
            rewards.append(reward)
            done = truncated or terminated
            obs = next_obs
            count += 1
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
