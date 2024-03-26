import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from pynvml import *
import time
from cleanrl_utils.wrappers.metaworld_wrappers import OneHotWrapper, RandomTaskSelectWrapper

def new_evaluation_procedure(
    agent,
    eval_envs: gym.vector.VectorEnv,
    num_episodes: int,
    device = torch.device("cpu")
):
    obs, _ = eval_envs.reset()
    successes = np.zeros(eval_envs.num_envs)
    episodic_returns = [[] for _ in range(eval_envs.num_envs)]

    start_time = time.time()

    while not all(len(returns) >= num_episodes for returns in episodic_returns):
        actions, _, _ = agent.get_action(torch.Tensor(obs).to(device))
        obs, _, _, _, infos = eval_envs.step(actions.detach().cpu().numpy())
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

    return (successes / num_episodes).mean(), np.mean(episodic_returns)


def evaluation_procedure(writer, agent, classes, tasks, keys, update, num_envs, add_onehot=True, device=None):
    workers = []
    manager = mp.Manager()
    shared_queue = manager.Queue(num_envs)
    num_evals = 50
    eval_rewards = []
    mean_success_rate = 0.0
    task_results = []
    agent = agent.to('cuda:0')
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
    while info.free < (info.total/2):
        print('waiting for memory')
        time.sleep(30)
        info = nvmlDeviceGetMemoryInfo(h)

    batch_size = 10 if num_envs >= 10 else num_envs
    itrs = int(num_envs/batch_size)
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
                writer.add_scalar(f"charts/{worker_result['task_name']}_success_rate", worker_result['success_rate']/50, update - 1)
                writer.add_scalar(f"charts/{worker_result['task_name']}_avg_eval_rewards", np.mean(worker_result['eval_rewards']), update - 1)
    success_rate = float(mean_success_rate) / (num_envs * num_evals)
    writer.add_scalar("charts/mean_success_rate", success_rate, update - 1)
    #agent = agent.to('cuda:0')
    return success_rate

def multiprocess_eval(env_cls, env_tasks, env_name, agent, shared_queue, num_evals, add_onehot, idx, num_envs, device):
    # print(f"Agent Device for {env_name} {next(agent.parameters()).device}")
    agent.eval()
    env = env_cls()
    if add_onehot:
        env = OneHotWrapper(env, idx, num_envs)
    env = RandomTaskSelectWrapper(env, env_tasks)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    rewards = []
    success = 0.0
    for m in range(num_evals):
        print(f"{env_name} {m}")
        obs, info = env.reset()
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
