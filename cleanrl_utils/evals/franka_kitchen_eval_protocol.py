
import gymnasium as gym
import torch
import torch.multiprocessing as mp
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo
from datetime import datetime
import sys
import os


def evaluation_procedure(writer, agent, tasks, update, num_envs, add_onehot=True, device=torch.device("cpu")):
    workers = []
    manager = mp.Manager()
    shared_queue = manager.Queue(num_envs)
    num_evals = 50
    eval_rewards = []
    mean_success_rate = 0.0
    task_results = []

    batch_size = 10 if num_envs >= 10 else num_envs
    itrs = int(num_envs/batch_size)
    for i, task in enumerate(tasks):
        print(f"process for {task}")
        p = mp.Process(target=multiprocess_eval, args=(task, agent, shared_queue, num_evals, add_onehot, i, num_envs, torch.device("cpu"), update))
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
    writer.add_scalar("charts/mean_success_rate", float(mean_success_rate) / (num_envs * num_evals), update - 1)

def multiprocess_eval(env_tasks, agent, shared_queue, num_evals, add_onehot, idx, num_envs, device=torch.device("cpu"), update=None):
    print(f"Agent Device for {env_tasks} {next(agent.parameters()).device}")
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=[env_tasks], render_mode='rgb_array')
    #if add_onehot:
    #    env = OneHotV0(env, num_envs=num_envs, task_idx=idx)
    rewards = []
    success = 0.0
    one_hot = np.zeros(10)
    one_hot[idx] = 1
    record_episodes = [np.random.randint(0, num_evals) for _ in range(1)]
    print(f"{env_tasks} {record_episodes}")
    for x in range(num_evals):
        if x in record_episodes:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            if not os.path.isdir(f'/data/recorded_videos/PPO/metaworld/{env_tasks}_{update}_{current_time}'):
                os.mkdir(f'/data/recorded_videos/PPO/metaworld/{env_tasks}_{update}_{current_time}')
            env = RecordVideo(env, f'/data/recorded_videos/PPO/metaworld/{env_tasks}_{update}_{current_time}')
            print(current_time)
            env.start_video_recorder()
        sys.stdout.flush()
        obs, info = env.reset()
        count = 0
        done = False
        while count < 500 and not done:
            obs = np.concatenate([obs['observation'], one_hot])
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(torch.from_numpy(obs).to(torch.float32).to(device).unsqueeze(0))
            next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy(), env_tasks)
            rewards.append(reward)
            done = truncated or terminated
            obs = next_obs
            count += 1
            if int(info['success']) == 1:
                success += 1
                done = True
            if done:
                break
        if x in record_episodes:
            env.close()
            env = gym.make('FrankaKitchen-v1', tasks_to_complete=[env_tasks], render_mode='rgb_array')

    shared_queue.put({
        'eval_rewards' : rewards,
        'success_rate' : success,
        'task_name'    : f'Franka-Kitchen {env_tasks}'
    })


