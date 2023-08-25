# ruff: noqa: E402
import torch
import torch.multiprocessing as mp

import time
from typing import List, Optional, Tuple

import gymnasium as gym
import jax
import numpy as np
import numpy.typing as npt

from cleanrl_utils.buffers_metaworld import MultiTaskRolloutBuffer

from cleanrl_utils.wrappers.metaworld_wrappers import OneHotWrapper, RandomTaskSelectWrapper
#from cleanrl.varibad_ppo import get_latent_for_policy

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
        with torch.no_grad():
            actions, _, _ = agent.get_action(torch.tensor(obs, device=device))
        obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())
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





def evaluation(
    agent,
    encoder,
    args,
    eval_envs: gym.vector.VectorEnv,
    num_episodes: int,
    task_names: Optional[List[str]] = None,
) -> Tuple[float, float, npt.NDArray, jax.random.PRNGKey]:
    print(f"Evaluating for {num_episodes} episodes.")
    obs, _ = eval_envs.reset()

    if task_names is not None:
        successes = {task_name: 0 for task_name in set(task_names)}
        episodic_returns = {task_name: [] for task_name in set(task_names)}
        envs_per_task = {task_name: task_names.count(task_name) for task_name in set(task_names)}
    else:
        successes = np.zeros(eval_envs.num_envs)
        episodic_returns = [[] for _ in range(eval_envs.num_envs)]

    start_time = time.time()

    def eval_done(returns):
        if type(returns) is dict:
            return all(len(r) >= (num_episodes * envs_per_task[task_name]) for task_name, r in returns.items())
        else:
            return all(len(r) >= num_episodes for r in returns)

    latent_mean, latent_sample, latent_logvar, hidden_state = encoder.prior(eval_envs.num_envs)

    while not eval_done(episodic_returns):
        latent = get_latent_for_policy(args=args, latent_mean=latent_mean, latent_logvar=latent_logvar, latent_sample=latent_sample)
        actions = agent.act(obs, latent, None, None, True)
        obs, reward, termn, trunc, infos = eval_envs.step(actions)
        done = np.logical_or(termn, trunc)
        if done is not None:
            hidden_state = encoder.reset_hidden(hidden_state, done)
        if encoder:
            latent_sample, latent_mean, latent_logvar, hidden_state = encoder(actions=actions.double(),
                                                                              states=obs,
                                                                              rewards=reward,
                                                                              hidden_state=hidden_state,
                                                                              return_prior=False)
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                if task_names is not None:
                    episodic_returns[task_names[i]].append(float(info["episode"]["r"][0]))
                    if len(episodic_returns[task_names[i]]) <= num_episodes * envs_per_task[task_names[i]]:
                        successes[task_names[i]] += int(info["success"])
                else:
                    episodic_returns[i].append(float(info["episode"]["r"][0]))
                    if len(episodic_returns[i]) <= num_episodes:
                        successes[i] += int(info["success"])

    if type(episodic_returns) is dict:
        episodic_returns = {
            task_name: returns[: (num_episodes * envs_per_task[task_name])]
            for task_name, returns in episodic_returns.items()
        }
    else:
        episodic_returns = [returns[:num_episodes] for returns in episodic_returns]

    print(f"Evaluation time: {time.time() - start_time:.2f}s")

    if type(successes) is dict:
        success_rate_per_task = np.array(
            [successes[task_name] / (num_episodes * envs_per_task[task_name]) for task_name in set(task_names)]
        )
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(list(episodic_returns.values()))
    else:
        success_rate_per_task = successes / num_episodes
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(episodic_returns)

    return mean_success_rate, mean_returns, success_rate_per_task


def metalearning_evaluation(
    agent,
    encoder,
    args,
    eval_envs: gym.vector.VectorEnv,
    eval_episodes: int,
    num_evals: int,
    key: jax.random.PRNGKey,
    task_names: Optional[List[str]] = None,
):
    # Adaptation
    total_mean_success_rate = 0.0
    total_mean_return = 0.0

    if task_names is not None:
        success_rate_per_task = np.zeros((num_evals, len(set(task_names))))
    else:
        success_rate_per_task = np.zeros((num_evals, eval_envs.num_envs))

    for i in range(num_evals):
        eval_envs.call("toggle_sample_tasks_on_reset", False)
        obs, _ = zip(*eval_envs.call("sample_tasks"))
        # Evaluation
        eval_envs.call("toggle_terminate_on_success", True)
        mean_success_rate, mean_return, _success_rate_per_task = evaluation(agent, encoder, args, eval_envs, eval_episodes, task_names)
        total_mean_success_rate += mean_success_rate
        total_mean_return += mean_return
        success_rate_per_task[i] = _success_rate_per_task

    success_rates = (success_rate_per_task).mean(axis=0)
    task_success_rates = {task_name: success_rates[i] for i, task_name in enumerate(set(task_names))}

    return total_mean_success_rate / num_evals, total_mean_return / num_evals, task_success_rates, key
