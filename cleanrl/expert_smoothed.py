from scipy.ndimage import convolve1d, gaussian_filter1d
import time
from metaworld import MT1
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS, MT10_V2
import importlib
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
import pickle
from PIL import Image
envs = list(MT10_V2.keys())
rewards = {key: [] for key in envs}
for e in envs:
    print(e)
    parts = e.split('-')
    new_parts = []
    for i in range(len(parts)):
        new_parts.append(parts[i][0].upper() + parts[i][1:])
    p_name = 'Sawyer' + ''.join(new_parts) + 'Policy'
    if e != 'peg-insert-side-v2':
        policy_class = importlib.import_module(f'metaworld.policies.sawyer_{e.replace("-", "_")}_policy')
        policy = getattr(policy_class, p_name)
    else:
        from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy as policy
    p = policy()
    mt1 = MT1(e, seed=42)
    task_num = 0
    current_results = []
    alpha = 0.8
    noisy = False
    num_steps = 500
    if noisy:
        folder = f'videos//{e}/'
    else:
        folder = f'videos/success_videos/{e}/'
    task_rewards = 0.0
    task_success = 0
    grasp_act = None
    for idx, task in enumerate(mt1.train_tasks):
        env = mt1.train_classes[e](render_mode='rgb_array', reward_func_version='v1')
        env.set_task(task)
        obs, info = env.reset()
        info['success'] = 0.0
        count = 0
        success_count = 0
        success_reward = 0.0
        first_grasp = None
        traj_rewards = []
        while count < num_steps:
            a = p.get_action(obs)
            next_state, reward, terminate, truncate, info = env.step(a)
            traj_rewards.append(reward)
            obs = next_state
            count += 1
        if env:
            env.close()
        rewards[e].append((traj_rewards, int(info['success'])))

with open('mw_rewards_unsmoothed.pkl', 'wb') as f:
    pickle.dump(rewards, f, protocol=pickle.HIGHEST_PROTOCOL)
