def evaluation_procedure(evaluation_envs, writer, agent, benchmark, keys, update, num_envs):
    mean_success_rate = 0.0
    success_rate_dicts = dict()
    for i, env in enumerate(evaluation_envs):
        current_task_success_rate = 0.0
        print(f'Evaluating {keys[i]}')
        tasks = [task for task in benchmark.train_tasks if task.env_name == keys[i]]
        for x in range(50):
            env.set_task(tasks[x])
            obs, _ = env.reset()
            count = 0
            done = False
            while count < 500 and not done:
                action, _, _, _ = agent.get_action_and_value(torch.from_numpy(obs).to(torch.float32).to(device))
                next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                done = truncated or terminated
                obs = next_obs
                count += 1
                if int(info['success']) == 1:
                    current_task_success_rate += 1
                    mean_success_rate += 1
                    done = True
                if done:
                    break
        success_rate_dicts[keys[i]] = float(current_task_success_rate) / 50
        writer.add_scalar(f"charts/{keys[i]}_success_rate", success_rate_dicts[keys[i]], update - 1)
    writer.add_scalar("charts/mean_success_rate", float(mean_success_rate) / (num_envs * 50), update - 1)