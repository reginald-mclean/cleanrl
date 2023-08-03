# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
from functools import partial
import os
import random
import time
from distutils.util import strtobool
from typing import Optional, Type

import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from cleanrl_utils.buffers_metaworld import MultiTaskReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

from cleanrl_utils.evals.meta_world_eval_protocol import new_evaluation_procedure
from cleanrl_utils.wrappers import metaworld_wrappers
from cleanrl.softmodules_metaworld_jax import make_eval_envs


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Metaworld-CleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MT10", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=15_000_000, help="total timesteps of the experiments")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="maximum number of timesteps in one episode during training")
    parser.add_argument("--evaluation-frequency", type=int, default=100_000, help="how many updates to before evaluating the agent")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50, help="the number episodes to run per evaluation")

    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="the replay memory buffer size for a single task")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=1280, help="the batch size of sample from the replay memory for a single task")
    parser.add_argument("--learning-starts", type=int, default=4e3, help="timestep to start learning")
    parser.add_argument("--gradient-steps", type=int, default=5)

    parser.add_argument("--policy-lr", type=float, default=3e-4, help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=1, help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="noise clip parameter of the Target Policy Smoothing Regularization")

    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            400,
        )
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, num_task_heads):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
        self.num_task_heads = num_task_heads
        self.fc2 = nn.Linear(400, 400 * self.num_task_heads)

        self.fc_mean = nn.Linear(400, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(400, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x.shape[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # extract the task ids from the one-hot encodings of the observations
        task_idx = (
            x[:, -self.num_task_heads :].argmax(1).unsqueeze(1).detach().to(x.device)
        )
        indices = torch.arange(400).unsqueeze(0).to(x.device) + task_idx * 400
        x = x.gather(1, indices)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action_and_value(self, x):
        return *self.get_action(x), None


# from garage
def get_log_alpha(log_alpha, num_tasks, data: ReplayBufferSamples):
    obs = data.observations
    one_hots = obs[:, -num_tasks:]
    if (
        log_alpha.shape[0] != one_hots.shape[1]
        or one_hots.shape[1] != num_tasks
        or log_alpha.shape[0] != num_tasks
    ):
        raise ValueError(
            "The number of tasks in the environment does "
            "not match self._num_tasks. Are you sure that you passed "
            "The correct number of tasks?"
        )
    ret = torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()
    return ret.unsqueeze(-1)


def make_envs(
    benchmark: metaworld.Benchmark,
    seed: int,
    max_episode_steps: Optional[int] = None,
    use_one_hot: bool = True,
) -> gym.vector.VectorEnv:
    def init_each_env(env_cls: Type[SawyerXYZEnv], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            env = metaworld_wrappers.OneHotWrapper(
                env, env_id, len(benchmark.train_classes)
            )
        tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        env = metaworld_wrappers.RandomTaskSelectWrapper(env, tasks)
        env.action_space.seed(seed)
        return env

    return gym.vector.AsyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id)
            for env_id, (name, env_cls) in enumerate(benchmark.train_classes.items())
        ]
    )


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if args.env_id == "MT10":
        benchmark = metaworld.MT10(seed=args.seed)
    elif args.env_id == "MT50":
        benchmark = metaworld.MT50(seed=args.seed)
    else:
        benchmark = metaworld.MT1(args.env_id, seed=args.seed)

    NUM_TASKS = len(benchmark.train_classes)

    use_one_hot_wrapper = "MT10" in args.env_id or "MT50" in args.env_id
    envs = make_envs(
        benchmark, args.seed, args.max_episode_steps, use_one_hot=use_one_hot_wrapper
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    actor = Actor(envs, NUM_TASKS).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.tensor(
            [0] * NUM_TASKS, device=device, dtype=torch.float32
        ).requires_grad_()
        a_optimizer = optim.Adam([log_alpha] * NUM_TASKS, lr=args.q_lr)
    else:
        log_alpha = torch.tensor(
            [0] * NUM_TASKS, device=device, dtype=torch.float32
        ).log()

    envs.single_observation_space.dtype = np.float32
    rb = MultiTaskReplayBuffer(
        capacity=args.buffer_size,
        num_tasks=NUM_TASKS,
        envs=envs,
        use_torch=True,
        seed=args.seed,
        device=device,
    )

    start_time = time.time()

    global_episodic_return = deque([], maxlen=20 * envs.num_envs)
    global_episodic_length = deque([], maxlen=20 * envs.num_envs)

    obs, info = envs.reset()

    global_step = 0
    env_steps = 0
    grad_steps = 0

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        env_steps += actions.shape[0]

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                global_episodic_return.append(info["episode"]["r"])
                global_episodic_length.append(info["episode"]["l"])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncations):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 500 == 0 and global_episodic_return:
            writer.add_scalar(
                "charts/mean_episodic_return",
                np.mean(global_episodic_return),
                global_step,
            )
            writer.add_scalar(
                "charts/mean_episodic_length",
                np.mean(global_episodic_length),
                global_step,
            )
            print(
                f"global_step={global_step}, env_steps={env_steps} mean_episodic_return={np.mean(global_episodic_return)}"
            )

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            qf_grad_num_elements = [p.numel() if p.requires_grad is True else 0 for group in q_optimizer.param_groups
                                    for p in group['params']]
            qf_grad_shapes = [p.shape if p.requires_grad is True else None for group in q_optimizer.param_groups
                                    for p in group['params']]
            actor_grad_num_element = [p.numel() if p.requires_grad is True else 0 for group in
                                      actor_optimizer.param_groups for p in group['params']]
            actor_grad_shapes = [p.shape if p.requires_grad is True else None for group in actor_optimizer.param_groups
                              for p in group['params']]

            for _ in range(args.gradient_steps):
                data = rb.sample(args.batch_size)
                qf_losses = []
                policy_losses = []
                qf_grad_tasks = []
                actor_grad_tasks = []
                q_optimizer.zero_grad()
                actor_optimizer.zero_grad()
                for i in range(args.num_envs):
                    assert 1 == 0, 'Need to sample data per task'
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _ = actor.get_action(
                            data.next_observations
                        )
                        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                        min_qf_next_target = (
                            torch.min(qf1_next_target, qf2_next_target)
                            - get_log_alpha(log_alpha, NUM_TASKS, data).exp()
                            * next_state_log_pi
                        )
                        next_q_value = data.rewards.flatten() + (
                            1 - data.dones.flatten()
                        ) * args.gamma * (min_qf_next_target).view(-1)

                    qf1_a_values = qf1(
                        data.observations, data.actions.type(torch.float32)
                    ).view(-1)
                    qf2_a_values = qf2(
                        data.observations, data.actions.type(torch.float32)
                    ).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    # q_optimizer should be zeroed for each task's data
                    qf_loss.backward()
                    devices = [
                        p.device for group in q_optimizer.param_groups for p in group['params']]
                    qf_grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                               else None for group in q_optimizer.param_groups for p in group['params']]
                    qf_grad_tasks.append(torch.cat([g if g is not None else torch.zeros(
                        qf_grad_num_elements[i], device=devices[i]) for i, g in enumerate(qf_grad)]))

                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = (
                        (get_log_alpha(log_alpha, NUM_TASKS, data).exp() * log_pi)
                        - min_qf_pi
                    ).mean()

                    actor_loss.backward()

                    devices = [p.device for group in actor_optimizer.param_groups for p in group['params']]
                    actor_grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                               else None for group in actor_optimizer.param_groups for p in group['params']]
                    actor_grad_tasks.append(torch.cat([g if g is not None else torch.zeros(
                        actor_grad_num_elements[i], device=devices[i]) for i, g in enumerate(actor_grad)]))

                    # get the gradients from each network, project them, then update networks

                    actor_optimizer.zero_grad()
                    q_optimizer.zero_grad()

                random.shuffle(qf_grad_tasks)
                random.shuffle(actor_grad_tasks)

                qf_grads_task = torch.stack(qf_grad_tasks, dim=0)
                qf_proj_grad = qf_grads_task.clone()

                actor_grads_task = torch.stack(actor_grad_tasks, dim=0)
                actor_proj_grad = actor_grads_task.clone()


                def _project_gradients_actor(grads):
                    for i in range(args.num_envs):
                        inner_product = torch.sum(grads * actor_grads_task[i])
                        projection_direction = inner_product / \
                                               (torch.sum(actor_grads_task[i]*actor_grads_task[i]) + 1e-12)
                        grads = grads - torch.min(projection_direction,
                                                          torch.zeros_like(projection_direction)) ** actor_grads_task[i]
                    return grads

                def _project_gradients_qf(grads):
                    for i in range(args.num_envs):
                        inner_product = torch.sum(grads * qf_grads_task[i])
                        projection_direction = inner_product / \
                                               (torch.sum(qf_grads_task[i]*qf_grads_task[i]) + 1e-12)
                        grads = grads - torch.min(projection_direction,
                                                          torch.zeros_like(projection_direction)) ** qf_grads_task[i]
                    return grads
                #  torch.vmap() ?? 
                proj_qf_grads = torch.sum(torch.stack(list(map(_project_gradients_qf, list(qf_proj_grad)))), dim=0)
                proj_actor_grads = torch.sum(torch.stack(list(map(_project_gradients_actor,
                                                                  list(actor_proj_grad)))), dim=0)
                indices_qf = [0, ] + [v for v in accumulate(qf_grad_num_elements)]
                params_qf = [p for group in q_optimizer.param_groups for p in group['params']]
                assert len(params_qf) == len(qf_grad_shapes) == len(indices_qf[:-1])

                for param, grad_shape, start_idx, end_idx in zip(params_qf, qf_grad_shapes, indices_qf[:-1], indices_qf[1:]):
                    if grad_shape is not None:
                        param.grad[...] = proj_qf_grads[start_idx:end_idx].view(grad_shape)  # copy proj grad

                indices_actor = [0, ] + [v for v in accumulate(actor_grad_num_element)]
                params_actor = [p for group in actor_optimizer.param_groups for p in group['params']]
                assert len(params_actor) == len(actor_grad_shapes) == len(indices_actor[:-1])
                for param, grad_shape, start_idx, end_idx in zip(params_actor, actor_grad_shapes, indices_actor[:-1], indices_actor[1:]):
                    if grad_shape is not None:
                        param.grad[...] = proj_actor_grads[start_idx:end_idx].view(grad_shape)  # copy proj grad



                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (
                        -get_log_alpha(log_alpha, NUM_TASKS, data)
                        * (log_pi + target_entropy)
                    ).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.sum().exp().item()

                # update the target networks
                if grad_steps % args.target_network_frequency == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

                grad_steps += 1

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

                writer.add_scalar(
                    "charts/grad_steps",
                    grad_steps,
                    global_step,
                )
                writer.add_scalar(
                    "charts/env_steps",
                    env_steps,
                    global_step,
                )

                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
        # Evaluation
        if global_step % args.evaluation_frequency == 0 and global_step > 0:
            print(f"Evaluating... at global_step={global_step}")
            eval_envs = make_eval_envs(
                benchmark,
                args.seed,
                args.max_episode_steps,
                use_one_hot=use_one_hot_wrapper,
            )
            eval_success_rate, eval_returns = new_evaluation_procedure(
                actor, eval_envs, args.evaluation_num_episodes, device
            )
            writer.add_scalar(
                "charts/mean_success_rate", eval_success_rate, global_step
            )
            writer.add_scalar(
                "charts/mean_evaluation_return", eval_returns, global_step
            )
            print(
                f"global_step={global_step}, mean evaluation success rate: {eval_success_rate:.4f}"
                + f" return: {eval_returns:.4f}"
            )

    envs.close()
    writer.close()
