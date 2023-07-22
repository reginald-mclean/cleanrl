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
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from cleanrl_utils.evals.meta_world_eval_protocol import evaluation_procedure
from cleanrl_utils.wrappers.metaworld_wrappers import (
    OneHotWrapper,
    RandomTaskSelectWrapper,
)


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
    parser.add_argument("--wandb-project-name", type=str, default="test",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="meta-world-benchmark",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MT10", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="total timesteps of the experiments")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="maximum number of timesteps in one episode during training")
    parser.add_argument("--evaluation-frequency", type=int, default=250_000, help="how many updates to before evaluating the agent")

    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=128, help="the batch size of sample from the replay memory")
    parser.add_argument("--learning-starts", type=int, default=5e3, help="timestep to start learning")

    parser.add_argument("--policy-lr", type=float, default=3e-4, help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="noise clip parameter of the Target Policy Smoothing Regularization")

    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


class MultiTaskReplayBuffer:
    def __init__(self, capacity, num_tasks, envs, device):
        self.capacity = capacity
        self.num_tasks = num_tasks
        self.device = device
        state_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.array(envs.single_action_space.shape).prod()
        self.obs = np.zeros((capacity, num_tasks, state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, num_tasks, action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_tasks), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_tasks, state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, num_tasks), dtype=np.float32)
        self.pos = 0

    def add(self, obs, action, reward, next_obs, done):
        task_idx = obs[:, -self.num_tasks :].argmax(1)

        self.obs[self.pos, task_idx] = obs
        self.actions[self.pos, task_idx] = action
        self.rewards[self.pos, task_idx] = reward
        self.next_obs[self.pos, task_idx] = next_obs
        self.dones[self.pos, task_idx] = done

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, single_task_batch_size):
        sample_idx = np.random.randint(
            0,
            high=max(self.pos, single_task_batch_size),
            size=(single_task_batch_size,),
        )
        obs = torch.tensor(self.obs[sample_idx]).to(self.device)
        actions = torch.tensor(self.actions[sample_idx]).to(self.device)
        rewards = torch.tensor(self.rewards[sample_idx]).to(self.device)
        next_obs = torch.tensor(self.next_obs[sample_idx]).to(self.device)
        dones = torch.tensor(self.dones[sample_idx]).to(self.device)
        mt_batch_size = single_task_batch_size * self.num_tasks
        return ReplayBufferSamples(
            observations=obs.reshape(mt_batch_size, -1),
            actions=actions.reshape(mt_batch_size, -1),
            next_observations=next_obs.reshape(mt_batch_size, -1),
            dones=dones.reshape(mt_batch_size, -1),
            rewards=rewards.reshape(mt_batch_size, -1),
        )


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
) -> gym.Env:
    def init_each_env(env_cls: Type[gym.Env], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            env = OneHotWrapper(env, env_id, len(benchmark.train_classes))
        env = RandomTaskSelectWrapper(
            env, [task for task in benchmark.train_tasks if task.env_name == name]
        )
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
    rb = MultiTaskReplayBuffer(args.buffer_size, NUM_TASKS, envs, device)

    start_time = time.time()

    global_episodic_return = deque([], maxlen=20 * envs.num_envs)
    global_episodic_length = deque([], maxlen=20 * envs.num_envs)

    obs, info = envs.reset()

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                global_episodic_return.append(info["episode"]["r"])
                global_episodic_length.append(info["episode"]["l"])
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncations):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, actions, rewards, real_next_obs, terminations)

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
                f"global_step={global_step}, mean_episodic_return={np.mean(global_episodic_return)}"
            )

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
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

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = (
                        (get_log_alpha(log_alpha, NUM_TASKS, data).exp() * log_pi)
                        - min_qf_pi
                    ).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

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
            if global_step % args.target_network_frequency == 0:
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
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
            # Evaluation
            if global_step % args.evaluation_frequency == 0:
                print("Evaluating...")
                eval_device = torch.device("cpu")
                eval_agent = Actor(envs, NUM_TASKS).to(eval_device)
                eval_success_rate = evaluation_procedure(
                    writer=writer,
                    agent=eval_agent,
                    classes=benchmark.train_classes,
                    tasks=benchmark.train_tasks,
                    keys=list(benchmark.train_classes.keys()),
                    update=global_step,
                    num_envs=envs.num_envs,
                    device=eval_device,
                )
                print(f"Evaluation success_rate: {eval_success_rate:.4f}")

    envs.close()
    writer.close()
