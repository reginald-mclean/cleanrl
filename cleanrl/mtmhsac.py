import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Tuple

import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cleanrl_utils.buffers_metaworld import MultiTaskReplayBuffer
from cleanrl_utils.evals.meta_world_eval_protocol import new_evaluation_procedure
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch.utils.tensorboard import SummaryWriter

from cleanrl.cleanrl_utils.env_setup_metaworld import make_envs, make_eval_envs

DISABLE_COMPILE = os.environ.get("DISABLE_COMPILE", False)


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
    parser.add_argument("--max-episode-steps", type=int, default=None,
        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--evaluation-frequency", type=int, default=100_000,
        help="how many updates to before evaluating the agent")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes to run per evaluation")

    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size for a single task")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the replay memory for a single task")
    parser.add_argument("--learning-starts", type=int, default=4e3, help="timestep to start learning")
    parser.add_argument("--gradient-steps", type=int, default=200)

    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")

    parser.add_argument("--alpha", type=float, default=1.0, help="Entropy regularization coefficient.")
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
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

    def get_action_and_value(self, x):
        return *self.get_action(x), None


# from garage
@torch.compile(mode="reduce-overhead", disable=DISABLE_COMPILE)
def get_log_alpha(log_alpha, num_tasks, data: ReplayBufferSamples):
    one_hots = data.observations[:, -num_tasks:]
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
    return torch.mm(one_hots, log_alpha.unsqueeze(0).t())


@torch.compile(mode="reduce-overhead", disable=DISABLE_COMPILE)
def get_actions(actor: Actor, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        actions, _, _ = actor.get_action(obs)
    return actions


QFs = Tuple[SoftQNetwork, SoftQNetwork]


@torch.compile(mode="reduce-overhead", disable=DISABLE_COMPILE)
def sac_loss(
    actor: Actor,
    qfs: QFs,
    target_qfs: QFs,
    log_alpha: torch.Tensor,
    data: ReplayBufferSamples,
    autotune: bool = True,
    target_entropy: float = 0.0,
    optimizers=Tuple[optim.Optimizer, ...],
) -> dict:
    qf1, qf2 = qfs
    qf1_target, qf2_target = target_qfs
    q_optimizer, actor_optimizer, a_optimizer = optimizers
    alpha = get_log_alpha(log_alpha, NUM_TASKS, data).exp().detach()

    # QF Loss
    with torch.no_grad():
        next_state_actions, next_state_log_pi, _ = actor.get_action(
            data.next_observations
        )
        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
        min_qf_next_target = (
            torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        )
        next_q_value = data.rewards.flatten() + (
            1 - data.dones.flatten()
        ) * args.gamma * (min_qf_next_target).view(-1)

    qf1_a_values = qf1(data.observations, data.actions.type(torch.float32)).view(-1)
    qf2_a_values = qf2(data.observations, data.actions.type(torch.float32)).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    q_optimizer.zero_grad()
    qf_loss.backward()
    q_optimizer.step()

    # Actor loss
    pi, log_pi, _ = actor.get_action(data.observations)
    qf1_pi = qf1(data.observations, pi)
    qf2_pi = qf2(data.observations, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    if autotune:  # Alpha loss
        alpha_loss = (
            -get_log_alpha(log_alpha, NUM_TASKS, data)
            * (log_pi.detach() + target_entropy)
        ).mean()

        a_optimizer.zero_grad()
        alpha_loss.backward()
        a_optimizer.step()
        alpha = log_alpha.sum().exp().item()

    logs = {
        "losses/qf1_values": qf1_a_values.mean().item(),
        "losses/qf2_values": qf2_a_values.mean().item(),
        "losses/qf1_loss": qf1_loss.item(),
        "losses/qf2_loss": qf2_loss.item(),
        "losses/qf_loss": qf_loss.item() / 2.0,
        "losses/actor_loss": actor_loss.item(),
        "losses/alpha": alpha,
    }

    if autotune:
        logs["losses/alpha_loss"] = alpha_loss.item()

    return logs


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
    print(f"Using: {device}")

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
    eval_envs = make_eval_envs(
        benchmark,
        args.seed,
        args.max_episode_steps,
        use_one_hot=use_one_hot_wrapper,
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
            [args.alpha] * NUM_TASKS, device=device, dtype=torch.float32
        ).requires_grad_()
        a_optimizer = optim.Adam([log_alpha] * NUM_TASKS, lr=args.q_lr)
    else:
        log_alpha = torch.tensor(
            [args.alpha] * NUM_TASKS, device=device, dtype=torch.float32
        )

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

    global_episodic_return = deque([], maxlen=20 * NUM_TASKS)
    global_episodic_length = deque([], maxlen=20 * NUM_TASKS)

    obs, info = envs.reset()

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(NUM_TASKS)]
            )
        else:
            actions = get_actions(actor, torch.tensor(obs, device=device)).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                global_episodic_return.append(info["episode"]["r"])
                global_episodic_length.append(info["episode"]["l"])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, t in enumerate(truncations):
            if t:
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
                f"global_step={global_step}, mean_episodic_return={np.mean(global_episodic_return)}"
            )

        # ALGO LOGIC: training.
        if (
            global_step > args.learning_starts
            and global_step % args.gradient_steps == 0
        ):  # torchrl-style training loop
            for epoch_step in range(args.gradient_steps):
                current_step = global_step + epoch_step

                data = rb.sample(args.batch_size)
                logs = sac_loss(
                    actor=actor,
                    qfs=(qf1, qf2),
                    target_qfs=(qf1_target, qf2_target),
                    log_alpha=log_alpha,
                    data=data,
                    autotune=args.autotune,
                    target_entropy=target_entropy,
                    optimizers=(q_optimizer, actor_optimizer, a_optimizer),
                )

                # update the target networks
                if current_step % args.target_network_frequency == 0:
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

                if current_step % 100 == 0:
                    for k, v in logs.items():
                        writer.add_scalar(k, v, current_step)
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

        # Evaluation
        if global_step % args.evaluation_frequency == 0 and global_step > 0:
            print(f"Evaluating... at global_step={global_step}")
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
