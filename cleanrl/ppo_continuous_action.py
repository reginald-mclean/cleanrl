# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import sys
sys.path.append('/home/reggiemclean/cleanrl')
import copy
from cleanrl_utils.evals.metaworld_jax_eval import evaluation
from cleanrl_utils.wrappers.metaworld_wrappers import OneHotV0, SyncVectorEnv
from cleanrl_utils.env_setup_metaworld import make_envs, make_eval_envs
from typing import Deque, NamedTuple, Optional, Tuple, Union
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers.vector_list_info import VectorListInfo
import metaworld

from scipy.ndimage import gaussian_filter1d, convolve1d

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
    parser.add_argument("--wandb-project-name", type=str, default="Reward Smoothing",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='reggies-phd-research',
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MT10",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2e7,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=10,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=10000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.97,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=16,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=5e-3,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.0,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--eval-freq", type=int, default=2,
        help="how many updates to do before evaluating the agent")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="how many episodes per environment to run an evaluation for")
    # reward smoothing
    parser.add_argument("--reward-filter", type=str, default=None)
    parser.add_argument('--filter-mode', type=str, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--kernel-type', type=str, default=None)

    # reward normalization
    parser.add_argument('--normalize-rewards', type=lambda x: bool(strtobool(x)), default=False, help='normalize after smoothing')

    # reward version
    parser.add_argument("--reward-version", default="v2", help="the reward function of the environment")

    args = parser.parse_args()
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape).to('cuda:0')
        self.var = torch.ones(shape).to('cuda:0')
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, axis=0)
        batch_var = torch.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512), std=1.0),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512), std=0.01),
            nn.Tanh(),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_eval(self, x, device='cuda:0'):
        x = torch.from_numpy(x).to(device)
        action, _, _, _ = self.get_action_and_value(x)
        return action.cpu().detach().numpy()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        if x.size()[0] == 10:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    print(os.environ['CUDA_DEVICE_ORDER'])
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    SAVE_PATH = f"/reggieUSB/RewardSmoothing/runs/{run_name}"

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.env_id == 'MT10':
        benchmark = metaworld.MT10(seed=args.seed)
    elif args.env_id == 'MT50':
        benchmark = metaworld.MT50(seed=args.seed)

    use_one_hot_wrapper = True if 'MT10' in args.env_id or 'MT50' in args.env_id else False
    print(use_one_hot_wrapper)
    # env setup

    envs = make_envs(
        benchmark, args.seed, 500, use_one_hot=use_one_hot_wrapper, reward_func_version=args.reward_version
    )

    keys = list(benchmark.train_classes.keys())

    args.num_envs = len(keys)

    global_episodic_return: Deque[float] = deque([], maxlen=20 * args.num_envs)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(torch.float32).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminations = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int).to(device)



    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = int(args.total_timesteps // args.batch_size)
    gamma = args.gamma
    epsilon = 1e-8
    e_returns = torch.zeros((envs.num_envs)).to(device)
    return_rms = RunningMeanStd(shape=(envs.num_envs, ))

    best_success = None
    best_success_epoch = None

    if args.delta:
        args.delta = int(args.delta)

    for update in range(1, num_updates + 1):
        if (update - 1) % args.eval_freq == 0:
            agent.eval()
            envs.set_attr('terminate_on_success', True)
            (
                eval_success_rate,
                eval_returns,
                eval_success_per_task,
            ) = evaluation(
                agent=agent,
                eval_envs=envs,
                num_episodes=args.evaluation_num_episodes,
            )
            eval_metrics = {
                "charts/mean_success_rate": float(eval_success_rate),
                "charts/mean_evaluation_return": float(eval_returns),
            } | {
                f"charts/{env_name}_success_rate": float(eval_success_per_task[i])
                for i, (env_name, _) in enumerate(benchmark.train_classes.items())
            }
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, global_step)
            print(
                f"total_steps={global_step}, mean evaluation success rate: {eval_success_rate:.4f}"
                + f" return: {eval_returns:.4f}"
            )

            if best_success is None or eval_success_rate > best_success:
                best_success = eval_success_rate
                torch.save(agent, f'{SAVE_PATH}/best.pth')

            envs.set_attr('terminate_on_success', False)
            next_obs, info = envs.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            agent.train()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            terminations[step] = torch.tensor(terminated).to(device)

            # need to apply normalization & smoothing per every 500 steps/1 episode



            if step > 0 and step % 500 == 499:
               if args.reward_filter:
                   if args.reward_filter == 'gaussian':
                       rewards[(step-499):step, :] = torch.from_numpy(gaussian_filter1d(rewards[(step-499):step, :].cpu().numpy(), args.sigma, mode=args.filter_mode, axis=0))
                   elif args.reward_filter == 'exponential':
                       rewards_cpu = rewards[(step-499):step, :].cpu().numpy()
                       rsmooth = np.zeros_like(rewards_cpu)
                       rsmooth[-1, :] = rewards_cpu[0, :]
                       beta = 1 - args.alpha
                       for i, rew_raw in enumerate(rewards_cpu):
                           rsmooth[i, :] = args.alpha * rsmooth[i - 1, :] + beta * rew_raw
                       rewards[(step-499):step, :] = torch.from_numpy(rsmooth).to(device)
                   elif args.reward_filter == 'uniform':
                       if args.kernel_type == 'uniform':
                           filter = (1.0 / args.delta) * np.array([1] * args.delta)
                       elif args.kernel_type == 'uniform_before':
                           filter = (1.0/args.delta) * np.array([1] * args.delta + [0] * (args.delta-1))
                       elif args.kernel_type == 'uniform_after':
                           filter = (1.0 / args.delta) * np.array([0] * (args.delta - 1) + [1] * args.delta)
                       else:
                           raise NotImplementedError('Invalid kernel type for uniform smoothing')
                       rewards[(step-499):step, :] = torch.from_numpy(convolve1d(rewards[(step-499):step, :].cpu().numpy(), filter, mode=args.filter_mode, axis=0))

               if args.normalize_rewards:
                   terminated = 1 - terminations[(step-499):step, :]
                   e_returns = e_returns * gamma * (1 - terminated) + rewards[(step-499):step, :]
                   return_rms.update(e_returns)
                   rewards[(step-499):step, :] = rewards[(step-499):step, :] / torch.sqrt(return_rms.var + epsilon)
                   # rewards[(step-499):step, :] = np.asarray(rewards[(step-499):step, :])



            #print(infos)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                #print(i, info)
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                global_episodic_return.append(info["episode"]["r"])


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar('charts/VF_Values', np.mean(np.mean(values.cpu().numpy(), axis=1)), global_step)
        print(
            f"global_step={global_step}, mean_episodic_return={np.mean(list(global_episodic_return))}"
        )

        torch.save(agent, f'{SAVE_PATH}/{global_step}_model.pth')


    envs.close()
    writer.close()
