import argparse
import os
import random
import time
from distutils.util import strtobool

from cleanrl_utils.wrappers import metaworld_wrappers
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Type
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.buffers_metaworld import MetaRolloutBuffer
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
import metaworld
from collections import deque


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Metaworld-CleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # RL^2 arguments
    parser.add_argument("--n-episodes-per-trial", type=int, default=30,
                        help="number of episodes sampled per trial/meta-batch")
    parser.add_argument("--recurrent-state-size", type=int, default=128)
    parser.add_argument("--encoder-hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-num-layers", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="maximum number of timesteps in one episode during training")

    parser.add_argument("--use-gae", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ML10",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2e7,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=15,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=2e-3,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--eval-freq", type=int, default=100_000,
        help="how many steps to do before evaluating the agent")
    parser.add_argument("--evaluation-num-episodes", type=int, default=10,
        help="the number episodes to run per evaluation")

    args = parser.parse_args()
    # fmt: on
    return args


class Critic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.fc1 = nn.Linear(
            args.recurrent_state_size,
            400,
        )
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.fc1 = nn.Linear(args.recurrent_state_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc_mean = nn.Linear(400, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(400, np.prod(env.single_action_space.shape))

    def forward(self, x, action=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        action_std = torch.exp(log_std)
        probs = torch.distributions.Normal(mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(-1).view(*x.shape[:-1], 1),
            probs.entropy().sum(-1).view(*x.shape[:-1], 1),
        )


class Agent(nn.Module):
    def __init__(self, envs, args, device):
        super().__init__()
        self.args = args
        self.actor = Actor(envs, args)
        self.critic = Critic(envs, args)
        self.device = device

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        self.obs_enc = nn.Linear(obs_dim, args.encoder_hidden_size).to(device)

        self.recurrent_state_size = args.recurrent_state_size
        self.rnn = nn.GRU(
            args.encoder_hidden_size,
            self.recurrent_state_size,
            batch_first=True,
            num_layers=args.recurrent_num_layers,
        ).to(device)

    def init_state(self, batch_size):
        self.rnn_state = torch.randn(
            self.args.recurrent_num_layers, batch_size, self.recurrent_state_size
        ).to(self.device)

    def recurrent_state(self, obs, rnn_state, training=False):
        obs_enc = self.obs_enc(obs)
        rnn_out, rnn_state_out = self.rnn(obs_enc, rnn_state)
        if not training:
            rnn_out = rnn_out.squeeze(1)
        return rnn_out, rnn_state_out

    def get_value(self, obs, training=False):
        rnn_out, self.rnn_state = self.recurrent_state(obs, self.rnn_state, training)
        return self.critic(rnn_out)

    def get_action_and_log_prob(self, obs, action=None, training=False):
        rnn_out, self.rnn_state = self.recurrent_state(obs, self.rnn_state, training)
        action, log_prob, entropy = self.actor(rnn_out, action)
        return action, log_prob, entropy

    def get_action(self, obs):
        with torch.no_grad():
            rnn_out, self.rnn_state = self.recurrent_state(obs, self.rnn_state)
            return self.actor(rnn_out)[0]

    def step(self, obs):
        with torch.no_grad():
            rnn_out, self.rnn_state = self.recurrent_state(obs, self.rnn_state)

            (
                action,
                log_prob,
                entropy,
            ) = self.actor(rnn_out)
            value = self.critic(rnn_out)

        return action, value, log_prob


def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    max_episode_steps: Optional[int] = None,
    terminate_on_success: bool = False,
    train=True,
) -> gym.vector.VectorEnv:
    def init_each_env(env_cls: Type[SawyerXYZEnv], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env = metaworld_wrappers.RL2Env(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if train:
            tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        else:
            tasks = [task for task in benchmark.test_tasks if task.env_name == name]
        env = metaworld_wrappers.RandomTaskSelectWrapper(env, tasks)
        env.action_space.seed(seed)
        return env

    if train:
        classes = benchmark.train_classes
    else:
        classes = benchmark.test_classes

    return gym.vector.SyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id)
            for env_id, (name, env_cls) in enumerate(classes.items())
        ]
    )


make_envs = partial(_make_envs_common, terminate_on_success=False)
make_eval_envs = partial(_make_envs_common, terminate_on_success=True)


def rl2_evaluation(
    args,
    agent,
    eval_envs: gym.vector.VectorEnv,
    num_episodes: int,
    device=torch.device("cpu"),
):
    obs, _ = eval_envs.reset()
    NUM_TASKS = eval_envs.num_envs

    successes = np.zeros(NUM_TASKS)
    episodic_returns = [[] for _ in range(NUM_TASKS)]

    start_time = time.time()
    agent.init_state(NUM_TASKS)

    while not all(len(returns) >= num_episodes for returns in episodic_returns):
        with torch.no_grad():
            obs = torch.tensor(obs).to(device).float().unsqueeze(1)
            action = agent.get_action(obs)

        obs, reward, _, _, infos = eval_envs.step(action.cpu().numpy())

        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                # reset states of finished episodes
                _state = agent.rnn_state[:, i : i + 1]
                agent.rnn_state[:, i : i + 1] = torch.randn_like(_state)
                episodic_returns[i].append(float(info["episode"]["r"][0]))
                if len(episodic_returns[i]) <= num_episodes:
                    successes[i] += int(info["success"])

    episodic_returns = [returns[:num_episodes] for returns in episodic_returns]

    print(f"Evaluation time: {time.time() - start_time:.2f}s")

    success_rate_per_task = successes / num_episodes
    mean_success_rate = np.mean(success_rate_per_task)
    mean_returns = np.mean(episodic_returns)

    return mean_success_rate, mean_returns, success_rate_per_task


def update_rl2_ppo(agent: Agent, meta_rb: MetaRolloutBuffer, device, total_steps, args):
    clipfracs = []
    batch, batch_size = meta_rb.sample()
    # TODO: Should meta trial only contain episodes from the same task?
    obs_batch = (
        batch["obs"].to(device).view(-1, args.max_episode_steps, batch["obs"].shape[-1])
    )
    action_batch = (
        batch["action"]
        .to(device)
        .view(-1, args.max_episode_steps, batch["action"].shape[-1])
    )
    value_batch = batch["value"].to(device).view(-1, args.max_episode_steps, 1)
    advantage_batch = batch["advantage"].to(device).view(-1, args.max_episode_steps, 1)
    old_logprob_batch = batch["log_prob"].to(device).view(-1, args.max_episode_steps, 1)
    return_batch = batch["return"].to(device).view(-1, args.max_episode_steps, 1)

    for epoch in range(args.update_epochs):
        agent.init_state(obs_batch.shape[0])
        _, newlogprob, entropy = agent.get_action_and_log_prob(
            obs_batch,
            action=action_batch,
            training=True,
        )

        logratio = (
            newlogprob.sum(-1).reshape(-1, args.max_episode_steps, 1)
            - old_logprob_batch
        )
        ratio = torch.exp(logratio)

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        if args.norm_adv:
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                advantage_batch.std() + 1e-8
            )

        pg_loss1 = -advantage_batch * ratio
        pg_loss2 = -advantage_batch * torch.clamp(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalue = agent.get_value(
            obs_batch,
            training=True,
        )

        if args.clip_vloss:
            v_loss_unclipped = (newvalue - return_batch) ** 2
            v_clipped = value_batch + torch.clamp(
                newvalue - value_batch,
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - return_batch) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - return_batch) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                print("BREAK KL")
                break

        y_pred, y_true = (
            value_batch.cpu().numpy(),
            return_batch.cpu().numpy(),
        )
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        return {
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
        }


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
    if args.env_id == "ML10":
        benchmark = metaworld.ML10(seed=args.seed)
    elif args.env_id == "ML50":
        benchmark = metaworld.ML50(seed=args.seed)
    else:
        benchmark = metaworld.ML1(args.env_id, seed=args.seed)

    # env setup
    envs = make_envs(benchmark, args.seed, args.max_episode_steps, train=True)
    keys = list(benchmark.train_classes.keys())

    eval_envs = make_eval_envs(
        benchmark, args.seed, train=args.env_id in ["MT10", "MT50"]
    )

    NUM_TASKS = len(benchmark.train_classes)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs, args, device).to(torch.float32).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    global_episodic_return = deque([], maxlen=20 * NUM_TASKS)
    global_episodic_length = deque([], maxlen=20 * NUM_TASKS)

    meta_rb = MetaRolloutBuffer(envs, args, device, NUM_TASKS)
    total_steps = 0

    for global_step in range(int(args.total_timesteps // args.n_episodes_per_trial)):
        assert args.n_episodes_per_trial > NUM_TASKS
        # collect a trial of n episodes per task
        # https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/default_worker.py
        num_trial_episodes = args.n_episodes_per_trial // NUM_TASKS
        for meta_ep in range(num_trial_episodes):
            # RL^2 stuff
            # reset hidden state for each meta trial
            agent.init_state(NUM_TASKS)
            for meta_step in range(args.max_episode_steps):
                total_steps += NUM_TASKS
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, value, log_prob = agent.step(
                        torch.tensor(obs).to(device).float().unsqueeze(1)
                    )

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminated, truncated)
                meta_rb.store(
                    obs,
                    action,
                    reward,
                    done,
                    value,
                    log_prob,
                )
                obs = next_obs

                # Only print when at least 1 env is done
                if "final_info" not in infos:
                    continue

                for i, info in enumerate(infos["final_info"]):
                    # Skip the envs that are not done
                    if info is None:
                        continue
                    global_episodic_return.append(info["episode"]["r"])
                    global_episodic_length.append(info["episode"]["l"])

            with torch.no_grad():
                value = agent.get_value(
                    torch.tensor(obs).to(device).float().unsqueeze(1)
                )
            # Collect episode batch https://github.com/rlworkgroup/garage/blob/master/src/garage/_dtypes.py#L455
            meta_rb.finish_meta_trial(
                value.cpu(), torch.tensor(done, dtype=torch.float32).unsqueeze(-1)
            )

        if global_step % 500 == 0 and global_episodic_return:
            mean_ep_return = np.mean(global_episodic_return)
            print(f"global_step={total_steps}, mean_episodic_return={mean_ep_return}")
            writer.add_scalar(
                "charts/mean_episodic_return",
                mean_ep_return,
                total_steps,
            )
            writer.add_scalar(
                "charts/mean_episodic_length",
                np.mean(global_episodic_length),
                total_steps,
            )

        logs = update_rl2_ppo(agent, meta_rb, device, total_steps, args)
        meta_rb.reset()

        if global_step % 100 == 0:
            for _key, value in logs.items():
                writer.add_scalar(_key, value, total_steps)
            print("SPS:", int(total_steps / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(total_steps / (time.time() - start_time)),
                total_steps,
            )

        if total_steps % args.eval_freq == 0 and total_steps > 0:
            print(f"Evaluating... at total_steps={total_steps}")
            agent.eval()
            eval_success_rate, eval_returns, eval_success_per_task = rl2_evaluation(
                args, agent, eval_envs, args.evaluation_num_episodes, device
            )
            eval_metrics = {
                "charts/mean_success_rate": float(eval_success_rate),
                "charts/mean_evaluation_return": float(eval_returns),
                **{
                    f"charts/{env_name}_success_rate": float(eval_success_per_task[i])
                    for i, (env_name, _) in enumerate(benchmark.test_classes.items())
                },
            }

            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, total_steps)
            print(
                f"total_steps={total_steps}, mean evaluation success rate: {eval_success_rate:.4f}"
                + f" return: {eval_returns:.4f}"
            )
            agent.train()

    envs.close()
    writer.close()
