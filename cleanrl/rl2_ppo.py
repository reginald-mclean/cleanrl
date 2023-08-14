# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
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
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Optional, Type, Tuple
from functools import partial
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument("--wandb-project-name", type=str, default="Meta-World Benchmarking",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # RL^2 arguments
    parser.add_argument("--num-meta-episodes", type=int, default=10,
        help="maximum number of meta episodes per batch")
    parser.add_argument("--recurrent-state-size", type=int, default=128,
        help="")
    parser.add_argument("--meta-episode-length", type=int, default=500,
        help="TODO")

    parser.add_argument("--use-gae", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        )

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

    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GRU(nn.Module):
    def __init__(self, input_size: int, state_size: int):
        super().__init__()

        self._gru = nn.GRU(input_size, state_size)

        for name, param in self._gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)


    def forward(
        self,
        x: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(0) == recurrent_states.size(0):
            if recurrent_state_masks is None:
                recurrent_state_masks = torch.ones(recurrent_states.shape)

            x = x.to(device)
            recurrent_states = recurrent_states.to(device)
            recurrent_state_masks = recurrent_state_masks.to(device)

            x, recurrent_states = self._gru(
                x.unsqueeze(0), (recurrent_states * recurrent_state_masks).unsqueeze(0)
            )

            x = x.squeeze(0)
            recurrent_states = recurrent_states.squeeze(0)

            return x, recurrent_states

        # x is a (T, N, -1) batch from the sampler that has been flattend to (T * N, -1)
        N = recurrent_states.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        # masks
        if recurrent_state_masks is None:
            recurrent_state_masks = torch.ones((T, N))
        else:
            recurrent_state_masks = recurrent_state_masks.view(T, N)

        has_zeros = (
            (recurrent_state_masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
        )

        # +1 to correct the recurrent_masks[1:] where zeros are present.
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [T]

        recurrent_state_masks.to(device)
        recurrent_states = recurrent_states.unsqueeze(0)
        outputs = []

        x = x.to(device)
        recurrent_states = recurrent_states.to(device)
        recurrent_state_masks = recurrent_state_masks.to(device)

        for i in range(len(has_zeros) - 1):
            # We can now process steps that don't have any zeros in done_masks together!
            # This is much faster
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, recurrent_states = self._gru(
                x[start_idx:end_idx],
                recurrent_states * recurrent_state_masks[start_idx].view(1, -1, 1),
            )

            outputs.append(rnn_scores)
            pass

        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)

        # flatten
        x = x.view(T * N, -1)
        recurrent_states = recurrent_states.squeeze(0)
        pass

        return x, recurrent_states

class GRUActor(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        # self.gru = nn.GRU(obs_shape, args.recurrent_state_size, batch_first=True)
        self.gru = GRU(obs_shape, args.recurrent_state_size)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(args.recurrent_state_size, 512)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512), std=0.01),
            nn.Tanh(),
            layer_init(
                nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def forward(self, x: torch.Tensor, recurrent_state: torch.Tensor, action=None):
        x, recurrent_state = self.gru(x, recurrent_state, device=x.device)
        action_mean = self.actor_mean(x)
        if x.size()[0] == 10:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return {
                "action":action,
                "logprob": probs.log_prob(action).sum(1),
                "entropy": probs.entropy().sum(1),
                "actor_state": recurrent_state
                }

class GRUCritic(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        # self.gru = nn.GRU(obs_shape, args.recurrent_state_size, batch_first=True)
        self.gru = GRU(obs_shape, args.recurrent_state_size)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(args.recurrent_state_size, 512)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512), std=1.0),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
    def forward(self, x: torch.Tensor, recurrent_state: torch.Tensor):
        x, recurrent_state = self.gru(x, recurrent_state, device=x.device)
        return {"value": self.critic(x), "critic_state": recurrent_state}


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.actor = GRUActor(envs, args)
        self.critic = GRUCritic(envs, args)

    def get_value(self, x, state_critic):
        return self.critic(x, state_critic)["value"]

    def get_action_and_value(self, x, state_actor, state_critic, action=None):
        result = {}
        result.update(self.actor(x, state_actor, action))
        result.update(self.critic(x, state_critic))
        return result


def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    max_episode_steps: Optional[int] = None,
    terminate_on_success: bool = False,
) -> gym.vector.VectorEnv:
    def init_each_env(env_cls: Type[SawyerXYZEnv], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env = metaworld_wrappers.RL2Env(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        env = metaworld_wrappers.RandomTaskSelectWrapper(env, tasks)
        env.action_space.seed(seed)
        return env

    return gym.vector.SyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id)
            for env_id, (name, env_cls) in enumerate(benchmark.train_classes.items())
        ]
    )


make_envs = partial(_make_envs_common, terminate_on_success=False)


class MetaEpisode:
    def __init__(self, args):
        self.step = 1
        self.meta_episode_length = args.meta_episode_length

        self.obs = torch.zeros(
            args.meta_episode_length + 1,
            args.num_meta_episodes,
            *envs.single_observation_space.shape
        )
        self.actions = torch.zeros(
            args.meta_episode_length,
            args.num_meta_episodes,
            *envs.single_action_space.shape
        )
        self.rewards = torch.zeros(args.meta_episode_length, args.num_meta_episodes, 1)
        self.values = torch.zeros(args.meta_episode_length + 1, args.num_meta_episodes, 1)

        self.logprobs = torch.zeros(args.meta_episode_length, args.num_meta_episodes, *envs.single_action_space.shape)
        self.dones = torch.zeros(args.meta_episode_length + 1, args.num_meta_episodes, 1)

        self.states_actor = torch.zeros(
            args.meta_episode_length + 1,
            args.num_meta_episodes,
            args.recurrent_state_size,
        )
        self.states_critic = torch.zeros(
            args.meta_episode_length + 1,
            args.num_meta_episodes,
            args.recurrent_state_size,
        )

        self.returns = torch.zeros(
            args.meta_episode_length + 1, args.num_meta_episodes, 1
        )

    def to_device(self, device: torch.device) -> None:
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.logprobs = self.logprobs.to(device)
        self.dones = self.dones.to(device)
        self.states_actor = self.states_actor.to(device)
        self.states_critic = self.states_critic.to(device)
        self.returns = self.returns.to(device)

    def add(
        self,
        obs,
        states_actor,
        states_critic,
        actions,
        logprobs,
        values,
        rewards,
        dones,
    ):
        if self.step > self.meta_episode_length:
            raise IndexError("Step limit reached")

        self.obs[self.step + 1].copy_(torch.from_numpy(obs))
        self.dones[self.step + 1].copy_(torch.from_numpy(done[:, None]))
        self.actions[self.step].copy_(actions)
        self.logprobs[self.step].copy_(logprobs)
        self.values[self.step].copy_(values[:, None])
        self.rewards[self.step].copy_(torch.from_numpy(rewards[:, None]))

        self.states_actor[self.step + 1].copy_(states_actor)
        self.states_critic[self.step + 1].copy_(states_critic)

        self.step = self.step + 1

    def compute_returns(
        self, next_value: torch.Tensor, use_gae: bool, gamma: float, gae_lambda: float
    ) -> None:
        if use_gae:
            self.values[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.values[step + 1] * (1 - self.dones[step + 1])
                    - self.values[step]
                )
                gae = delta + gamma * gae_lambda * (1 - self.dones[step + 1]) * gae
                self.returns[step] = gae + self.values[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * (1 - self.dones[step + 1])
                    + self.rewards[step]
                )

class MetaBatchSampler:
    def __init__(self, batches, device: torch.device):
        self.meta_episode_batches = batches
        self.device = device

        self.obs = self._concat_attr("obs")
        self.rewards = self._concat_attr("rewards")
        self.values = self._concat_attr("values")
        self.returns = self._concat_attr("returns")
        self.logprobs = self._concat_attr("logprobs")
        self.actions = self._concat_attr("actions")

        self.states_actor = self._concat_attr("states_actor")
        self.states_critic = self._concat_attr("states_critic")

        self.dones = self._concat_attr("dones")

    def _concat_attr(self, attr: str) -> torch.Tensor:
        """
        Conacatenate attribute values.

        Args:
          attr (str): Attribute whose values to concatenate.

        Returns:
          torch.Tensor
        """
        tensors = [
            getattr(meta_episode_batch, attr)
            for meta_episode_batch in self.meta_episode_batches
        ]

        return torch.cat(tensors=tensors, dim=1).to(self.device)

    def sample(self, advantages: torch.Tensor, num_minibatches: int):
        meta_episode_length = self.rewards.shape[0]
        num_meta_episodes = self.rewards.shape[1]

        num_envs_per_batch = num_meta_episodes // num_minibatches
        perm = torch.randperm(num_meta_episodes)

        for start_ind in range(0, num_meta_episodes, num_envs_per_batch):
            obs_batch = []
            actions_batch = []
            values_batch = []
            return_batch = []
            dones_batch = []
            old_logprobs_batch = []
            adv_targ = []

            states_actor_batch = []
            states_critic_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                obs_batch.append(self.obs[:-1, ind])
                actions_batch.append(self.actions[:, ind])
                values_batch.append(self.values[:-1, ind])
                return_batch.append(self.returns[:-1, ind])

                dones_batch.append(self.dones[:-1, ind])
                old_logprobs_batch.append(self.logprobs[:, ind])
                adv_targ.append(advantages[:, ind])

                states_actor_batch.append(
                    self.states_actor[0:1, ind]
                )
                states_critic_batch.append(
                    self.states_critic[0:1, ind]
                )

            T, N = meta_episode_length, num_envs_per_batch
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            values_batch = torch.stack(values_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            dones_batch = torch.stack(dones_batch, 1)
            old_logprobs_batch = torch.stack(old_logprobs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # recurrent states
            states_actor_batch = torch.stack(
                states_actor_batch, 1
            ).view(N, -1)

            states_critic_batch = torch.stack(
                states_critic_batch, 1
            ).view(N, -1)

            # flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            values_batch = _flatten(T, N, values_batch)
            return_batch = _flatten(T, N, return_batch)
            dones_batch = _flatten(T, N, dones_batch)
            old_logprobs_batch = _flatten(T, N, old_logprobs_batch)

            adv_targ = _flatten(T, N, adv_targ)

        yield obs_batch, states_actor_batch, states_critic_batch, actions_batch, values_batch, return_batch, dones_batch, old_logprobs_batch, adv_targ

def _flatten(T: int, N: int, _tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten a given tensor containing rollout information.

    Args:
        T (int): Corresponds to the number of steps in the rollout.
        N (int): Number of processes running.
        _tensor (torch.Tensor): Tensor to flatten.

    Returns:
        torch.Tensor
    """
    return _tensor.view(T * N, *_tensor.size()[2:])

if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
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
    if args.env_id == "MT10":
        benchmark = metaworld.MT10(seed=args.seed)
    elif args.env_id == "MT50":
        benchmark = metaworld.MT50(seed=args.seed)

    # env setup
    envs = make_envs(benchmark, args.seed, args.meta_episode_length)
    keys = list(benchmark.train_classes.keys())

    NUM_TASKS = len(benchmark.train_classes)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs, args).to(torch.float32).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    num_updates = int(args.total_timesteps // args.batch_size)
    meta_episode_batch = list()

    global_episodic_return = deque([], maxlen=20 * NUM_TASKS)
    global_episodic_length = deque([], maxlen=20 * NUM_TASKS)

    for update in range(1, num_updates + 1):
        # if (update - 1) % args.eval_freq == 0:
        ### NEED TO SET TRAIN OR TEST TASKS
        # agent = agent.to("cpu")
        # agent.eval()
        # evaluation_procedure(
        #     num_envs=args.num_envs,
        #     writer=writer,
        #     agent=agent,
        #     update=update,
        #     keys=keys,
        #     classes=benchmark.train_classes,
        #     tasks=benchmark.train_tasks,
        #     device=device
        # )
        # agent = agent.to(device)
        # agent.train()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow


        meta_episodes = MetaEpisode(args)
        meta_episodes.obs[0].copy_(torch.from_numpy(next_obs))

        for meta_step in range(args.meta_episode_length - 1):
            global_step += 1 * args.num_envs
            # ALGO LOGIC: action logic
            with torch.no_grad():
                agent_dict = agent.get_action_and_value(meta_episodes.obs[meta_step].to(device),
                                    meta_episodes.states_actor[meta_step].to(device),
                                    meta_episodes.states_critic[meta_step].to(device))
                action = agent_dict["action"]
                logprob = agent_dict["logprob"].unsqueeze(-1)
                value = agent_dict["value"].unsqueeze(-1)
                actor_state = agent_dict["actor_state"]
                critic_state = agent_dict["critic_state"]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            meta_episodes.add(next_obs,
                                actor_state,
                                critic_state,
                                action, logprob, value.flatten(), reward, done)

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
            next_value = agent.get_value(
                    meta_episodes.obs[-1].to(device),
                    meta_episodes.states_critic[-1].to(device))
        meta_episodes.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
        meta_episode_batch.append(meta_episodes)

        if len(meta_episode_batch) == 32:

            clipfracs = []
            for epoch in range(args.update_epochs):

                minibatch_sampler = MetaBatchSampler(meta_episode_batch, device)
                advantages = minibatch_sampler.returns[:-1] - minibatch_sampler.values[:-1]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                minibatches = minibatch_sampler.sample(advantages, 4)

                for sample in minibatches:
                    (
                    obs_batch,
                    states_actor_batch,
                    states_critic_batch,
                    actions_batch,
                    values_batch,
                    return_batch,
                    dones_batch,
                    old_logprobs_batch,
                    adv_targ,
                    ) = sample

                    agent_dict = agent.get_action_and_value(
                            obs_batch, states_actor_batch, states_critic_batch)
                    newlogprob = agent_dict["logprob"].sum(-1)
                    newvalue = agent_dict["value"]
                    entropy = agent_dict["entropy"]
                    logratio = newlogprob - old_logprobs_batch
                    ratio = torch.exp(logratio)

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    pg_loss1 = - adv_targ * ratio
                    pg_loss2 = (
                        -adv_targ  * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)

                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - values_batch) ** 2
                        v_clipped = values_batch + torch.clamp(
                            newvalue - values_batch,
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
                            break

                    y_pred, y_true = values_batch.cpu().numpy(), return_batch.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                    # TRY NOT TO MODIFY: record rewards for plotting purposes
                    writer.add_scalar(
                        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
                    )
                    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                    writer.add_scalar("losses/explained_variance", explained_var, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS", int(global_step / (time.time() - start_time)), global_step
                    )

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
        global_step +=1

    envs.close()
    writer.close()
