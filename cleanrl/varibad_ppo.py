# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import sys
sys.path.append('/home/reginaldkmclean/cleanrl')

from cleanrl_utils.evals.meta_world_eval_protocol import evaluation_procedure
from cleanrl_utils.wrappers.metaworld_wrappers import OneHotV0, SyncVectorEnv
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers.vector_list_info import VectorListInfo
import metaworld

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Meta-World Benchmarking",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='reggies-phd-research',
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ML10",
        help="the id of the environment")
    parser.add_argument("--env-name", type=str, default="", help="for ML1 tests, reach/push/pick place")
    parser.add_argument("--total-timesteps", type=int, default=2e7,
        help="total timesteps of the experiments")
    parser.add_argument("--policy-learning-rate", type=float, default=0.0007,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=10,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=800,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.97,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.9,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=None,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--eval-freq", type=int, default=2,
        help="how many updates to do before evaluating the agent")
    args = parser.parse_args()
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on

    # --- VariBad Args --- #
    # --- GENERAL ---

    parser.add_argument('--max-rollouts-per-task', type=int, default=2, help='number of MDP episodes for adaptation')

    # --- POLICY ---

    # what to pass to the policy (note this is after the encoder)
    parser.add_argument('--pass-state-to-policy', type=boolean_argument, default=True, help='condition policy on state')
    parser.add_argument('--pass-latent-to-policy', type=boolean_argument, default=True,
                        help='condition policy on VAE latent')
    parser.add_argument('--pass-belief-to-policy', type=boolean_argument, default=False,
                        help='condition policy on ground-truth belief')
    parser.add_argument('--pass-task-to-policy', type=boolean_argument, default=False,
                        help='condition policy on ground-truth task description')

    # using separate encoders for the different inputs ("None" uses no encoder)
    parser.add_argument('--policy-state-embedding-dim', type=int, default=64)
    parser.add_argument('--policy-latent-embedding-dim', type=int, default=64)
    parser.add_argument('--policy-belief-embedding-dim', type=int, default=None)
    parser.add_argument('--policy-task-embedding-dim', type=int, default=None)

    # normalising (inputs/rewards/outputs)
    parser.add_argument('--norm-state-for-policy', type=boolean_argument, default=True, help='normalise state input')
    parser.add_argument('--norm-latent-for-policy', type=boolean_argument, default=True, help='normalise latent input')
    parser.add_argument('--norm-belief-for-policy', type=boolean_argument, default=True, help='normalise belief input')
    parser.add_argument('--norm-task-for-policy', type=boolean_argument, default=True, help='normalise task input')
    parser.add_argument('--norm-rew-for-policy', type=boolean_argument, default=True, help='normalise rew for RL train')
    parser.add_argument('--norm-actions-pre-sampling', type=boolean_argument, default=False,
                        help='normalise policy output')
    parser.add_argument('--norm-actions-post-sampling', type=boolean_argument, default=False,
                        help='normalise policy output')

    # network
    parser.add_argument('--policy-layers', nargs='+', default=[128, 128, 128])
    parser.add_argument('--policy-activation-function', type=str, default='tanh', help='tanh/relu/leaky-relu')
    parser.add_argument('--policy-initialisation', type=str, default='normc', help='normc/orthogonal')

    # RL algorithm
    parser.add_argument('--policy-optimiser', type=str, default='adam', help='choose: rmsprop, adam')

    # PPO specific
    parser.add_argument('--ppo-num-epochs', type=int, default=16, help='number of epochs per PPO update')
    parser.add_argument('--ppo-num-minibatch', type=int, default=4, help='number of minibatches to split the data')
    parser.add_argument('--ppo-use-huberloss', type=boolean_argument, default=True, help='use huberloss instead of MSE')
    parser.add_argument('--ppo-clip-param', type=float, default=0.1, help='clamp param')

    # other hyperparameters
    parser.add_argument('--policy-eps', type=float, default=1e-8, help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy-use-gae', type=boolean_argument, default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--policy-tau', type=float, default=0.9, help='gae parameter')
    parser.add_argument('--use-proper-time-limits', type=boolean_argument, default=True,
                        help='treat timeout and death differently (important in mujoco)')
    parser.add_argument('--policy-max-grad-norm', type=float, default=None, help='max norm of gradients')
    parser.add_argument('--encoder-max-grad-norm', type=float, default=1.0, help='max norm of gradients')
    parser.add_argument('--decoder-max-grad-norm', type=float, default=1.0, help='max norm of gradients')

    # --- VAE TRAINING ---

    # general
    parser.add_argument('--lr-vae', type=float, default=0.001)
    parser.add_argument('--size-vae-buffer', type=int, default=10000,
                        help='how many trajectories (!) to keep in VAE buffer')
    parser.add_argument('--precollect-len', type=int, default=5000,
                        help='how many frames to pre-collect before training begins (useful to fill VAE buffer)')
    parser.add_argument('--vae-buffer-add-thresh', type=float, default=1,
                        help='probability of adding a new trajectory to buffer')
    parser.add_argument('--vae-batch-num-trajs', type=int, default=15,
                        help='how many trajectories to use for VAE update')
    parser.add_argument('--tbptt-stepsize', type=int, default=None,
                        help='stepsize for truncated backpropagation through time; None uses max (horizon of BAMDP)')
    parser.add_argument('--vae-subsample-elbos', type=int, default=None,
                        help='for how many timesteps to compute the ELBO; None uses all')
    parser.add_argument('--vae-subsample-decodes', type=int, default=None,
                        help='number of reconstruction terms to subsample; None uses all')
    parser.add_argument('--vae-avg-elbo-terms', type=boolean_argument, default=False,
                        help='Average ELBO terms (instead of sum)')
    parser.add_argument('--vae-avg-reconstruction-terms', type=boolean_argument, default=False,
                        help='Average reconstruction terms (instead of sum)')
    parser.add_argument('--num-vae-updates', type=int, default=3,
                        help='how many VAE update steps to take per meta-iteration')
    parser.add_argument('--pretrain-len', type=int, default=0, help='for how many updates to pre-train the VAE')
    parser.add_argument('--kl-weight', type=float, default=1.0, help='weight for the KL term')

    parser.add_argument('--split-batches-by-task', type=boolean_argument, default=False,
                        help='split batches up by task (to save memory or if tasks are of different length)')
    parser.add_argument('--split-batches-by-elbo', type=boolean_argument, default=False,
                        help='split batches up by elbo term (to save memory of if ELBOs are of different length)')

    # - encoder
    parser.add_argument('--action-embedding-size', type=int, default=16)
    parser.add_argument('--state-embedding-size', type=int, default=32)
    parser.add_argument('--reward-embedding-size', type=int, default=16)
    parser.add_argument('--encoder-layers_before-gru', nargs='+', type=int, default=[])
    parser.add_argument('--encoder-gru_hidden-size', type=int, default=128, help='dimensionality of RNN hidden state')
    parser.add_argument('--encoder-layers-after-gru', nargs='+', type=int, default=[])
    parser.add_argument('--latent-dim', type=int, default=5, help='dimensionality of latent space')

    # - decoder: rewards
    parser.add_argument('--decode-reward', type=boolean_argument, default=True, help='use reward decoder')
    parser.add_argument('--rew-loss_-coeff', type=float, default=1.0, help='weight for state loss (vs reward loss)')
    parser.add_argument('--input-prev-state', type=boolean_argument, default=False, help='use prev state for rew pred')
    parser.add_argument('--input-action', type=boolean_argument, default=False, help='use prev action for rew pred')
    parser.add_argument('--reward-decoder-layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--multihead-for-reward', type=boolean_argument, default=False,
                        help='one head per reward pred (i.e. per state)')
    parser.add_argument('--rew-pred-type', type=str, default='deterministic',
                        help='choose: '
                             'bernoulli (predict p(r=1|s))'
                             'categorical (predict p(r=1|s) but use softmax instead of sigmoid)'
                             'deterministic (treat as regression problem)')

    # - decoder: state transitions
    parser.add_argument('--decode-state', type=boolean_argument, default=False, help='use state decoder')
    parser.add_argument('--state-loss-coeff', type=float, default=1.0, help='weight for state loss')
    parser.add_argument('--state-decoder-layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--state-pred-type', type=str, default='deterministic', help='choose: deterministic, gaussian')

    # - decoder: ground-truth task ("varibad oracle", after Humplik et al. 2019)
    parser.add_argument('--decode-task', type=boolean_argument, default=False, help='use task decoder')
    parser.add_argument('--task-loss-coeff', type=float, default=1.0, help='weight for task loss')
    parser.add_argument('--task-decoder-layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--task-pred-type', type=str, default='task_id', help='choose: task_id, task_description')

    # --- ABLATIONS ---

    # for the VAE
    parser.add_argument('--disable-decoder', type=boolean_argument, default=False,
                        help='train without decoder')
    parser.add_argument('--disable-stochasticity-in-latent', type=boolean_argument, default=False,
                        help='use auto-encoder (non-variational)')
    parser.add_argument('--disable-kl-term', type=boolean_argument, default=False,
                        help='dont use the KL regularising loss term')
    parser.add_argument('--decode-only-past', type=boolean_argument, default=False,
                        help='only decoder past observations, not the future')
    parser.add_argument('--kl-to-gauss-prior', type=boolean_argument, default=False,
                        help='KL term in ELBO to fixed Gaussian prior (instead of prev approx posterior)')

    # combining vae and RL loss
    parser.add_argument('--rlloss-through-encoder', type=boolean_argument, default=False,
                        help='backprop rl loss through encoder')
    parser.add_argument('--add-nonlinearity-to-latent', type=boolean_argument, default=False,
                        help='Use relu before feeding latent to policy')
    parser.add_argument('--vae-loss-coeff', type=float, default=1.0,
                        help='weight for VAE loss (vs RL loss)')

    # for the policy training
    parser.add_argument('--sample-embeddings', type=boolean_argument, default=False,
                        help='sample embedding for policy, instead of full belief')

    # for other things
    parser.add_argument('--disable-metalearner', type=boolean_argument, default=False,
                        help='Train feedforward policy')
    parser.add_argument('--single-task-mode', type=boolean_argument, default=False,
                        help='train policy on one (randomly chosen) environment only')

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log-interval', type=int, default=25, help='log interval, one log per n updates')
    parser.add_argument('--save-interval', type=int, default=500, help='save interval, one save per n updates')
    parser.add_argument('--save-intermediate-models', type=boolean_argument, default=False, help='save all models')
    parser.add_argument('--eval-interval', type=int, default=25, help='eval interval, one eval per n updates')
    parser.add_argument('--vis-interval', type=int, default=500, help='visualisation interval, one eval per n updates')
    parser.add_argument('--results-log-dir', default=None, help='directory to save results (None uses ./logs)')

    # general settings
    parser.add_argument('--seed', nargs='+', type=int, default=[73])
    parser.add_argument('--deterministic-execution', type=boolean_argument, default=False,
                        help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN')

    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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

    def get_value(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        latent = get_latent_for_policy(args, latent_sample, latent_mean, latent_logvar)
        return self.critic(state, belief, task, latent).detach()

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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.env_id == 'ML1':
        benchmark = metaworld.ML1(env_name=args.env_name, seed=args.seed)
    elif args.env_id == 'ML10':
        benchmark = metaworld.ML10(seed=args.seed)
    elif args.env_id == 'ML450':
        benchmark = metaworld.MT50(seed=args.seed)

    use_one_hot_wrapper = True if 'MT10' in args.env_id or 'MT50' in args.env_id else False

    # env setup
    envs = SyncVectorEnv(
        benchmark.train_classes, benchmark.train_tasks, use_one_hot_wrapper=use_one_hot_wrapper
    )
    keys = list(benchmark.train_classes.keys())

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(torch.float32).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.policy_eps)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = int(args.total_timesteps // args.batch_size)


    for update in range(1, num_updates + 1):
        if (update - 1) % args.eval_freq == 0:
            ### NEED TO SET TRAIN OR TEST TASKS
            agent = agent.to('cpu')
            agent.eval()
            evaluation_procedure(num_envs=args.num_envs, writer=writer, agent=agent,
                                 update=update, keys=keys, classes=benchmark.train_classes, tasks=benchmark.train_tasks)
            evaluation_procedure(num_envs=args.num_envs, writer=writer, agent=agent,
                                 update=update, keys=keys, classes=benchmark.test_classes, tasks=benchmark.test_tasks)
            agent = agent.to(device)
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
                action, logprob, _, value = agent.get_action_and_value(args, next_obs, belief, task, deterministic,
                                                                       latent_sample, latent_mean, latent_logvar)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            [next_obs, belief, task], (reward_raw, reward_normalized), terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            with torch.no_grad():
                # update embeddings
                latent_sample, latent_mean, latent_logvar, hidden_state = update_encoding(vae_encoder, next_obs,
                                                                            action, reward_raw, done, hidden_state)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                #print(i, info)
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        if args.precollect_len <= global_step:
            # update policy and vae
            if update >= args.pretrain_len and update > 0:


                # bootstrap value if not done
                with torch.no_grad():
                    next_value = agent.get_value(next_obs, belief, task, latent_sample, latent_mean, latent_logvar).reshape(1, -1)
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

                update_vae(next_obs, belief, task, latent_sample, latent_mean, latent_logvar)

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
                        if args.max_grad_norm:
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

    envs.close()
    writer.close()
