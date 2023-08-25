# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import sys
sys.path.append('/home/reginaldkmclean/cleanrl')

from cleanrl_utils.evals.meta_world_eval_protocol import evaluation_procedure
from cleanrl_utils.wrappers.metaworld_wrappers import SyncVectorEnv
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument("--num-steps", type=int, default=200,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.97,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.9,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
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

    # fmt: on

    # --- VariBad Args --- #
    # --- GENERAL ---

    parser.add_argument('--max-rollouts-per-task', type=int, default=2, help='number of MDP episodes for adaptation')

    # --- POLICY ---

    # what to pass to the policy (note this is after the encoder)
    parser.add_argument('--pass-state-to-policy', type=bool, default=True, help='condition policy on state')
    parser.add_argument('--pass-latent-to-policy', type=bool, default=True,
                        help='condition policy on VAE latent')
    parser.add_argument('--pass-belief-to-policy', type=bool, default=False,
                        help='condition policy on ground-truth belief')
    parser.add_argument('--pass-task-to-policy', type=bool, default=False,
                        help='condition policy on ground-truth task description')

    # using separate encoders for the different inputs ("None" uses no encoder)
    parser.add_argument('--policy-state-embedding-dim', type=int, default=64)
    parser.add_argument('--policy-latent-embedding-dim', type=int, default=64)
    parser.add_argument('--policy-belief-embedding-dim', type=int, default=None)
    parser.add_argument('--policy-task-embedding-dim', type=int, default=None)

    # normalising (inputs/rewards/outputs)
    parser.add_argument('--norm-state-for-policy', type=bool, default=True, help='normalise state input')
    parser.add_argument('--norm-latent-for-policy', type=bool, default=True, help='normalise latent input')
    parser.add_argument('--norm-belief-for-policy', type=bool, default=True, help='normalise belief input')
    parser.add_argument('--norm-task-for-policy', type=bool, default=True, help='normalise task input')
    parser.add_argument('--norm-rew-for-policy', type=bool, default=True, help='normalise rew for RL train')
    parser.add_argument('--norm-actions-pre-sampling', type=bool, default=False,
                        help='normalise policy output')
    parser.add_argument('--norm-actions-post-sampling', type=bool, default=False,
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
    parser.add_argument('--ppo-use-huberloss', type=bool, default=True, help='use huberloss instead of MSE')
    parser.add_argument('--ppo-clip-param', type=float, default=0.1, help='clamp param')

    # other hyperparameters
    parser.add_argument('--policy-eps', type=float, default=1e-8, help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy-use-gae', type=bool, default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--policy_init_std', type=float, default=1.0, help='only used for continuous actions')
    parser.add_argument('--policy-tau', type=float, default=0.9, help='gae parameter')
    parser.add_argument('--use-proper-time-limits', type=bool, default=True,
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
    parser.add_argument('--vae-avg-elbo-terms', type=bool, default=False,
                        help='Average ELBO terms (instead of sum)')
    parser.add_argument('--vae-avg-reconstruction-terms', type=bool, default=False,
                        help='Average reconstruction terms (instead of sum)')
    parser.add_argument('--num-vae-updates', type=int, default=3,
                        help='how many VAE update steps to take per meta-iteration')
    parser.add_argument('--pretrain-len', type=int, default=0, help='for how many updates to pre-train the VAE')
    parser.add_argument('--kl-weight', type=float, default=1.0, help='weight for the KL term')

    parser.add_argument('--split-batches-by-task', type=bool, default=False,
                        help='split batches up by task (to save memory or if tasks are of different length)')
    parser.add_argument('--split-batches-by-elbo', type=bool, default=False,
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
    parser.add_argument('--decode-reward', type=bool, default=True, help='use reward decoder')
    parser.add_argument('--rew-loss-coeff', type=float, default=1.0, help='weight for state loss (vs reward loss)')
    parser.add_argument('--input-prev-state', type=bool, default=False, help='use prev state for rew pred')
    parser.add_argument('--input-action', type=bool, default=False, help='use prev action for rew pred')
    parser.add_argument('--reward-decoder-layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--multihead-for-reward', type=bool, default=False,
                        help='one head per reward pred (i.e. per state)')
    parser.add_argument('--rew-pred-type', type=str, default='deterministic',
                        help='choose: '
                             'bernoulli (predict p(r=1|s))'
                             'categorical (predict p(r=1|s) but use softmax instead of sigmoid)'
                             'deterministic (treat as regression problem)')

    # - decoder: state transitions
    parser.add_argument('--decode-state', type=bool, default=False, help='use state decoder')
    parser.add_argument('--state-loss-coeff', type=float, default=1.0, help='weight for state loss')
    parser.add_argument('--state-decoder-layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--state-pred-type', type=str, default='deterministic', help='choose: deterministic, gaussian')

    # - decoder: ground-truth task ("varibad oracle", after Humplik et al. 2019)
    parser.add_argument('--decode-task', type=bool, default=False, help='use task decoder')
    parser.add_argument('--task-loss-coeff', type=float, default=1.0, help='weight for task loss')
    parser.add_argument('--task-decoder-layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--task-pred-type', type=str, default='task_id', help='choose: task_id, task_description')

    # --- ABLATIONS ---

    # for the VAE
    parser.add_argument('--disable-decoder', type=bool, default=False,
                        help='train without decoder')
    parser.add_argument('--disable-stochasticity-in-latent', type=bool, default=False,
                        help='use auto-encoder (non-variational)')
    parser.add_argument('--disable-kl-term', type=bool, default=False,
                        help='dont use the KL regularising loss term')
    parser.add_argument('--decode-only-past', type=bool, default=False,
                        help='only decoder past observations, not the future')
    parser.add_argument('--kl-to-gauss-prior', type=bool, default=False,
                        help='KL term in ELBO to fixed Gaussian prior (instead of prev approx posterior)')

    # combining vae and RL loss
    parser.add_argument('--rlloss-through-encoder', type=bool, default=False,
                        help='backprop rl loss through encoder')
    parser.add_argument('--add-nonlinearity-to-latent', type=bool, default=False,
                        help='Use relu before feeding latent to policy')
    parser.add_argument('--vae-loss-coeff', type=float, default=1.0,
                        help='weight for VAE loss (vs RL loss)')

    # for the policy training
    parser.add_argument('--sample-embeddings', type=bool, default=False,
                        help='sample embedding for policy, instead of full belief')

    # for other things
    parser.add_argument('--disable-metalearner', type=bool, default=False,
                        help='Train feedforward policy')
    parser.add_argument('--single-task-mode', type=bool, default=False,
                        help='train policy on one (randomly chosen) environment only')

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log-interval', type=int, default=25, help='log interval, one log per n updates')
    parser.add_argument('--save-interval', type=int, default=500, help='save interval, one save per n updates')
    parser.add_argument('--save-intermediate-models', type=bool, default=False, help='save all models')
    parser.add_argument('--eval-interval', type=int, default=25, help='eval interval, one eval per n updates')
    parser.add_argument('--vis-interval', type=int, default=500, help='visualisation interval, one eval per n updates')
    parser.add_argument('--results-log-dir', default=None, help='directory to save results (None uses ./logs)')

    # general settings
    parser.add_argument('--deterministic-execution', type=bool, default=False,
                        help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN')
    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def squash_action(action, args):
    if args.norm_actions_post_sampling:
        return torch.tanh(action)
    else:
        return action

def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std, norm_actions_pre_sampling):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.fc_mean = self.fc_mean.double()
        self.logstd = nn.Parameter(np.log(torch.zeros(num_outputs) + init_std)).to('cuda:0').double()
        self.norm_actions_pre_sampling = norm_actions_pre_sampling
        self.min_std = torch.tensor([1e-6]).to(device)
        self.fixedNormal = torch.distributions.Normal
        log_prob_normal = self.fixedNormal.log_prob
        self.fixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
    def forward(self, x):
        self.fc_mean = self.fc_mean.double()
        action_mean = self.fc_mean(x.double())
        if self.norm_actions_pre_sampling:
            action_mean = torch.tanh(action_mean)
        std = torch.max(self.min_std, self.logstd.exp())
        dist = self.fixedNormal(action_mean, std)

        return dist

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).double().to(device)
        self.var = torch.ones(shape).double().to(device)
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_latent_for_policy(args, latent_sample=None, latent_mean=None, latent_logvar=None):
    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None
    if args.add_nonlinearity_to_latent:
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)
    latent = latent_sample
    if latent.shape[0] == 1:
        latent = latent.squeeze(0)
    return latent

class Agent(nn.Module):
    def __init__(self, envs, args, activation_function, hidden_layers, init_std,
                 pass_state_to_policy, pass_latent_to_policy, pass_task_to_policy, pass_belief_to_policy):
        super().__init__()
        self.args = args
        dim_task = 0
        dim_state = args.state_dim
        dim_belief = 0
        dim_latent = args.latent_dim
        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError
        if args.policy_initialisation == 'normc':
            init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain(activation_function))
        else:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain(activation_function))

        self.pass_state_to_policy = pass_state_to_policy
        self.pass_latent_to_policy = pass_latent_to_policy
        self.pass_task_to_policy = pass_task_to_policy
        self.pass_belief_to_policy = pass_belief_to_policy

        # set normalisation parameters for the inputs
        # (will be updated from outside using the RL batches)
        self.norm_state = self.args.norm_state_for_policy and (dim_state is not None)
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms = RunningMeanStd(shape=(dim_state))
        self.norm_latent = self.args.norm_latent_for_policy and (dim_latent is not None)
        if self.pass_latent_to_policy and self.norm_latent:
            self.latent_rms = RunningMeanStd(shape=(dim_latent))
        self.norm_belief = self.args.norm_belief_for_policy and (dim_belief is not None)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms = RunningMeanStd(shape=(dim_belief))
        self.norm_task = self.args.norm_task_for_policy and (dim_task is not None)
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms = RunningMeanStd(shape=(dim_task))

        curr_input_dim = dim_state * int(self.pass_state_to_policy) + \
                         dim_latent * int(self.pass_latent_to_policy) + \
                         dim_belief * int(self.pass_belief_to_policy) + \
                         dim_task * int(self.pass_task_to_policy)
        # initialise encoders for separate inputs
        self.use_state_encoder = self.args.policy_state_embedding_dim is not None
        if self.pass_state_to_policy and self.use_state_encoder:
            self.state_encoder = FeatureExtractor(dim_state, self.args.policy_state_embedding_dim,
                                                      self.activation_function).double().to(device)
            curr_input_dim = curr_input_dim - dim_state + self.args.policy_state_embedding_dim
        self.use_latent_encoder = self.args.policy_latent_embedding_dim is not None
        if self.pass_latent_to_policy and self.use_latent_encoder:
            self.latent_encoder = FeatureExtractor(dim_latent, self.args.policy_latent_embedding_dim,
                                                       self.activation_function).double().to(device)
            curr_input_dim = curr_input_dim - dim_latent + self.args.policy_latent_embedding_dim
        self.use_belief_encoder = self.args.policy_belief_embedding_dim is not None
        if self.pass_belief_to_policy and self.use_belief_encoder:
            self.belief_encoder = FeatureExtractor(dim_belief, self.args.policy_belief_embedding_dim,
                                                       self.activation_function).double().to(device)
            curr_input_dim = curr_input_dim - dim_belief + self.args.policy_belief_embedding_dim
        self.use_task_encoder = self.args.policy_task_embedding_dim is not None
        if self.pass_task_to_policy and self.use_task_encoder:
            self.task_encoder = FeatureExtractor(dim_task, self.args.policy_task_embedding_dim,
                                                     self.activation_function).double().to(device)
            curr_input_dim = curr_input_dim - dim_task + self.args.policy_task_embedding_dim

        self.critic = nn.ModuleList([]).double()
        self.actor = nn.ModuleList([]).double()
        for i in range(len(hidden_layers)):
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]).to(device).double())
            self.actor.append(fc.double())
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]).to(device).double())
            self.critic.append(fc.double())
            curr_input_dim = hidden_layers[i]
        self.critic_linear = nn.Linear(hidden_layers[-1], 1).to(device)
        self.critic_linear = self.critic_linear.double()
        num_outputs = envs.single_action_space.shape[0]
        self.dist = DiagGaussian(hidden_layers[-1], num_outputs, init_std, args.norm_actions_pre_sampling)

    def get_actor_params(self):
        return [*self.actor.parameters(), *self.dist.parameters()]

    def get_critic_params(self):
        return [*self.critic.parameters(), *self.critic_linear.parameters()]

    def forward_actor(self, inputs):
        h = inputs
        for i in range(len(self.actor)):
            self.actor[i] = self.actor[i].double()
            h = self.actor[i](h)
            h = self.activation_function(h)
        return h

    def forward_critic(self, inputs):
        h = inputs.double()
        for i in range(len(self.critic)):
            self.critic[i] = self.critic[i].double()
            h = self.critic[i](h)
            h = self.activation_function(h)
        return h

    def forward(self, state, latent, belief, task):
        device = 'cuda:0'
        # handle inputs (normalise + embed)

        if self.pass_state_to_policy:
            if self.norm_state:
                state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
            if self.use_state_encoder:
                self.state_encoder = self.state_encoder.double()
                state = self.state_encoder(state)

        else:
            state = torch.zeros(0, ).to(device)
        if self.pass_latent_to_policy:
            if self.norm_latent:
                latent = (latent - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
            if self.use_latent_encoder:
                latent = self.latent_encoder(latent)
        else:
            latent = torch.zeros(0, ).to(device)
        if self.pass_belief_to_policy:
            if self.norm_belief:
                belief = (belief - self.belief_rms.mean) / torch.sqrt(self.belief_rms.var + 1e-8)
            if self.use_belief_encoder:
                belief = self.belief_encoder(belief.float())
        else:
            belief = torch.zeros(0, ).to(device)
        if self.pass_task_to_policy:
            if self.norm_task:
                task = (task - self.task_rms.mean) / torch.sqrt(self.task_rms.var + 1e-8)
            if self.use_task_encoder:
                task = self.task_encoder(task.float())
        else:
            task = torch.zeros(0, ).to(device)
        # concatenate inputs
        inputs = torch.cat((torch.squeeze(state), torch.squeeze(latent), belief, task), dim=-1)
        # forward through critic/actor part
        hidden_critic = self.forward_critic(inputs)
        hidden_actor = self.forward_actor(inputs)
        self.critic_linear = self.critic_linear.double()
        return self.critic_linear(hidden_critic), hidden_actor

    def act(self, state, latent, belief, task, deterministic=False):
        """
        Returns the (raw) actions and their value.
        """
        value, actor_features = self.forward(state=state.double().to('cuda:0'), latent=latent.double().to('cuda:0'), belief=belief, task=task)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        # action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
        return value, action, dist.log_prob(action).sum(dim=1), dist.entropy().sum(1)

    def get_value(self, state, latent, belief, task):
        value, _ = self.forward(state, latent, belief, task)
        return value

    def get_action_and_value(self, x, action=None):
        action_mean = self.forward_actor(x)
        if x.size()[0] == 10:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def update_rms(self, args, policy_storage):
        """ Update normalisation parameters for inputs with current data """
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms.update(policy_storage.prev_state[:-1])
        if self.pass_latent_to_policy and self.norm_latent:
            latent = get_latent_for_policy(args,
                                               torch.cat(policy_storage.latent_samples[:-1]),
                                               torch.cat(policy_storage.latent_mean[:-1]),
                                               torch.cat(policy_storage.latent_logvar[:-1])
                                               )
            self.latent_rms.update(latent)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms.update(policy_storage.beliefs[:-1])
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms.update(policy_storage.tasks[:-1])

    def evaluate_actions(self, state, latent, belief, task, action):
        value, actor_features = self.forward(state, latent, belief, task)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def turn_off_grads(self):
        super().eval()
        nets = [self.actor, self.critic, self.critic_linear]
        if self.pass_state_to_policy and self.use_state_encoder:
            self.state_encoder.eval()
        if self.pass_latent_to_policy and self.use_latent_encoder:
            self.latent_encoder.eval()
        if self.pass_belief_to_policy and self.use_belief_encoder:
            self.belief_encoder.eval()
        if self.pass_task_to_policy and self.use_task_encoder:
            self.task_encoder.eval()
        for net in nets:
            net.eval()
        self.dist.fc_mean.eval()



class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """
    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size).to(torch.float64).to(device)
            self.fc = self.fc.double()
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            self.fc = self.fc.double()
            return self.activation_function(self.fc(inputs.double()))
        else:
            return torch.zeros(0, ).to(device)


class RNNEncoder(nn.Module):
    def __init__(self, args,
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 ):
        super(RNNEncoder, self).__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterize = self._sample_gaussian
        self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu).double().to(device)
        self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu).double().to(device)
        self.reward_encoder = FeatureExtractor(reward_size, reward_embed_size, F.relu).double().to(device)

        # adding FC layers before GRU
        current_input_dim = action_embed_dim + reward_embed_size + state_embed_dim
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(current_input_dim, layers_before_gru[i]).double().to(device))
            current_input_dim = layers_before_gru[i]

        self.gru = nn.GRU(input_size=current_input_dim, hidden_size=hidden_size, num_layers=1).to(device).double()

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        current_output_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(current_output_dim, layers_after_gru[i]))
            current_output_dim = layers_after_gru[i]

        # output layers
        self.fc_mu = nn.Linear(current_output_dim, latent_dim).double()
        self.fc_logvar = nn.Linear(current_output_dim, latent_dim).double()

    def _sample_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done.long())
        return hidden_state

    def prior(self, batch_size, sample=True):
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to('cuda:0')
        h = hidden_state
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        latent_mean = self.fc_mu(h.double())
        latent_logvar = self.fc_logvar(h.double())
        if sample:
            latent_sample = self.reparameterize(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True, detach_every=None):
        # actions = squash actions
        actions = squash_action(actions, self.args)
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        if return_prior:
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(actions.shape[1])
            hidden_state = prior_hidden_state.clone()

        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2).to(device)
        hidden_state = hidden_state.to(device)
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_grup[i](h))

        if detach_every is None:
            if len(h.size()) != len(hidden_state.size()):
                hidden_state = hidden_state.unsqueeze(0).to(device)

            output, _ = self.gru(h.double(), hidden_state.double())
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                current_input = h[i*detach_every:i*detach_every+detach_every]
                current_output, hidden_state = self.gru(current_input, hidden_state)
                output.append(current_output)
                hidden_state = hidden_state.detach()
            output = torch.cat(output, dim=0)
        gru_h = output.clone()

        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)

        if sample:
            latent_sample = self.reparameterize(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output = torch.cat((prior_hidden_state, output))
        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]
        return latent_sample, latent_mean, latent_logvar, output

    def turn_off_grads(self):
        super().eval()
        nets = [self.state_encoder, self.action_encoder, self.reward_encoder, self.fc_mu,
                self.fc_logvar, self.fc_after_gru, self.fc_before_gru, self.gru]
        for net in nets:
            net.eval()

    def turn_on_grads(self):
        super().train()
        nets = [self.state_encoder, self.action_encoder, self.reward_encoder, self.fc_mu,
                self.fc_logvar, self.fc_after_gru, self.fc_before_gru, self.gru]
        for net in nets:
            net.train()

class RewardDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 num_states=0,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=True,
                 input_action=True):
        super(RewardDecoder, self).__init__()
        self.args = args

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        if self.multi_head:
            current_input_dim = latent_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(current_input_dim, layers[i]))
                current_input_dim = layers[i]
            self.fc_out = nn.Linear(current_input_dim, num_states)
        else:
            self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu).to(device)
            if self.input_action:
                self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu).to(device)
            else:
                self.action_encoder = None
            current_input_dim = latent_dim + state_embed_dim
            if input_prev_state:
                current_input_dim += state_embed_dim
            if input_action:
                current_input_dim += action_embed_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(current_input_dim, layers[i]).to(device))
                current_input_dim = layers[i]
            if pred_type == 'gaussian':
                self.fc_out = nn.Linear(current_input_dim, 2).to(device)
            else:
                self.fc_out = nn.Linear(current_input_dim, 1).to(device)

    def forward(self, latent_state, next_state, prev_state=None, actions=None):
        if actions is not None:
            actions = squash_action(actions, args)
        if self.multi_head:
            h = latent_state.clone()
        else:
            hns = self.state_encoder(next_state)
            h = torch.cat((latent_state, hns), dim=-1)
            if self.input_action:
                ha = self.action_encoder(actions)
                h = torch.cat((h, ha), dim=-1)
            if self.input_prev_state:
                hps = self.state_encoder(prev_state)
                h = torch.cat((h, hps), dim=-1)
        for i in range(len(self.fc_layers)):
            self.fc_layers[i] = self.fc_layers[i].double()
            h = F.relu(self.fc_layers[i](h.double()))
        self.fc_out = self.fc_out.double()
        return self.fc_out(h)


class RolloutStorageVAE(object):
    def __init__(self, num_envs, max_trajectory_length, zero_pad, max_num_rollouts, state_dim, action_dim,
                 vae_buffer_add_thresh, task_dim):
        self.obs_dim = state_dim
        self.act_dim = action_dim
        self.task_dim = task_dim

        self.vae_buffer_add_thresh = vae_buffer_add_thresh
        self.max_buffer_size = max_num_rollouts
        self.insert_idx = 0
        self.buffer_len = 0

        self.max_traj_length = max_trajectory_length
        self.zero_pad = zero_pad

        if self.max_buffer_size > 0:
            self.prev_state = torch.zeros((self.max_traj_length, self.max_buffer_size, state_dim))
            self.next_state = torch.zeros((self.max_traj_length, self.max_buffer_size, state_dim))
            self.actions = torch.zeros((self.max_traj_length, self.max_buffer_size, action_dim))
            self.rewards = torch.zeros((self.max_traj_length, self.max_buffer_size, 1))
            self.tasks = None
            self.trajectory_lens = [0] * self.max_buffer_size

        self.num_envs = num_envs
        self.curr_timstep = torch.zeros((num_envs)).long()
        self.running_prev_state = torch.zeros((self.max_traj_length, num_envs, state_dim)).to(device)
        self.running_next_state = torch.zeros((self.max_traj_length, num_envs, state_dim)).to(device)
        self.running_rewards = torch.zeros((self.max_traj_length, num_envs, 1)).to(device)
        self.running_actions = torch.zeros((self.max_traj_length, num_envs, action_dim)).to(device)
        self.running_tasks = None

    def get_running_batch(self):
        return self.running_prev_state, self.running_next_state, self.running_actions, self.running_rewards, self.curr_timstep

    def insert(self, prev_state, actions, next_state, rewards, done, task):
        for i in range(self.num_envs):
            # adding to running state
            self.running_prev_state[self.curr_timstep[i], i] = prev_state[i]
            self.running_next_state[self.curr_timstep[i], i] = next_state[i]
            self.running_rewards[self.curr_timstep[i], i] = rewards[i]
            self.running_actions[self.curr_timstep[i], i] = actions[i]
            if self.running_tasks is not None:
                self.running_tasks[i] = task[i]
            self.curr_timstep[i] += 1

            if done[i]:
                if self.max_buffer_size > 0:
                    if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                        if self.insert_idx + 1 > self.max_buffer_size:
                            self.buffer_len = self.insert_idx
                            self.insert_idx = 0
                        else:
                            self.buffer_len = max(self.buffer_len, self.insert_idx)
                        self.prev_state[:, self.insert_idx] = self.running_prev_state[:, i].to('cpu')
                        self.next_state[:, self.insert_idx] = self.running_next_state[:, i].to('cpu')
                        self.actions[:, self.insert_idx] = self.running_actions[:, i].to('cpu')
                        self.rewards[:, self.insert_idx] = self.running_rewards[:, i].to('cpu')
                        if self.tasks is not None:
                            self.tasks[self.insert_idx] = self.running_tasks[i].to('cpu')
                        self.trajectory_lens[self.insert_idx] = self.curr_timstep[i].clone()
                        self.insert_idx += 1
                self.running_prev_state[:, i] *= 0
                self.running_next_state[:, i] *= 0
                self.running_rewards[:, i] *= 0
                self.running_actions[:, i] *= 0
                if self.running_tasks is not None:
                    self.running_tasks[i] *= 0
                self.curr_timstep[i] = 0

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False):
        batchsize = min(self.buffer_len, batchsize)
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]
        prev_obs = self.prev_state[:, rollout_indices, :]
        next_obs = self.next_state[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        if self.tasks is not None:
            tasks = self.tasks[rollout_indices].to(device)
        else:
            tasks = None

        return prev_obs.to(device), next_obs.to(device), actions.to(device), \
            rewards.to(device), tasks, trajectory_lens


class VariBadVae:
    def __init__(self, args, writer, get_iter_idx):
        self.args = args
        self.writer = writer
        self.get_iter_idx = get_iter_idx
        self.task_dim = None  # don't decode task for MW
        self.num_tasks = None  # don't decode task for MW
        self.encoder = RNNEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_space,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size
        ).to(device)
        self.state_decoder = None  # don't decode state for MW
        self.reward_decoder = RewardDecoder(
            args=self.args,
            layers=self.args.reward_decoder_layers,
            latent_dim=self.args.latent_dim,
            state_dim=39,
            state_embed_dim=self.args.state_embedding_size,
            action_dim=4,
            action_embed_dim=self.args.action_embedding_size,
            multi_head=self.args.multihead_for_reward,
            pred_type=self.args.rew_pred_type,
            input_prev_state=self.args.input_prev_state,
            input_action=self.args.input_action
        ).to(device)
        self.task_decoder = None  # don't decode task for MW, although for ML10 and/or ML45 this may help
        self.storage = RolloutStorageVAE(
            num_envs=args.num_envs,
            max_trajectory_length=args.num_steps,
            zero_pad=True,
            max_num_rollouts=self.args.size_vae_buffer,
            state_dim=39,
            action_dim=4,
            vae_buffer_add_thresh=self.args.vae_buffer_add_thresh,
            task_dim=args.policy_task_embedding_dim
        )

        self.optimizer_vae = torch.optim.Adam([*self.encoder.parameters(), *self.reward_decoder.parameters()],
                                              lr=self.args.lr_vae)

    def compute_rew_reconstruction_loss(self, latent, prev_obs, next_obs, action, reward, return_predictions=False):
        prediction = self.reward_decoder(latent.to(device), next_obs.to(device), prev_obs.to(device), action.float().to(device))
        loss_rew = (prediction - reward).pow(2).mean(dim=-1)
        if return_predictions:
            return loss_rew, prediction
        else:
            return loss_rew

    def compute_kl_loss(self, latent_mean, latent_logvar, elbo_indices):
        if self.args.kl_to_gauss_prior:
            kl_divergences = (-0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1))
        else:
            gauss_dim = latent_mean.shape[-1]
            all_means = torch.cat((torch.zeros(1, *latent_mean.shape[1:]).to(device), latent_mean))
            all_logvars = torch.cat((torch.zeros(1, *latent_logvar.shape[1:]).to(device), latent_logvar))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
                1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))
        if elbo_indices is not None:
            batchsize = kl_divergences.shape[-1]
            task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)
            kl_divergences = kl_divergences[elbo_indices, task_indices].reshape((self.args.vae_subsample_elbos, batchsize))
        return kl_divergences

    def compute_loss(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, trajectory_lens):
        num_unique_traj_lens = len(np.unique(trajectory_lens))
        assert (num_unique_traj_lens == 1) or (self.args.vae_subsample_elbos and self.args.vae_subsample_decodes)
        assert not self.args.decode_only_past

        max_traj_len = np.max(trajectory_lens)
        latent_mean = latent_mean[:max_traj_len+1]
        latent_logvar = latent_logvar[:max_traj_len+1]
        vae_prev_obs = vae_prev_obs[:max_traj_len+1]
        vae_next_obs = vae_next_obs[:max_traj_len+1]
        vae_actions = vae_actions[:max_traj_len+1]
        vae_rewards = vae_rewards[:max_traj_len+1]



        if not self.args.disable_stochasticity_in_latent:
            latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar)
        else:
            latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)

        num_elbos = latent_samples.shape[0]
        num_decodes = vae_prev_obs.shape[0]
        batchsize = latent_samples.shape[1]  # number of trajectories

        # if self.args.vae_subsample_elbos
        elbo_indices = None

        dec_prev_obs = vae_prev_obs.unsqueeze(0).expand((num_elbos, *vae_prev_obs.shape)).to(device)
        dec_next_obs = vae_next_obs.unsqueeze(0).expand((num_elbos, *vae_next_obs.shape)).to(device)
        dec_actions = vae_actions.unsqueeze(0).expand((num_elbos, *vae_actions.shape)).to(device)
        dec_rewards = vae_rewards.unsqueeze(0).expand((num_elbos, *vae_rewards.shape)).to(device)

        dec_embedding = latent_samples.unsqueeze(0).expand((num_decodes, *latent_samples.shape)).transpose(1, 0).to(device)

        if self.args.decode_reward:
            rew_reconstruction_loss = self.compute_rew_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs,
                                                                           dec_actions, dec_rewards)
            if self.args.vae_avg_elbo_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
            if self.args.vae_avg_reconstruction_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
            rew_reconstruction_loss = rew_reconstruction_loss.mean()
        else:
            rew_reconstruction_loss = torch.tensor(0).to(device)
        if self.args.decode_state:
            raise NotImplementedError
        else:
            state_reconstruction_loss = 0
        if self.args.decode_task:
            raise NotImplementedError
        else:
            task_reconstruction_loss = 0

        if not self.args.disable_kl_term:
            kl_loss = self.compute_kl_loss(latent_mean=latent_mean, latent_logvar=latent_logvar,
                                           elbo_indices=elbo_indices)
            if self.args.vae_avg_elbo_terms:
                kl_loss = kl_loss.mean(dim=0)
            else:
                kl_loss = kl_loss.sum(dim=0)
            kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss

    def compute_vae_loss(self, update=False, pretrain_index=None):
        if not self.storage.ready_for_update():
            return 0
        if self.args.disable_decoder and self.args.disable_kl_term:
            return 0
        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, trajectory_lens = self.storage.get_batch(
            batchsize=self.args.vae_batch_num_trajs
        )
        _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions, states=vae_next_obs, rewards=vae_rewards,
                                                        hidden_state=None, return_prior=True, detach_every=None)
        if self.args.split_batches_by_task:
            raise NotImplementedError
        elif self.args.split_batches_by_elbo:
            raise NotImplementedError
        else:
            losses = self.compute_loss(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions, vae_rewards,
                                       vae_tasks, trajectory_lens)
        rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss = losses

        loss = (self.args.rew_loss_coeff * rew_reconstruction_loss +
                self.args.state_loss_coeff * state_reconstruction_loss +
                self.args.task_loss_coeff * task_reconstruction_loss +
                self.args.kl_weight * kl_loss).mean()
        if not self.args.disable_kl_term:
            assert kl_loss.requires_grad
        if self.args.decode_reward:
            assert rew_reconstruction_loss.requires_grad
        if self.args.decode_state:
            assert state_reconstruction_loss.requires_grad
        if self.args.decode_task:
            assert task_reconstruction_loss.requires_grad

        elbo_loss = loss.mean()

        if update:
            self.optimizer_vae.zero_grad()
            elbo_loss.backward()
            # could add additional checks for gradient clipping
            self.optimizer_vae.step()

    def log(self, elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss,
            pretrain_index=None):

        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = - self.args.pretrain_len * self.args.num_vae_updates_per_pretrain + pretrain_index

        if curr_iter_idx % self.args.log_interval == 0:

            if self.args.decode_reward:
                self.writer.add_scalar('vae_losses/reward_reconstr_err', rew_reconstruction_loss.mean(), curr_iter_idx)
            if self.args.decode_state:
                self.writer.add_scalar('vae_losses/state_reconstr_err', state_reconstruction_loss.mean(), curr_iter_idx)
            if self.args.decode_task:
                self.writer.add_scalar('vae_losses/task_reconstr_err', task_reconstruction_loss.mean(), curr_iter_idx)

            if not self.args.disable_kl_term:
                self.writer.add_scalar('vae_losses/kl', kl_loss.mean(), curr_iter_idx)
            self.writer.add_scalar('vae_losses/sum', elbo_loss, curr_iter_idx)





if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    print(args)
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
    benchmark = None
    if args.env_id == 'ML1':
        benchmark = metaworld.ML1(env_name=args.env_name, seed=args.seed)
    elif args.env_id == 'ML10':
        benchmark = metaworld.ML10(seed=args.seed)
    elif args.env_id == 'ML450':
        benchmark = metaworld.ML45(seed=args.seed)

    use_one_hot_wrapper = True if 'MT10' in args.env_id or 'MT50' in args.env_id else False

    # env setup
    envs = SyncVectorEnv(
        benchmark.train_classes, benchmark.train_tasks, use_one_hot_wrapper=use_one_hot_wrapper
    )

    args.state_dim = envs.single_observation_space.shape[0]
    args.action_space = envs.single_action_space.shape[0]
    args.task_dim = 0
    args.belief_dim = 0

    keys = list(benchmark.train_classes.keys())

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    #  envs, args, activation_function, hidden_layers, dim_state, dim_latent,
    #                  dim_belief, dim_task, init_std, pass_state_to_policy, pass_latent_to_policy, pass_task_to_policy,
    #                  pass_belief_to_policy
    agent = Agent(envs, args, args.policy_activation_function, args.policy_layers,
                  args.policy_init_std, args.pass_state_to_policy, args.pass_latent_to_policy,
                  args.pass_task_to_policy, args.pass_belief_to_policy).to(torch.float32).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.policy_learning_rate, eps=args.policy_eps)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    masks = torch.ones((args.num_steps, args.num_envs, 1)).to(device)
    bad_masks = torch.ones((args.num_steps, args.num_envs, 1))
    prev_obs_storage = torch.zeros((args.num_steps, args.num_envs, args.state_dim)).to(device)
    next_obs_storage = torch.zeros((args.num_steps, args.num_envs, args.state_dim)).to(device)
    hidden_state_storage = torch.zeros((args.num_steps, args.num_envs, args.encoder_gru_hidden_size)).to(device)
    latent_mean_storage = []
    latent_sample_storage = []
    latent_logvars_storage = []


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    prev_obs, _ = envs.reset(seed=args.seed)
    belief, task = None, None
    prev_obs = torch.Tensor(prev_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = int(args.total_timesteps // args.batch_size)
    update = 1
    prev_obs_storage[0].copy_(prev_obs)

    vae = VariBadVae(args, writer, lambda: update)

    for update in range(1, num_updates + 1):
        if (update - 1) % args.eval_freq == 1564654654:
            ### NEED TO SET TRAIN OR TEST TASKS
            agent = agent.to('cpu')
            agent = agent.eval()
            agent.turn_off_grads()
            vae.encoder = vae.encoder.eval()
            vae.encoder.turn_off_grads()

            evaluation_procedure(num_envs=args.num_envs, writer=writer, agent=agent,
                                 update=update, keys=keys, classes=benchmark.train_classes,
                                 tasks=benchmark.train_tasks, task_on_reset=False, writer_append='train',
                                 encoder=vae.encoder, args=args, add_onehot=use_one_hot_wrapper)
            evaluation_procedure(num_envs=args.num_envs, writer=writer, agent=agent,
                                 update=update, keys=keys, classes=benchmark.test_classes,
                                 tasks=benchmark.test_tasks, task_on_reset=False, writer_append='test',
                                 encoder=vae.encoder, args=args, add_onehot=use_one_hot_wrapper)
            agent = agent.to(device)
            agent = agent.train()
            vae.encoder = vae.encoder.train()
            vae.encoder.turn_on_grads()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        with (torch.no_grad()):
            prev_obs_batch, next_obs_batch, act_batch, rew_batch, lens_batch = vae.storage.get_running_batch()
            all_latent_samples, all_latent_means, all_latent_logvars, \
            all_hidden_states = vae.encoder(actions=act_batch, states=next_obs_batch,
                                          rewards=rew_batch,
                                          hidden_state=None,
                                          return_prior=True)
            latent_sample = (torch.stack([all_latent_samples[lens_batch[i]][i] for i in range(len(lens_batch))])).to(device)
            latent_mean = (torch.stack([all_latent_means[lens_batch[i]][i] for i in range(len(lens_batch))])).to(device)
            latent_logvars = (torch.stack([all_latent_logvars[lens_batch[i]][i] for i in range(len(lens_batch))])).to(device)
            hidden_state = (torch.stack([all_hidden_states[lens_batch[i]][i] for i in range(len(lens_batch))])).to(device)

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = prev_obs
            dones[step] = next_done

            latent_sample_storage.append(latent_sample.detach().clone())
            latent_mean_storage.append(latent_mean.detach().clone())
            latent_logvars_storage.append(latent_logvars.detach().clone())
            hidden_state_storage[step] = hidden_state

            # ALGO LOGIC: action logic
            with torch.no_grad():
                latent = get_latent_for_policy(args, latent_sample, latent_mean, latent_logvars)
                action = agent.act(state=prev_obs, latent=latent, belief=belief, task=task, deterministic=False)
                if isinstance(action, tuple):
                    value, action, logprob, entropy = action
                else:
                    value = None
                    logprob = None
                    entropy = None
                action = action.to(device)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            reward_normalized = 0
            next_obs, reward_raw, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            rewards[step] = torch.tensor(reward_raw).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            with torch.no_grad():
                if True in done:
                    hidden_state = vae.encoder.reset_hidden(hidden_state.cpu(), torch.tensor(done))
                r = rewards[step].reshape(10, 1)
                r = r.squeeze(0)

                latent_sample, latent_mean, latent_logvar, hidden_state = vae.encoder(actions=action.float(),
                                                                                      states=next_obs,
                                                                                      rewards=r,
                                                                                      hidden_state=hidden_state.cpu(),
                                                                                      return_prior=False)
            vae.storage.insert(prev_obs.clone(),
                               actions[step].detach().clone(),
                               next_obs.clone(), r.clone(),
                               done, None
                               )

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
                with torch.no_grad():  # state, latent, belief, task
                    next_value, action, logprob, entropy = agent.act(next_obs, latent, belief, task)
                    next_value = torch.squeeze(next_value)
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
                        latent_sample_batch = torch.cat(latent_sample_storage[:-1])[mb_inds].detach()
                        latent_mean_batch = torch.cat(latent_mean_storage[:-1])[mb_inds].detach()
                        latent_logvar_batch = torch.cat(latent_logvars_storage[:-1])[mb_inds].detach()
                        state_batch = b_obs[mb_inds].detach()
                        latent_batch = get_latent_for_policy(args, latent_sample_batch, latent_mean_batch, latent_logvar_batch)
                        newvalue, action_log_probs, dist_entropy = agent.evaluate_actions(state_batch, latent_batch,
                                                                                        None, None, b_actions[mb_inds].detach())
                        logratio = action_log_probs - b_logprobs[mb_inds]
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

                vae.compute_vae_loss(update=True)

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
