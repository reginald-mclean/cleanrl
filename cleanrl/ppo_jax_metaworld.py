import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Deque, NamedTuple, Optional, Tuple, Type, Union, Sequence
from functools import partial
import functools
from collections import deque
import sys
sys.path.append('/home/reggiemclean/clip4clip/cleanrl')

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.05"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991



import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
import distrax
from cleanrl_utils.evals.metaworld_jax_eval import ppo_evaluation
from jax.config import config
from clip4clip.reward import RewardCalculator
from clip4clip.util   import get_args as clip_args
import torch

# config.update('jax_disable_jit', True)
# config.update("jax_enable_x64", True)

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
    parser.add_argument("--wandb-project-name", type=str, default="VLM-PPO-State-Based",
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
    parser.add_argument("--vf-coef", type=float, default=0.001,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--eval-freq", type=int, default=2,
        help="how many updates to do before evaluating the agent")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes to run per evaluation")

    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")

    parser.add_argument(
        "--do_pretrain", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        default=True,
        action="store_true",
        help="Whether to run eval.",
    )
    parser.add_argument(
        "--eval_on_val",
        default=False,
        action="store_true",
        help="If True, run do_eval evaluation on the validation set. Else run on the test set.",
    )

    parser.add_argument("--train_csv", type=str, default="data/.train.csv", help="")
    parser.add_argument("--val_csv", type=str, default="data/.val.csv", help="")
    parser.add_argument(
        "--data_path",
        type=str,
        default=f"/home/reggiemclean/clip4clip/clip4clip_data/vlm_dataset/",
        help="data pickle file path",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=f"/home/reggiemclean/clip4clip/clip4clip_data/",
        help="feature path",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="data path for test data",
    )
    parser.add_argument(
        "--test_features_path",
        type=str,
        default=None,
        help="feature path for test data",
    )
    parser.add_argument(
        "--evaluate_test_accuracy",
        default=False,
        action="store_true",
        help="If True, evaluate accuracy on test data.",
    )
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="If True, use only a few batches of data in training and validation for development purposes.",
    )

    parser.add_argument("--num_thread_reader", type=int, default=1, help="")
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--batch_size_val", type=int, default=16, help="batch size eval"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="Learning rate exp epoch decay"
    )
    parser.add_argument(
        "--n_display", type=int, default=100, help="Information display frequence"
    )
    parser.add_argument(
        "--video_dim", type=int, default=1024, help="video feature dimension"
    )
    parser.add_argument(
        "--video_max_len", type=int, default=-1, help="Max length of video"
    )
    parser.add_argument(
        "--deduplicate_captions",
        action="store_true",
        default=False,
        help="Whether to remove duplicate captions in each batch.",
    )
    parser.add_argument("--max_words", type=int, default=32, help="")
    parser.add_argument("--max_frames", type=int, default=12, help="")
    parser.add_argument("--feature_framerate", type=int, default=1, help="")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for loss")
    parser.add_argument(
        "--hard_negative_rate",
        type=float,
        default=0.5,
        help="rate of intra negative sample",
    )
    parser.add_argument(
        "--augment_images",
        action="store_true",
        default=False,
        help="Whether to add image augmentations.",
    )
    parser.add_argument(
        "--add_reversed_negatives",
        action="store_true",
        default=False,
        help="Whether to use the videos played in reverse as negative examples.",
    )
    parser.add_argument(
        "--test_on_reversed_negatives",
        action="store_true",
        default=False,
        help="Whether to use the videos played in reverse as negative examples also at evaluation time.",
    )
    parser.add_argument(
        "--use_failures_as_negatives_only",
        action="store_true",
        default=False,
        help="Whether to only use the videos demonstrating failures as negative examples for other tasks.",
    )
    parser.add_argument(
        "--success_data_only",
        action="store_true",
        default=False,
        help="Whether the dataset includes only successful trajectories. If True, do not compute strict AUC.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        help="Type of loss function to use",
    )
    parser.add_argument(
        "--dist_type",
        type=str,
        default="cosine",
        help="Type of distance function to use between embeddings",
    )
    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.2,
        help="Margin to enforce between positive and negative pairs in triplet loss",
    )
    parser.add_argument(
        "--progress_margin",
        type=float,
        default=None,
        help="Margin to enforce between successively longer segments of the video.",
    )
    parser.add_argument(
        "--main_eval_metric",
        type=str,
        default="loss",
        help="Which metric to use for model selection",
    )
    parser.add_argument(
        "--other_eval_metrics",
        type=str,
        default=None,
        help="Other eval metrics for which to keep best checkpoints.",
    )
    parser.add_argument(
        "--n_ckpts_to_keep",
        type=int,
        default=1,
        help="The number of checkpoints to keep in addition to the best epoch. Set to -1 to keep all.",
    )

    parser.add_argument(
        "--negative_weighting",
        type=int,
        default=1,
        help="Weight the loss for intra negative",
    )
    parser.add_argument(
        "--n_pair", type=int, default=1, help="Num of pair to output from data loader"
    )

    parser.add_argument(
        "--output_dir",
        default=f"/home/reggiemclean/clip4clip/ckpts/ckpt_vlm_retrieval_looseType",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--wandb_entity",
        default="",
        type=str,
        help="Name of the weights & biases entity to use, if any.",
    )
    parser.add_argument(
        "--wandb_project",
        default="",
        type=str,
        help="Name of the weights & biases project to use, if any.",
    )
    parser.add_argument(
        "--cross_model",
        default="cross-base",
        type=str,
        required=False,
        help="Cross module",
    )
    parser.add_argument(
        "--init_model",
        default=None,
        type=str,
        required=False,
        help="Initial model.",
    )
    parser.add_argument(
        "--resume_model",
        default=None,
        type=str,
        required=False,
        help="Resume train model.",
    )
    parser.add_argument(
        "--resume_from_latest",
        action="store_true",
        default=False,
        help="Whether to resume from the latest checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite experiment in output_dir.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="Changed in the execute process."
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--task_type",
        default="retrieval",
        type=str,
        help="Point the task `retrieval` to finetune.",
    )
    parser.add_argument(
        "--datatype", default="vlm", type=str, help="Point the dataset to finetune."
    )
    parser.add_argument(
        "--test_datatype", default=None, type=str, help="Point the dataset to test on."
    )
    parser.add_argument(
        "--test_set_name",
        default="test",
        type=str,
        help="Test data type added to output path when saving eval results.",
    )

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument(
        "--coef_lr", type=float, default=1.0, help="coefficient for bert branch."
    )
    parser.add_argument(
        "--use_mil",
        action="store_true",
        help="Whether use MIL as Miech et. al. (2020).",
    )
    parser.add_argument(
        "--sampled_use_mil",
        action="store_true",
        help="Whether MIL, has a high priority than use_mil.",
    )

    parser.add_argument(
        "--text_num_hidden_layers", type=int, default=12, help="Layer NO. of text."
    )
    parser.add_argument(
        "--visual_num_hidden_layers", type=int, default=12, help="Layer NO. of visual."
    )
    parser.add_argument(
        "--cross_num_hidden_layers", type=int, default=4, help="Layer NO. of cross."
    )

    parser.add_argument(
        "--loose_type",
        action="store_true",
        help="Default using tight type for retrieval.",
    )
    parser.add_argument("--expand_msrvtt_sentences", action="store_true", help="")

    parser.add_argument(
        "--train_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )
    parser.add_argument(
        "--eval_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )

    parser.add_argument(
        "--freeze_layer_num",
        type=int,
        default=0,
        help="Layer NO. of CLIP need to freeze.",
    )
    parser.add_argument(
        "--slice_framepos",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly; 3: sample randomly from uniform intervals.",
    )
    parser.add_argument(
        "--test_slice_framepos",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly; 3: sample randomly from uniform intervals.",
    )
    parser.add_argument(
        "--linear_patch",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="linear projection of flattened patches.",
    )
    parser.add_argument(
        "--sim_header",
        type=str,
        default="meanP",
        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
        help="choice a similarity header.",
    )

    parser.add_argument(
        "--pretrained_clip_name",
        default="ViT-B/32",
        type=str,
        help="Choose a CLIP version",
    )

    args = parser.parse_args()


    args.loose_type = args.sim_header != "tightTransf"
    if args.test_datatype is None:
        args.test_datatype = args.datatype

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    args.ppo_batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.ppo_batch_size // args.num_minibatches)
    args.num_updates = int(int(args.total_timesteps) // args.ppo_batch_size)

    # fmt: on
    return args


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512, kernel_init=orthogonal(0.01, dtype=jnp.float32), bias_init=constant(0.0, dtype=jnp.float32))(x)
        x = nn.tanh(x)
        x = nn.Dense(512, kernel_init=orthogonal(0.01, dtype=jnp.float32), bias_init=constant(0.0, dtype=jnp.float32))(x)
        x = nn.tanh(x)
        #log_std_init = functools.partial(nn.initializers.ones, dtype=jnp.float32)
        #log_std = self.param('log_std', log_std_init, (self.action_dim,))
        #expanded_log_std = jnp.tile(log_std[None, :], (x.shape[0], 1))
        #expanded_log_std = jnp.clip(expanded_log_std, LOG_STD_MIN, LOG_STD_MAX)
        #std = jnp.exp(expanded_log_std)
        return nn.Dense(2*self.action_dim, kernel_init=orthogonal(0.01, dtype=jnp.float32), bias_init=constant(1.0, dtype=jnp.float32))(x)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32, kernel_init=orthogonal(1, dtype=jnp.float32), bias_init=constant(0.0, dtype=jnp.float32))(x)
        x = nn.tanh(x)
        x = nn.Dense(32, kernel_init=orthogonal(1, dtype=jnp.float32), bias_init=constant(0.0, dtype=jnp.float32))(x)
        x = nn.tanh(x)
        return nn.Dense(1, kernel_init=orthogonal(1, dtype=jnp.float32), bias_init=constant(0.0, dtype=jnp.float32))(x)


@flax.struct.dataclass
class AgentParams:
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array

@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array

import metaworld
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
from cleanrl_utils.wrappers import metaworld_wrappers

def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    max_episode_steps: Optional[int] = None,
    use_one_hot: bool = True,
    terminate_on_success: bool = False,
) -> gym.vector.VectorEnv:
    def init_each_env(env_cls: Type[SawyerXYZEnv], name: str, env_id: int) -> gym.Env:
        env = env_cls(render_mode='rgb_array')
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            env = metaworld_wrappers.OneHotWrapper(env, env_id, len(benchmark.train_classes))
        tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        if not terminate_on_success:
            tasks = tasks[env_id:(env_id+5)]
        env = metaworld_wrappers.PseudoRandomTaskSelectWrapper(env, tasks, True)
        env.action_space.seed(seed)
        return env

    return gym.vector.AsyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=list(benchmark.train_classes.keys())[0], env_id=env_id)
            for env_id, (name, env_cls) in enumerate(benchmark.train_classes.items())
        ]
    )


LOG_STD_MAX = 1.5
LOG_STD_MIN = 0.5

if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(f'seed {args.seed}')
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
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    # env setup
    make_envs = partial(_make_envs_common, terminate_on_success=False)
    make_eval_envs = partial(_make_envs_common, terminate_on_success=True)

    if args.env_id == "MT10":
        benchmark = metaworld.MT10(seed=args.seed)
        args.num_envs = 10
    elif args.env_id == "MT50":
        benchmark = metaworld.MT50(seed=args.seed)
        args.num_envs = 50
    else:
        benchmark = metaworld.MT1(args.env_id, seed=args.seed)
        eval_benchmark = metaworld.MT1(args.env_id, seed=args.seed)
        args.num_envs = 10
        for i in range(1, 10):
            benchmark.train_classes[str(args.env_id) + f' {i}'] = benchmark.train_classes[args.env_id]

    use_one_hot_wrapper = True if "MT10" in args.env_id or "MT50" in args.env_id else False
    envs = make_envs(benchmark, args.seed, args.num_steps, use_one_hot=use_one_hot_wrapper)
    eval_envs = make_eval_envs(eval_benchmark, args.seed, args.num_steps, use_one_hot=use_one_hot_wrapper)
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )
    reward_model = RewardCalculator(args=args)
    reward_model.model.eval()

    frames = np.zeros((args.num_envs, 500, 480, 480, 3))


    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    print(envs.single_action_space)
    actor = Actor(action_dim=envs.single_action_space.shape[0])
    critic = Critic()
    #network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            actor.init(actor_key, np.array([envs.single_observation_space.sample()])),
            critic.init(critic_key, np.array([envs.single_observation_space.sample()])),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=args.learning_rate, eps=1e-5
            ),
        ),
    )
    #network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=jnp.float32),
        actions=jnp.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=jnp.float32),
        logprobs=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        dones=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        values=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        advantages=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        returns=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        rewards=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
    )

    #@jax.jit
    def get_action_and_value(
        state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        output = actor.apply(state.params.actor_params, next_obs)
        #print(output.shape)
        mean, std = output[:, :4], output[:, 4:]
        #print(mean.shape, std.shape)
        std = jnp.clip(std, LOG_STD_MIN, LOG_STD_MAX) # LOG_STD_MAX
        std = jnp.exp(std)
        #exit(0)
        key, subkey = jax.random.split(key)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        action = dist.sample(seed=subkey)
        logprob = dist.log_prob(action)
        value = critic.apply(state.params.critic_params, next_obs)
        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value.squeeze()),
        )
        action = jax.device_get(action)
        return storage, action, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        output = actor.apply(params.actor_params, x)
        mean, std = output[:, :4], output[:, 4:]
        std = jnp.clip(std, LOG_STD_MIN, LOG_STD_MAX) # LOG_STD_MAX
        std = jnp.exp(std)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        value = critic.apply(params.critic_params, x)
        logprob = dist.log_prob(action)
        return logprob, dist.entropy(), value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)


    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(agent_state.params.critic_params, next_obs).squeeze()
        print(next_value.shape)
        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        print(storage.values.shape, next_value.shape)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    #@jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        b_obs = storage.obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = storage.logprobs.reshape(-1)
        b_actions = storage.actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = storage.advantages.reshape(-1)
        b_returns = storage.returns.reshape(-1)
        b_values = storage.values.reshape(-1)

        @jax.jit
        def ppo_loss(params, x, a, mb_values, logp, mb_advantages, mb_returns):
            newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.reshape(-1)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + jnp.clip(newvalue - mb_values, -args.clip_coef, args.clip_coef)
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        for a in range(args.update_epochs):
            key, subkey = jax.random.split(key)
            b_inds = jax.random.permutation(subkey, args.ppo_batch_size, independent=True)
            for start in range(0, args.ppo_batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_values[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                )
                agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    print(f'next obs {next_obs.shape}')
    next_done = np.zeros(args.num_envs)

    global_episodic_return: Deque[float] = deque([], maxlen=20 * args.num_envs)
    global_episodic_length: Deque[int] = deque([], maxlen=20 * args.num_envs)

    text_description = 'pick the red object up and place it on the blue goal'
    def rollout(agent_state, next_obs, next_done, storage, key, global_step):
        for step in range(0, args.num_steps):
            images_needed = np.linspace(0, step%500, num=12, dtype=int)
            global_step += 1 * args.num_envs
            storage, action, key = get_action_and_value(agent_state, next_obs, next_done, storage, step, key)

            # TRY NOT TO MODIFY: execute the game and log data.
            
            next_obs, _, truncate, terminate, infos = envs.step(action)
            current_frames = envs.call('render')
            for idx, f in enumerate(current_frames):
                frames[idx, step % 500] = f
                # frames = np.zeros((args.num_envs, 500, 480, 480, 3))

            current_videos = frames[:, images_needed]
            data = []
            reward = []
            batch_list_t = []
            batch_list_v = []
            batch_sequence_output_list, batch_visual_output_list = [], []
            for env_num in range(args.num_envs):
                input_ids, input_mask, segment_ids, video, video_mask = reward_model.dataloader.get_encoded_data(current_videos[env_num], text_description)
                sequence_output, visual_output = reward_model.model.get_sequence_visual_output(torch.from_numpy(input_ids).to('cuda:0'), torch.from_numpy(segment_ids).to('cuda:0'), torch.from_numpy(input_mask).to('cuda:0'),
                         torch.from_numpy(video).unsqueeze(0).to('cuda:0'), torch.from_numpy(video_mask).to('cuda:0'))
                res = reward_model.model.get_similarity_logits(sequence_output, visual_output,  torch.from_numpy(input_mask).to('cuda:0'),  torch.from_numpy(video_mask).to('cuda:0'), loose_type=reward_model.model.loose_type)
                res = res[0].cpu().detach().numpy()[0][0]
                reward.append(res)
            next_done = np.logical_or(truncate, terminate)
            storage = storage.replace(rewards=storage.rewards.at[step].set(jnp.asarray(reward)))
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
        return agent_state, next_obs, next_done, storage, key, global_step

    #print(f'num updates {args.num_updates}')

    for update in range(1, args.num_updates + 1):
        #print(agent_state.params.actor_params)
        #print('actor')
        for k in agent_state.params.actor_params['params']:
            if 'Dense' in k:
                print(k)
                print(jnp.linalg.norm(agent_state.params.actor_params['params'][k]['kernel']))
        print('critic')
        for k in agent_state.params.critic_params['params']:
            if 'Dense' in k:
                print(k) 
                print(jnp.linalg.norm(agent_state.params.critic_params['params'][k]['kernel']))

        if (update - 1) % args.eval_freq == 0:
            eval_success_rate, eval_returns, eval_success_per_task, key = ppo_evaluation(
                    agent_state=agent_state,
                    actor=actor,
                    eval_envs=eval_envs,
                    num_episodes=args.evaluation_num_episodes,
                    key=key,
                    #task_names=list(benchmark.train_classes.keys())
            )
            print(eval_success_rate, eval_returns, eval_success_per_task)
            eval_metrics = {
                    "charts/mean_success_rate": float(eval_success_rate),
                    "charts/mean_evaluation_return": float(eval_returns),
                } | {
                    f"charts/{env_name}_success_rate": float(eval_success_per_task[i])
                    for i, (env_name, _) in enumerate(eval_benchmark.train_classes.items())
            }
            #print(eval_metrics)
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, global_step)
            print(
                    f"global_step={global_step}, mean evaluation success rate: {eval_success_rate:.4f}"
                    + f" return: {eval_returns:.4f}"
            )

            # Checkpointing
            if args.save_model:
                ckpt = agent.get_ckpt()
                ckpt["rng_key"] = key
                ckpt["global_step"] = global_step
                save_args = orbax_utils.save_args_from_target(ckpt)
                ckpt_manager.save(
                    step=global_step, items=ckpt, save_kwargs={"save_args": save_args}, metrics=eval_metrics
                )
                print(f"model saved to {ckpt_manager.directory}")

        update_time_start = time.time()
        agent_state, next_obs, next_done, storage, key, global_step = rollout(
            agent_state, next_obs, next_done, storage, key, global_step
        )
        #print(update)
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        #print(jax.make_jaxpr(update_ppo)(agent_state, storage, key))
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )

        avg_episodic_return = np.mean(list(global_episodic_return))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)), global_step
        )
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - update_time_start)), global_step
        )

    envs.close()
    writer.close()
