"""
CartPole_argparse : contains functions to initalize hyperparams for PPO learning

:parse_args: Referenced from Costa Huang's tutorial on PPO implementation
             Originally used for LunarLander-v2; repurposed for my exploration with CartPole
             https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

:override_args: a function such that hyperparameter exploration is easier
                Call it to override hyperparams instead of rewriting default args in parse_args
"""

import argparse
import os
from distutils.util import strtobool


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
    #     help="the name of this experiment")

    try:
        default_name = os.path.basename(__file__).rstrip(".py")
    except NameError:
        default_name = "PPO Exploration"

    parser.add_argument("--exp-name", type=str, default=default_name, 
        help="the name of this experiment")
    
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")

    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, # was 0.1
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.05, # was 0.01
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    # Evaluation arguments
    parser.add_argument("--eval-freq", type=int, default=20, # was 10
        help="evaluate the agent every N updates (0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=5,
        help="number of episodes to use for evaluation")
    
    args = parser.parse_args() # For .py script use
    # args, _ = parser.parse_known_args() # For Jupyter use
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # fmt: on
    return args

def override_args(**overrides):
    """
    Build args from defaults, then apply keyword overrides for easier exploration
    """
    import sys
    sys.argv = ["vectorized_arch.py"]   # clear any previous argv
    args = parse_args()                 # load all defaults

    for key, val in overrides.items():
        setattr(args, key, val)         # apply overrides directly

    # recompute derived values
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args