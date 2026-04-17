
"""
model_init : contains functions to initalize the environment and networks for learning

Some of the functions are originally used for LunarLander-v2;
Referenced and edited for my exploration with CartPole
Original base code from Costa Huang
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def _evaluate_agent(env, n_eval_episodes, policy, device):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment (a single, non-vectorised env)
    :param n_eval_episodes: Number of episodes to evaluate the agent
    :param policy: The agent (Agent instance)
    :param device: torch device
    :return: (mean_reward, std_reward)
    """
    episode_rewards = []

    for episode in range(n_eval_episodes):
        state, info  = env.reset()
        done         = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.Tensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(state_tensor)

            action_np = action.cpu().numpy()[0]

            state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            total_reward += reward

        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards) ; std_reward  = np.std(episode_rewards)

    return mean_reward, std_reward


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(gym_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env) 
        # env = gym.wrappers.NormalizeReward(env)    
        # env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = envs.single_action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
            # layer_init(nn.Linear(64, 1), std=50.0) # Critic adjustment for CartPole-v1
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x) ; probs  = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)