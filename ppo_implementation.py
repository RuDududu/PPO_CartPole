from CartPole_argparse import parse_args, override_args
from model_init import _evaluate_agent, make_env, layer_init, Agent, get_action, get_action_and_value
from plot_viz import plot_from_tensorboard

import gymnasium as gym
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

args = parse_args()

args = override_args(
    anneal_lr       = False,
    learning_rate   = 2.5e-4,
    target_kl       = 0.02,      # slightly looser as allow bigger policy steps now critic is stable
    num_envs        = 16,
    num_steps       = 512,
    ent_coef        = 0.005,     # reduce further as entropy is already low
    clip_coef       = 0.2,
    clip_vloss      = True,
    vf_coef         = 0.5,
    update_epochs   = 8,         # more passes as stable critic can support more epochs
    total_timesteps = 5000000,   # extend as the policy is still improving at 3M
)


run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

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

# Seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# Vectorised training envs
envs = gym.vector.SyncVectorEnv(
    [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
)

assert isinstance(envs.single_action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)), \
    "only Discrete / MultiDiscrete action spaces are supported"

# Single eval env (not vectorised)
eval_env = gym.make(args.gym_id)
eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

agent     = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# Storage
obs      = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions  = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards  = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones    = torch.zeros((args.num_steps, args.num_envs)).to(device)
values   = torch.zeros((args.num_steps, args.num_envs)).to(device)

global_step = 0
start_time  = time.time()
next_obs, _ = envs.reset(seed=args.seed)
next_obs    = torch.Tensor(next_obs).to(device)
next_done   = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size


for update in range(1, num_updates + 1):

    # Learning rate annealing
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * args.learning_rate

    # Rollout collection
    for step in range(args.num_steps):
        global_step += args.num_envs
        obs[step]   = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()

        actions[step]  = action
        logprobs[step] = logprob

        next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = terminated | truncated  # element-wise OR over the env batch
        
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs      = torch.Tensor(next_obs).to(device)
        next_done     = torch.Tensor(done).to(device)

        if "episode" in info:
            ep_return = float(info["episode"]["r"][0])
            print(f"global_step={global_step}, episodic_return={ep_return:.2f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", float(info["episode"]["l"][0]), global_step)

    # Bootstrap value
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        if args.gae:
            advantages  = torch.zeros_like(rewards).to(device)
            lastgaelam  = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues      = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues      = values[t + 1]

                delta          = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t]  = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        else:
            returns = torch.zeros_like(rewards).to(device)

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return     = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return     = returns[t + 1]

                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return

            advantages = returns - values

    # Flatten batch
    b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs   = logprobs.reshape(-1)
    b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns    = returns.reshape(-1)
    b_values     = values.reshape(-1)

    # Policy & value optimisation
    b_inds    = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end      = start + args.minibatch_size
            mb_inds  = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions.long()[mb_inds]
            )

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio    = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl     = ((ratio - 1) - logratio).mean()
                clipfracs    += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2

                v_clipped        = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )

                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss         = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss         = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    # Logging
    y_pred, y_true  = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y           = np.var(y_true)
    explained_var   = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    writer.add_scalar("charts/learning_rate",    optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss",       v_loss.item(),         global_step)
    writer.add_scalar("losses/policy_loss",      pg_loss.item(),        global_step)
    writer.add_scalar("losses/entropy",          entropy_loss.item(),   global_step)
    writer.add_scalar("losses/old_approx_kl",   old_approx_kl.item(),  global_step)
    writer.add_scalar("losses/approx_kl",        approx_kl.item(),      global_step)
    writer.add_scalar("losses/clipfrac",         np.mean(clipfracs),    global_step)
    writer.add_scalar("losses/explained_variance", explained_var,       global_step)
    sps = int(global_step / (time.time() - start_time))
    print(f"Update {update}/{num_updates} | SPS: {sps}")
    writer.add_scalar("charts/SPS", sps, global_step)

    # Periodic evaluation using _evaluate_agent
    if args.eval_freq > 0 and update % args.eval_freq == 0:
        mean_reward, std_reward = _evaluate_agent(eval_env, args.eval_episodes, agent, device)
        print(f"  [Eval] update={update} | mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        writer.add_scalar("eval/mean_reward", mean_reward, global_step)
        writer.add_scalar("eval/std_reward",  std_reward,  global_step)

envs.close()
eval_env.close()
writer.close()

fig = plot_from_tensorboard("runs/CartPole-v1__PPO Exploration__1__1776423291")
plt.show()