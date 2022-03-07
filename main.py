import argparse
from email.policy import default
import gym
import numpy as np
import torch

import SAC
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPL_NOISE = 0.1

def eval(agent, env_name, seed):
    env = gym.make(env_name)
    env.seed(seed+100)

    eval_list = list()
    for t in range(10): # モデルの評価回数は各10エピソードずつ
        state = env.reset()
        done = False    
        tmp_reward = 0
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            tmp_reward += reward
        eval_list.append(tmp_reward)
    
    return eval_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int) 
    parser.add_argument("--env", default="HalfCheetah-v1")
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--expl_timesteps", default=25e3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    env = gym.make(args.env)
    result = []

    action_dim = env.action_space.n
    state_dim = env.observation_space.n
    max_action = env.action_space.high 
    min_action = env.action_space.low

    buffer = utils.ReplayBuffer(state_dim, action_dim, int(args.max_timesteps), device)
    agent = SAC.SAC()

    # set seed
    env.seed(int(args.seed))
    np.seed(int(args.seed))

    state = env.reset()

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                agent.select_action(np.array(state))
                + np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            agent.train(buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            result.append(eval(agent, args.env, int(args.seed)))
            # if args.save_model: policy.save(f"./models/{file_name}")
    env.close()
