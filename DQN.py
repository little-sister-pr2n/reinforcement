import copy
import random
import gym
import utils 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EVALATE = 10
EXPL_NOISE = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Qnetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_space) -> None:
        super(Qnetwork, self).__init__()

        # define layer
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.a1 = torch.from_numpy(1/2 * (action_space[1] + action_space[0])).to(device)
        self.a2 = torch.from_numpy(1/2 * (action_space[1] - action_space[0])).to(device)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return self.a1 + self.a2 * x

class DQN:
    def __init__(self, ):
        pass

    def select_action(self, state):
        st = torch.from_numpy(state).float().to(self.device)
        action = self.actor(st).cpu().detach().numpy()
        return action

    def evaluate(self, env_name, seed):
        env = gym.make(env_name)
        seed_eval = seed + 100
        ret_list = list()
        torch.manual_seed(seed_eval)
        np.random.seed(seed_eval)
        random.seed(seed_eval)
        for t in range(EVALATE):
            state = env.reset()
            done = False
            ret_list.append(0)
            while not done:
                action = self.select_action(state) 
                next_state , reward , done, _ = env.step(action)
                ret_list[t] += float(reward)
                state = next_state
        env.close()
        return ret_list 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int) 
    parser.add_argument("--env", default="BipedalWalker-v3")
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--expl_timesteps", default=25e3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    env = gym.make(args.env)
    result = []
    turns = []
    episode_reward = 0

    max_action = env.action_space.high 
    min_action = env.action_space.low
    action_space = np.array([min_action, max_action])

    action_dim = len(max_action)
    state_dim = len(env.observation_space.high)
    

    buffer = utils.ReplayBuffer(state_dim, action_dim, int(args.max_timesteps), device)
    agent = TD3(state_dim, action_dim, action_space, device)

    # set seed
    env.seed(int(args.seed))
    env.action_space.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    state = env.reset()
    episode_timesteps = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < int(args.expl_timesteps):
            action = env.action_space.sample()
        else:
            action = (
                agent.select_action(np.array(state))
                + np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) # if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.expl_timesteps:
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
            turns.append(t)
            result.append(agent.evaluate(args.env, int(args.seed)))
            # if args.save_model: policy.save(f"./models/{file_name}")

    result = np.array(result)
    plt.plot(turns, np.mean(result.T, axis = 0), color="blue", zorder=3)
    plt.fill_between(turns, np.percentile(result.T, 25, axis=0), np.percentile(result.T, 75, axis=0), color="skyblue", zorder=1)
    plt.plot(turns, np.percentile(result.T, 25, axis=0), color="blue", zorder=2)
    plt.plot(turns, np.percentile(result.T, 75, axis=0), color="blue", zorder=2)
    plt.ylabel("cumulative rewards")
    plt.show()
    plt.clf()
    plt.close()
    env.close()
