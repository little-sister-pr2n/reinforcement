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

EVALATE = 30
EXPL_NOISE = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, dim_in, dim_out, action_space) -> None:
        super(Actor, self).__init__()

        # define layer
        self.l1 = nn.Linear(dim_in, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, dim_out)

        # action_space is like [[min], [max]] : shape as (2 x dim_in) 
        self.a1 = torch.from_numpy(1/2 * (action_space[1] + action_space[0])).to(device)
        self.a2 = torch.from_numpy(1/2 * (action_space[1] - action_space[0])).to(device)

    def forward(self, state):
        # layer1: ReLU
        # layer2: ReLU
        # layer3: ((max+min) + (max-min) * tanh ) / 2
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return self.a1 + self.a2 * x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(Critic, self).__init__()

        # define layer
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(torch.cat([x, action], 1)))
        return self.l3(x)

class DDPG:
    def __init__(self, dim_state, dim_action, action_space, device, 
        discount=0.99, noise_action=0.1, tau=0.005, noise_var=0.2,) -> None:
        self.device = device
        
        self.actor = Actor(dim_state, dim_action, action_space).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic1 = Critic(dim_state, dim_action).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), weight_decay=1e-2)
        self.critic1_target = copy.deepcopy(self.critic1)

        # other parameter
        self.discount = discount
        self.noise_var = noise_var
        self.tau = tau
        self.noise_action = noise_action
        self.count = 0
        self.max_action = action_space[1][0] # バグの温床になりそう

        sigma_beta = 0.1
        D = np.diag([(a[1]-a[0]) * sigma_beta/2 for a in action_space])
        self.D2 = D * D 

    def train(self, buffer, batch_size=256):
        if buffer.size >= batch_size:
            # Sample replay buffer 
            state, action, next_state, reward, not_done = buffer.sample(batch_size)

            with torch.no_grad():

                next_action = (
                    self.actor_target(next_state)
                )

            # Compute the target Q value
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q = target_Q1
            target_Q = reward + not_done * self.discount * target_Q
            target_Q.detach()

            # Optimize the critic
            current_Q1 = self.critic1(state, action)
            critic1_loss = F.mse_loss(current_Q1, target_Q) 
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            # Compute actor losse
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_((1-self.tau) * target_param.data + self.tau * param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_((1-self.tau) * target_param.data + self.tau * param.data)


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
    agent = DDPG(state_dim, action_dim, action_space, device)

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
