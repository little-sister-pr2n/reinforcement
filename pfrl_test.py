from re import A
import gym
import numpy as np
import torch
import torch.nn as nn
import pfrl
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3")      # env for train
    parser.add_argument("--policy", default="TD3")                # reinforcement learning algorithm
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--discount", default=0.99, type=float)   # discount for training
    parser.add_argument("--max_timesteps", default=2e5, type=int) # whole iter for train
    parser.add_argument("--expl_timesteps", default=1e4, type=int)
    parser.add_argument("--batch_size", default=256, type=int)    # num of data sampled from buffer
    parser.add_argument("--eval_n_runs", default=100, type=int)   # num of evaluating model
    parser.add_argument("--eval_interval", default=1e4, type=int)
    parser.add_argument("--eval-n-steps", type=int, default=125000)
    args = parser.parse_args()
    
    pfrl.utils.set_random_seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_env = gym.make(args.env)
    train_env.action_space.seed(int(args.seed))
    train_env.seed(int(args.seed))
    
    eval_env = gym.make(args.env)
    eval_env.seed(int(args.seed) + 100)

    state_space = train_env.observation_space
    action_space = train_env.action_space
    state_size = state_space.low.size
    hidden_size = 256 # hidden layer size
    lr = 0.001 # learning rate

    # state, action が gym.space.Box型でないenvは今回は無視する
    # 実装するならisinstance()で場合分けを行う．面倒で煩雑になるため今回はBox型がない前提で行う

    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity = int(args.max_timesteps))
    if args.policy == "TD3":
        actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space.low.size),
            nn.Tanh(),
            pfrl.policies.DeterministicHead()
        )
        critic1 = nn.Sequential(
            nn.Linear(state_size + action_space.low.size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        critic2 = nn.Sequential(
            nn.Linear(state_size + action_space.low.size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        opt_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        opt_critic1 = torch.optim.Adam(critic1.parameters(), lr=lr)
        opt_critic2 = torch.optim.Adam(critic2.parameters(), lr=lr)

        explorer = pfrl.explorers.AdditiveGaussian(
        scale=0.1, low=action_space.low, high=action_space.high
        )
        agent = pfrl.agents.TD3(
            policy=actor,
            q_func1=critic1,
            q_func2=critic2,
            policy_optimizer=opt_actor,
            q_func1_optimizer=opt_critic1,
            q_func2_optimizer=opt_critic2,
            replay_buffer=replay_buffer,
            explorer=explorer,
            gamma=float(args.discount),
            replay_start_size=int(args.expl_timesteps),
            minibatch_size=int(args.batch_size)
        )
        pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=train_env, 
            steps=int(args.max_timesteps),
            eval_n_steps=None,
            eval_n_episodes=int(args.eval_n_runs),
            outdir="results",
            eval_interval=int(args.eval_interval),
            eval_env=eval_env
        )
if __name__ == "__main__":
    main()
