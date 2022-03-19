import argparse
from asyncore import write
from concurrent.futures import process
import logging
import sys

import gym
import gym.wrappers
import numpy as np
import torch
from torch import distributions, nn

import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
from pfrl.nn.lmbda import Lambda
from distutils.version import LooseVersion

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="TD3"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Hopper-v2",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25e4,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=25e2,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    # for PPO
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to update model for per PPO iteration.",
    )
    parser.add_argument(
        "--policy-output-scale",
        type=float,
        default=1.0,
        help="Weight initialization scale of policy output.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    
    def make_env(test):
        env = gym.make(args.env)
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)
    args.outdir = f"./results/{args.env}/{args.agent}/{int(args.seed):03}"
    # args.outdir = f"{args.outdir}_seed={int(args.seed)}_env={args.env}_agent={args.agent}"
    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    # env.seed(int(args.seed)) make_env内で既にシード値が固定されているため不要
    torch.manual_seed(int(args.seed)) # torchの乱数シード

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    if args.agent == "TD3":
        rbuf = replay_buffers.ReplayBuffer(10 ** 6)
        policy = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_size),
            nn.Tanh(),
            pfrl.policies.DeterministicHead(),
        )
        policy_optimizer = torch.optim.Adam(policy.parameters())
        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(obs_size + action_size, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
            )
            q_func_optimizer = torch.optim.Adam(q_func.parameters())
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        explorer = explorers.AdditiveGaussian(
            scale=0.1, low=action_space.low, high=action_space.high
        )
        agent = pfrl.agents.TD3(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            rbuf,
            gamma=0.99,
            soft_update_tau=5e-3,
            explorer=explorer,
            replay_start_size=args.replay_start_size,
            gpu=args.gpu,
            minibatch_size=args.batch_size,
            burnin_action_func=burnin_action_func,
        )
    elif args.agent == "PPO":
        policy = torch.nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )

        vf = torch.nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        obs_normalizer = pfrl.nn.EmpiricalNormalization(
            obs_space.low.size, clip_threshold=5
        )

        def ortho_init(layer, gain):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

        ortho_init(policy[0], gain=1)
        ortho_init(policy[2], gain=1)
        ortho_init(policy[4], gain=1e-2)
        ortho_init(vf[0], gain=1)
        ortho_init(vf[2], gain=1)
        ortho_init(vf[4], gain=1)
    
        model = pfrl.nn.Branched(policy, vf)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        agent = pfrl.agents.PPO(
            model,
            opt,
            obs_normalizer=obs_normalizer,
            gpu=args.gpu,
            update_interval=args.update_interval,
            minibatch_size=args.batch_size,
            epochs=args.epochs,
            clip_eps_vf=None,
            entropy_coef=0,
            standardize_advantages=True,
            gamma=0.995,
            lambd=0.97,
        )
    elif args.agent == "SAC":
        if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
            raise Exception("This script requires a PyTorch version >= 1.5.0")

        def squashed_diagonal_gaussian_head(x):
            assert x.shape[-1] == action_size * 2
            mean, log_scale = torch.chunk(x, 2, dim=1)
            log_scale = torch.clamp(log_scale, -20.0, 2.0)
            var = torch.exp(log_scale * 2)
            base_distribution = distributions.Independent(
                distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
            )
            # cache_size=1 is required for numerical stability
            return distributions.transformed_distribution.TransformedDistribution(
                base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
            )
        policy = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size * 2),
            Lambda(squashed_diagonal_gaussian_head),
        )
        torch.nn.init.xavier_uniform_(policy[0].weight)
        torch.nn.init.xavier_uniform_(policy[2].weight)
        torch.nn.init.xavier_uniform_(policy[4].weight, gain=args.policy_output_scale)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(obs_size + action_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            torch.nn.init.xavier_uniform_(q_func[1].weight)
            torch.nn.init.xavier_uniform_(q_func[3].weight)
            torch.nn.init.xavier_uniform_(q_func[5].weight)
            q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        rbuf = replay_buffers.ReplayBuffer(10**6)

        agent = pfrl.agents.SoftActorCritic(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            rbuf,
            gamma=0.99,
            replay_start_size=args.replay_start_size,
            gpu=args.gpu,
            minibatch_size=args.batch_size,
            burnin_action_func=burnin_action_func,
            entropy_target=-action_size,
            temperature_optimizer_lr=3e-4,
        )
    eval_env = make_env(test=True)
    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
        import json
        import os

        with open(os.path.join(args.outdir, "demo_scores.json"), "w") as f:
            json.dump(eval_stats, f)
    else:
        """
        experiments.train_agent_batch(
            agent=agent,
            env=env,
            steps=args.steps,
            outdir=args.outdir,
            checkpoint_freq=int(int(args.steps)/20),
            train_max_episode_len=
        )
        """
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            eval_env=eval_env,
            eval_n_episodes=100,
            eval_interval=int(int(args.steps)/20),
            outdir=args.outdir,
            train_max_episode_len=timestep_limit,
        )
    with open("./manage.csv", mode="a") as f:
        f.write(f"{args.env},{args.agent},{int(args.seed)},{int(args.steps)}\n")

    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_table(f"{args.outdir}/scores.txt")
    plt.plot(df["steps"], df["mean"])
    plt.fill_between(df["steps"], df["mean"]-df["stdev"], df["mean"]+df["stdev"], alpha=0.3)
    plt.xlabel("steps")
    plt.ylabel("cumulative rewards")
    # plt.show()
    plt.savefig(f"{args.outdir}/figure.pdf")
    
   

if __name__ == "__main__":
    main()
