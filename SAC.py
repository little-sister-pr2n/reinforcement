import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_space,  hidden_dim=256) -> None:
        super(GaussianPolicy, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.mu_output_layer = nn.Linear(hidden_dim, action_dim)
        self.logstd_output_layer = nn.Linear(hidden_dim, action_dim)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        mu = torch.tanh(self.mu_output_layer(x))
        logstd = self.logstd_output_layer(x)
        logstd = torch.clamp(logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, logstd

    def sample(self, state):
        mu, logstd = self(state)
        std = logstd.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mean

class DoubleQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256) -> None:
        super(DoubleQNetwork, self).__init__()
        
        self.q1_input_layer = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.q1_output_layer = nn.Linear(hidden_dim, 1)

        self.q2_input_layer = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.q2_output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        q1 = F.relu(self.q1_input_layer(x))
        q1 = F.relu(self.q1_hidden_layer(q1))
        q1 = self.q1_output_layer(q1)

        q2 = F.relu(self.q2_input_layer(x))
        q2 = F.relu(self.q2_hidden_layer(q2))
        q2 = self.q2_output_layer(q2)
        return q1, q2

class SAC:
    def __init__(self, policy:GaussianPolicy, qfunc:DoubleQNetwork, policy_optimizer, qfunc_optimizer, action_space, discount=0.99, delay=2):
        self.policy = policy
        self.qfunc = qfunc
        self.policy_optimizer = policy_optimizer
        self.qfunc_optimizer = qfunc_optimizer
        self.action_space = action_space
        self.discount = discount
        self.delay = delay
        self.tau = 0.005
        self.alpha = 0.2 # 初期パラメータ

        self.target_policy = copy.deepcopy(policy)
        self.target_qfunc = copy.deepcopy(qfunc)

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
        return 

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if not evaluate:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def train(self, replay_buffer, batch_size, step):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, log_prob, _ = self.target_policy.sample(next_state)
            q1_target, q2_target = self.target_qfunc(next_state, next_action)
            q_target = reward + not_done * self.discount * (torch.min(q1_target, q2_target) - self.alpha * log_prob)
        
        q1, q2 = self.qfunc(state, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        self.qfunc_optimizer.zero_grad()
        q_loss.backward()
        self.qfunc_optimizer.step()

        if step % self.delay == 0:
            pi, log_pi, _ = self.policy.sample(state)

            q1_pi, q2_pi = self.qfunc(state, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = ((self.alpha * log_pi) - q_pi).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
                target_param.data.copy_((1-self.tau) * target_param.data + self.tau * param.data)

            for param, target_param in zip(self.qfunc.parameters(), self.target_qfunc.parameters()):
                target_param.data.copy_((1-self.tau) * target_param.data + self.tau * param.data)
        
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()

        return self.alpha

def main():
    import gym 
    from utils import ReplayBuffer
    seed = 0
    max_steps = 1e6
    start_steps = 1e3
    batch_size = 256
    learning_rate = 3e-4
    env = gym.make("Ant-v2")

    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    
    policy = GaussianPolicy(
        env.observation_space.shape[0], 
        env.action_space.shape[0],
        env.action_space
    )
    qfunc = DoubleQNetwork(
        env.observation_space.shape[0], 
        env.action_space.shape[0]
    )
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    qfunc_optimizer = torch.optim.Adam(qfunc.parameters(), lr=learning_rate)
    agent = SAC(
        policy,
        qfunc,
        policy_optimizer,
        qfunc_optimizer,
        env.action_space
    )
    replay_buffer = ReplayBuffer(
        env.observation_space.shape[0], 
        env.action_space.shape[0],
        int(max_steps)
    )
    state = env.reset()
    for step in range(int(max_steps)):
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action) 
        replay_buffer.add(
            state, action, next_state, reward, done
        )

        if step >= start_steps:
            agent.train(replay_buffer, batch_size, step)

        if done:
            state = env.reset()
        else:
            state = next_state

        

if __name__ == "__main__":
    main()