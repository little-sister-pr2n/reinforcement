import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256) -> None:
        super(GaussianPolicy, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.mu_output_layer = nn.Linear(hidden_dim, action_dim)
        self.logstd_output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        mu = torch.tanh(self.mu_output_layer(x))
        logstd = self.logstd_output_layer(x)

    def select_action(self, state):

        return action

class DoubleQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim) -> None:
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
    def __init__(self, policy, qfunc1, qfunc2, discount=0.99):
        self.policy = policy
        self.qfunc1 = qfunc1
        self.qfunc2 = qfunc2
        self.discount = discount
        return 

    def train(self, replay_buffer, batch_size):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        loss = reward + self.discount * 

    def __update_policy(self):
        pass

    def __update_Qfunc(self):
        return 