import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, dim_in, dim_out, action_space) -> None:
        super(Actor, self).__init__()

        # define layer
        self.l1 = nn.Linear(dim_in, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, dim_out)

        # action_space is like [[min], [max]] : shape as (2 x dim_in) 
        self.a1 = 1/2 * (action_space[1] + action_space[0])
        self.a2 = 1/2 * (action_space[1] - action_space[0])

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return self.a1 + self.a2 * x

class Critic(nn.Module):
    def __init__(self, dim_in) -> None:
        super(Critic, self).__init__()

        # define layer
        self.l1 = nn.Linear(dim_in, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class SAC:
    def __init__(self):
        pass

