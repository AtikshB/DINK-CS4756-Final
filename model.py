import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class SpaceInvLearner(nn.Module):
    def __init__(self, env, hidden_dim=256, random_prob=0.0):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, np.prod(env.action_space.shape))

        self.env = env
        self.random_prob = random_prob

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.tanh(self.fc_out(x))
        return out

    def get_action(self, obs):
        if np.random.random() < self.random_prob:
            return self.env.action_space.sample()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        action = self.forward((torch.Tensor([obs]).float()).to(device))
        return np.array(action[0].detach().cpu())
