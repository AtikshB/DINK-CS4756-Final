import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class SpaceInvLearner(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod(), hidden_dim, device=device
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.fc_out = nn.Linear(hidden_dim, env.action_space.n, device=device)

        self.env = env

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.tanh(self.fc_out(x))
        return out

    def get_action(self, obs):
        action = self.forward(obs)
        return np.array(action.cpu().detach().argmax())
