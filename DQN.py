import torch
from torch import nn
from torch import optim
import numpy as np
import tqdm

class DQN(nn.Module):
    def __init__(self, gamma, state_dim, action_dim, hidden_sizes=[10, 10]):
        super().__init__()
        self.gamma = gamma

        # neural net architecture
        self.network = self.make_network(state_dim, action_dim, hidden_sizes)

    def forward(self, states):
        qs = self.network(states)
        return qs

    def get_max_q(self, states):
        q_vals = self.forward(states)
        max_qs, max_indices = torch.max(q_vals, dim=1)
        return max_qs

    def get_eps(self, eps_param, t):
        eps = eps_param**t
        if eps <= 0.001:
            return 0.001
        return eps

    def get_action(self, state, eps):
        q_values = self.forward(state)
        q_max = q_values.argmax().item()
        d = q_values.shape[0]
        action = np.random.choice([np.random.randint(d), q_max], p=[eps, 1 - eps])
        return action.item()

    @torch.no_grad()
    def get_targets(self, rewards, next_states, dones):

        next_max_qs = self.get_max_q(next_states)

        target_q_vals = (
            rewards.flatten()
            + (torch.ones(dones.shape) - dones) * self.gamma * next_max_qs
        ).float()
        return target_q_vals

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)

    def make_network(self, state_dim, action_dim, hidden_sizes):
        layers = []
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))

        network = nn.Sequential(*layers).apply(self.initialize_weights)
        return network

    
