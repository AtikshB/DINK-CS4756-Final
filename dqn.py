import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm
import random


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, gamma=0.99):
        """
        Initialize the DQN model.

        Args:
            input_shape (tuple): Shape of the input state (without batch dimension).
            num_actions (int): Number of possible actions.
            gamma (float): Discount factor (default: 0.99).
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.gamma = gamma

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, 1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (tensor): Input state tensor.

        Returns:
            tensor: Q-values for each action.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_max_q(self, states):
        """
        Get the maximum Q-value for a batch of states.

        Args:
            states (tensor): Batch of input states.

        Returns:
            tensor: Maximum Q-values for each state.
        """
        q_vals = self.forward(states)
        max_qs, _ = torch.max(q_vals, dim=1)
        return max_qs

    def get_eps(self, eps_param, t):
        """
        Get the epsilon value for epsilon-greedy action selection.

        Args:
            eps_param (float): Initial epsilon value.
            t (int): Time step.

        Returns:
            float: Epsilon value.
        """
        eps = eps_param**t
        return max(0.001, eps)

    def get_action(self, state, eps):
        """
        Get action using epsilon-greedy policy.

        Args:
            state (tensor): Input state tensor.
            eps (float): Epsilon value for exploration.

        Returns:
            int: Selected action.
        """
        state = state.float()
        q_values = self.forward(state)
        if torch.rand(1).item() < eps:
            return torch.randint(0, q_values.size(1), (1,)).item()
        else:
            return q_values.argmax().item()

    @torch.no_grad()
    def get_targets(self, rewards, next_states, dones):
        """
        Get target Q-values for training.

        Args:
            rewards (tensor): Batch of rewards.
            next_states (tensor): Batch of next states.
            dones (tensor): Batch of done flags.

        Returns:
            tensor: Target Q-values.
        """
        next_max_qs = self.get_max_q(next_states)
        target_q_vals = rewards + (1 - dones) * self.gamma * next_max_qs
        return target_q_vals.float()


def train(
    network,
    env,
    observations,
    actions,
    rewards,
    next_observations,
    dones,
    save_path,
    batch_size=128,
    num_episodes=100,
    lr=1e-3,
    add_data_every=4,
):
    optimizer = optim.Adam(network.parameters(), lr=lr)
    data = list(zip(observations, actions, rewards, next_observations, dones))

    for i in tqdm(range(num_episodes)):
        # Add new data to the dataset after the first epoch and every 'add_data_every' epochs
        if i == 0 or (i % add_data_every == 0):
            new_data = collect_data(network, env, num_episodes=1)  # Collect new data
            data.extend(new_data)  # Add new data to the dataset

        # Shuffle the dataset before each epoch
        random.shuffle(data)

        # Mini-batch training
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start : batch_start + batch_size]
            obs_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = (
                zip(*batch)
            )
            obs_batch = torch.tensor(obs_batch)
            actions_batch = torch.tensor(actions_batch)
            rewards_batch = torch.tensor(rewards_batch)
            next_states_batch = torch.tensor(next_states_batch)
            dones_batch = torch.tensor(dones_batch)

            q_vals = network(obs_batch)[torch.arange(len(batch)), actions_batch]
            target_q_values = network.get_targets(
                rewards_batch, next_states_batch, dones_batch
            )
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(q_vals, target_q_values)
            loss.backward()
            optimizer.step()

    # Save final agent
    torch.save(network, save_path)


# Example function to collect new data
def collect_data(network, env, num_episodes=1):
    new_data = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = network.get_action(
                torch.tensor(obs).unsqueeze(0), eps=0.0
            )  # Greedy action
            next_obs, reward, done, _ = env.step(action)
            new_data.append((obs, action, reward, next_obs, done))
            obs = next_obs
    return new_data
