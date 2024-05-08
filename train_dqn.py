import torch
from torch import nn
from torch import optim
import numpy as np
import tqdm


def train(network, env, data, save_path, batch_size=128, num_episodes=100, lr=1e-3):
    optimizer = optim.Adam(network.parameters(), lr=lr)

    # training
    for i in tqdm(range(num_episodes)):
        for obs, actions, rewards, next_states, dones in data:
            q_vals = network(obs)[torch.arange(batch_size), actions]
            target_q_values = network.get_targets(rewards, next_states, dones)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(q_vals, target_q_values)
            loss.backward()
            optimizer.step()

    # save final agent
    torch.save(network, save_path)
