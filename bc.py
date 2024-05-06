import gym

from torch.nn.modules import L1Loss

import tqdm

from tqdm import tqdm

import torch.nn as nn

import torch

import numpy as np

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from torch.optim import optimizer


def train(learner, observations, actions, checkpoint_path, num_epochs=100):
    """Train function for learning a new policy using BC.


    Parameters:

        learner (Learner)

            A Learner object (policy)

        observations (list of numpy.ndarray)

            A list of numpy arrays of shape (7166, 11, )

        actions (list of numpy.ndarray)

            A list of numpy arrays of shape (7166, 3, )

        checkpoint_path (str)

            The path to save the best performing checkpoint

        num_epochs (int)

            Number of epochs to run the train function for


    Returns:

        learner (Learner)

            A Learner object (policy)

    """

    best_loss = float("inf")

    best_model_state = None

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

    dataset = TensorDataset(
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )  # Create your dataset

    dataloader = DataLoader(
        dataset, batch_size=256, shuffle=True
    )  # Create your dataloader

    # TODO: Complete the training loop here ###

    for epoch in tqdm(range(num_epochs)):

        loss = 0

        num_batch = 0

        for obs, act in dataloader:

            optimizer.zero_grad()

            predictions = learner.forward(obs)

            batch_loss = loss_fn(predictions, act)

            loss += batch_loss.item() * obs.size(0)

            num_batch += 1

            batch_loss.backward()

            optimizer.step()

        loss = loss / observations.size(0)

        # Saving model state if current loss is less than best loss

        if loss < best_loss:

            best_loss = loss

            best_model_state = learner.state_dict()

    learner.load_state_dict(best_model_state)

    # Save the best performing checkpoint

    torch.save(best_model_state, checkpoint_path)

    return learner
