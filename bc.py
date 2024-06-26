import tqdm
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import random


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
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        out = F.tanh(self.fc_out(x))
        return out

    def get_action(self, obs):
        action = self.forward(obs)
        return np.array(action.cpu().detach().argmax())


def train(
    learner, observations, actions, checkpoint_path, num_epochs=100, tqdm_disable=False
):
    print("Training the learner")
    torch.cuda.empty_cache()
    best_loss = float("inf")
    best_model_state = None
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(learner.parameters(), lr=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = list(zip(observations, actions))
    batch_size = 128
    print(f"Training for {num_epochs} epochs")
    for epoch in tqdm(range(num_epochs), disable=tqdm_disable):
        loss = 0
        num_batch = 0
        random.shuffle(data)
        # Mini-batch training
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start : batch_start + batch_size]
            obs_batch, actions_batch = zip(*batch)
            obs = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=device)
            act = torch.tensor(np.array(actions_batch), dtype=torch.long, device=device)
            optimizer.zero_grad()
            predictions = learner.forward(obs)
            action = torch.zeros((act.shape[0], 6), device=device)
            action[torch.arange(act.shape[0]), act] = 1
            batch_loss = loss_fn(predictions, action)
            loss += batch_loss.item() * obs.size(0)
            num_batch += 1
            batch_loss.backward()
            optimizer.step()
        loss = loss / len(observations)
        # Saving model state if current loss is less than best loss
        print(f"Epoch {epoch}, Loss: {loss}")
        if loss < best_loss:
            best_loss = loss
            best_model_state = learner.state_dict()
    learner.load_state_dict(best_model_state)
    # Save the best performing checkpoint
    torch.save(best_model_state, checkpoint_path)

    return learner
