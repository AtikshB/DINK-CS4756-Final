import tqdm
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def train(learner, observations, actions, checkpoint_path, num_epochs=100):
    print("Training the learner")
    best_loss = float("inf")
    best_model_state = None
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(learner.parameters(), lr=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(
        torch.tensor(np.array(observations), dtype=torch.float32, device=device),
        torch.tensor(np.array(actions), dtype=torch.long, device=device),
    )  # Create your dataset
    dataloader = DataLoader(
        dataset, batch_size=256, shuffle=True
    )  # Create your dataloader

    print(f"Training for {num_epochs} epochs")
    for epoch in tqdm(range(num_epochs)):
        loss = 0
        num_batch = 0
        for obs, act in dataloader:
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
