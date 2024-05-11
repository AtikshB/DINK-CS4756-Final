import bc
import torch
from expert.stack_frame import stack_frames


def interact(
    env,
    learner,
    expert,
    observations,
    actions,
    checkpoint_path,
    seed,
    num_epochs=100,
    tqdm_disable=False,
):

    NUM_INTERACTIONS = 25
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for episode in range(NUM_INTERACTIONS):
        total_learner_reward = 0
        done = False
        obs = env.reset(seed=seed)
        obs_proc = stack_frames(None, obs, True)
        episode_observations = []
        expert_actions = []
        while not done:
            with torch.no_grad():
                action = learner.get_action(torch.Tensor([obs]).to(device))
                expert_actions.append(expert.act(obs_proc)[0])
            episode_observations.append(obs.flatten())
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            obs_proc = stack_frames(obs_proc, next_obs, False)
            total_learner_reward += reward
            if done:
                break
        observations.extend(episode_observations)
        actions.extend(expert_actions)
        print(f"After interaction {episode}, reward = {total_learner_reward}")
        bc.train(
            learner,
            observations,
            actions,
            checkpoint_path,
            num_epochs,
            tqdm_disable,
        )
