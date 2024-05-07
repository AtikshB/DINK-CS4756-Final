import bc
import torch

def interact(
    env, learner, expert, observations, actions, checkpoint_path, seed, num_epochs=100
):

    NUM_INTERACTIONS = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for episode in range(NUM_INTERACTIONS):
        total_learner_reward = 0
        done = False
        obs = env.reset(seed=seed)
        episode_observations = []
        expert_actions = []
        while not done:
            with torch.no_grad():
                action = learner.get_action(obs)
                expert_actions.append(expert.get_expert_action(obs))
            episode_observations.append(obs)
            obs, reward, done, _ = env.step(action)
            total_learner_reward += reward
            if done:
                break
        observations.extend(episode_observations)
        actions.extend(expert_actions)
        print(f"After interaction {episode}, reward = {total_learner_reward}")
        bc.train(
            learner,
            torch.Tensor(observations).to(device),
            torch.Tensor(actions).to(device),
            checkpoint_path,
            num_epochs,
        )
