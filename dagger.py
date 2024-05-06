import bc

import torch


def interact(
    env, learner, expert, observations, actions, checkpoint_path, seed, num_epochs=100
):
    """Interact with the environment and update the learner policy using DAgger.



    This function interacts with the given Gym environment and aggregates to

    the BC dataset by querying the expert.



    Parameters:

        env (Env)

            The gym environment (in this case, the Hopper gym environment)

        learner (Learner)

            A Learner object (policy)

        expert (ExpertActor)

            An ExpertActor object (expert policy)

        observations (list of numpy.ndarray)

            An initially empty list of numpy arrays

        actions (list of numpy.ndarray)

            An initially empty list of numpy arrays

        checkpoint_path (str)

            The path to save the best performing model checkpoint

        seed (int)

            The seed to use for the environment

        num_epochs (int)

            Number of epochs to run the train function for

    """

    # Interact with the environment and aggregate your BC Dataset by querying the expert

    NUM_INTERACTIONS = 50  # changed from ED #140

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for episode in range(NUM_INTERACTIONS):

        total_learner_reward = 0

        done = False

        obs = env.reset(seed=seed)

        episode_observations = []

        expert_actions = []

        while not done:

            # TODO: Implement Hopper environment interaction and dataset aggregation here

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
