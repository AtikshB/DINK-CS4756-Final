import os
from collections import defaultdict

class DataLoader:
    '''Class to load trajectories from a folder.'''

    def __init__(self, dir):
        self.dir = dir
        self.trajectories = []
        pass

    def load_trajectories(self, open_ai = False) -> list:
        '''Load trajectories into memory.

        open_ai (bool): If True, the action space maps to OpenAI Gym's action space.
        
        Returns: list of tuples (frame, reward, score, terminal, action)'''
        # Read all text files from folder
        files = os.listdir(self.dir)

        count = 0
        action_space = defaultdict(int)

        open_ai_mappings = {
            0: 0,
            1: 1,
            3: 2,
            4: 3,
            11: 4,
            12: 5,
            2: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            13: 0,
            16: 0
        }

        for file in files:
            file_path = os.path.join(self.dir, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()[2:]
                trajectory = []
                for line in lines:
                    frame, reward, score, terminal, action = line.strip().split(",")
                    terminal_bool = terminal == 'True'

                    if open_ai:
                        action = open_ai_mappings[int(action)]
                    else:
                        action = int(action)

                    trajectory.append((int(frame), int(reward), int(score), terminal_bool, int(action)))
                    action_space[int(action)] += 1

                self.trajectories.append(trajectory)
            count += 1
        
        print("Trajectories loaded: " + str(count))
        print("Action space: " + str(len(action_space)))

    def get_trajectories(self):
        return self.trajectories
