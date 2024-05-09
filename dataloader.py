from os import path, listdir
import numpy as np
import cv2
import torch

import math
class AtariDataset():
    '''
        Class for loading the dataset of Atari replays.
        Source: Atari Grand Challenge repo'''

    TRAJS_SUBDIR = 'trajectories/spaceinvaders'
    SCREENS_SUBDIR = 'screens/spaceinvaders'

    def __init__(self, data_path):
        
        '''
            Loads the dataset trajectories into memory. 
            data_path is the root of the dataset (the folder, which contains
            the 'screens' and 'trajectories' folders. 
        '''

        self.open_ai_mappings = {
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

        self.trajs_path = path.join(data_path, AtariDataset.TRAJS_SUBDIR)       
        self.screens_path = path.join(data_path, AtariDataset.SCREENS_SUBDIR)

        #check that the we have the trajs where expected
        assert path.exists(self.trajs_path)
        
        self.trajectories = self.load_trajectories()


    def load_trajectories(self, top = 1):
        
        # trajectories is a list of tuples (traj_id, traj_data)
        trajectories = []
        for traj in listdir(self.trajs_path):
            traj_path = path.join(self.trajs_path, traj)
            curr_traj = []
            with open(traj_path) as f:
                for i,line in enumerate(f):
                    #first line is the metadata, second is the header
                    if i > 1:
                        curr_data = line.rstrip('\n').replace(" ","").split(',')
                        curr_trans = {}
                        curr_trans['frame']    = int(curr_data[0])
                        curr_trans['reward']   = int(curr_data[1])
                        curr_trans['score']    = int(curr_data[2])
                        curr_trans['terminal'] = (curr_data[3] == "True")
                        curr_trans['action']   = self.open_ai_mappings[int(curr_data[4])]
                        curr_traj.append(curr_trans)
            trajectories.append((int(traj.split('.txt')[0]), curr_traj))
        
        #sort by max score
        trajectories.sort(key=lambda x: x[1][-1]['score'], reverse=True)

        top_trajs = dict(trajectories[:top])
        print(len(top_trajs))
        return top_trajs
                   

    def compile_data(self):
        ''' Read in screenshots to add states to the dataset.
        Returns (observations, actions)'''

        observations = []
        actions = []
        shuffled_trajs = np.array(list(self.trajectories.keys()))
        np.random.shuffle(shuffled_trajs)

        for t in shuffled_trajs:
            st_dir   = path.join(self.screens_path, str(t))
            cur_traj = self.trajectories[t]
            cur_traj_len = len(listdir(st_dir))

            for pid in range(0, cur_traj_len):

                # Load the state in color
                state = (cv2.imread(path.join(st_dir, str(pid) + '.png'), cv2.IMREAD_GRAYSCALE)).flatten()

                assert state.shape == (210*160,)

                observations.append(state)
                actions.append(cur_traj[pid]['action'])

        return observations, actions