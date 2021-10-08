import gym
import os
import gym.utils.play
import pickle
import numpy as np
from torchvision import transforms
from datasets import ChannelFirst, FrameStack


class Callback:

    def __init__(self):

        self.preprocess = transforms.Compose([
            ChannelFirst(),
            transforms.Resize((84, 84)),
            transforms.Grayscale(),
            FrameStack(4)
        ])

        self.frames_path = './frames/stack'
        #self.folders = os.listdir(self.frames_path)
        self.actions = [0, 1, 2, 3]
        self.action_path = dict()
        self.action_idx = dict()
        for a in self.actions:
            self.action_path[a] = self.frames_path + str(a) + '/'
            self.action_idx[a] = len(os.listdir(self.action_path[a]))

    def __call__(self, obs_t, obs_tp1, action, rew, done, info):
        '''
        callback: lambda or None
            Callback if a callback is provided it will be executed after
            every step. It takes the following input:
                obs_t: observation before performing action
                obs_tp1: observation after performing action
                action: action that was executed
                rew: reward that was received
                done: whether the environment is done or not
                info: debug info
        '''

        obs_t = self.preprocess(np.array(obs_t))
        with open(self.action_path[action]+str(self.action_idx[action])+'_'+str(action)+'.pkl', 'wb') as f:
            pickle.dump(obs_t, f)
        if done:
            self.preprocess.transforms[-1].reset()
        self.action_idx[action] += 1


# Easier to play
env = gym.make('BreakoutNoFrameskip-v4')
env = gym.wrappers.Monitor(
    env, './videos', video_callable=lambda x: True, force=True)  # force=True

while True:
    gym.utils.play.play(env, zoom=3, callback=Callback(), fps=30)
