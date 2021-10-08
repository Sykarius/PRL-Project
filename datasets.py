import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from collections import deque
import os


class ImitationData(Dataset):

    def __init__(self, frames_path):

        self.frames_path = frames_path
        self.actions = [0, 2, 3]
        self.action_path = dict()
        self.action_files = dict()
        self.action_idx = dict()

        for a in self.actions:
            if a == 1:
                continue
            self.action_path[a] = self.frames_path + str(a) + '/'
            self.action_files[a] = [self.action_path[a] +
                                    x for x in os.listdir(self.action_path[a])]
            self.action_idx[a] = len(self.action_files[a])

        self.sample_size = min(self.action_idx.values())

        self.datapaths = []
        self.data_actions = []

        for a in self.actions:
            if a == 1:
                continue
            samples = np.random.choice(
                self.action_files[a], size=self.sample_size, replace=False)
            samples = list(samples)
            a_samples = [a]*len(samples)
            self.datapaths.extend(samples)
            self.data_actions.extend(a_samples)

        self.data_actions = np.array(self.data_actions)
        assert len(self.datapaths) == len(self.data_actions)

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):

        with open(self.datapaths[idx], 'rb') as f:
            state = pickle.load(f)
        state = state.squeeze(0)
        return state, self.data_actions[idx]


class ChannelFirst:

    def __call__(self, frame):
        # frame: 210 x 160 x 3
        frame = torch.from_numpy(frame)
        # frame: 210 x 160 x 3
        frame = frame.transpose(0, 2).contiguous()
        # frame: 3 x 160 x 210
        return frame


class FrameStack:

    def __init__(self, nframes):

        self.nframes = nframes
        self.frames = deque([], maxlen=self.nframes)

    def _getframes(self):

        assert len(self.frames) == self.nframes
        frames = torch.cat(tuple(self.frames), dim=0)
        frames = frames.unsqueeze(0)  # For batch Dimension
        return frames

    def __call__(self, frame):

        frame = frame.type(torch.FloatTensor)
        if not len(self.frames):
            for _ in range(self.nframes):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

        return self._getframes()

    def reset(self):
        self.frames.clear()
