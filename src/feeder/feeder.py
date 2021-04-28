# sys

import os
import sys
import pickle
import random
import numpy as np

# torch
import torch
from torch.utils.data import Dataset


# local
import tools


class Feeder(Dataset):
    """
    Data Feeder for skeleton data
    """
    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.debug = debug
        self.mmap = mmap

        # data shape : N,C,T,V,M

        # load label
        with open(self.label_path, 'rb') as fp:
            self.sample_name, self.sample_label = pickle.load(fp)

        # load data

        if mmap:
            # use mmap mode
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # debug mode
        if debug:
            self.sample_label = self.sample_label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return  len(self.sample_name)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.sample_label[index]

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size>0:
            data_numpy = tools.auto_pading(data_numpy, self.random_move)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label


if __name__ == '__main__':
    data_path = "../../data/NTU-RGB-D/xview/eval_data.npy"
    label_path = '../../data/NTU-RGB-D/xview/eval_label.pkl'
    video_id = 'S003C001P017R001A044'

    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(dataset=Feeder(data_path=data_path, label_path=label_path),
                                         batch_size=64,
                                         shuffle=False,
                                         num_workers=2)

    sample_name = loader.dataset.sample_name
    sample_id = [name.split('.')[0] for name in sample_name]
    index = sample_id.index(video_id)
    data, label = loader.dataset[index]
    data = data.reshape((1,) + data.shape)

    N, C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pose, = ax.plot(np.zeros(V * M), np.zeros(V * M), 'g^')
    ax.axis([-1, 1, -1, 1])

    for n in range(N):
        for t in range(T):
            x = data[n, 0, t, :, 0]
            y = data[n, 1, t, :, 0]
            z = data[n, 2, t, :, 0]
            pose.set_xdata(x)
            pose.set_ydata(y)
            fig.canvas.draw()
            plt.pause(1)

    plt.ioff()