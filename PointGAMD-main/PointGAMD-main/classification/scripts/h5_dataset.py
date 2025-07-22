import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset as TorchDataset

def load_data_part_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]      # shape: [B, N, 3]
    label = f['label'][:]    # shape: [B, 1]
    return data, label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0/3.0, high=3.0/2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    pointcloud += jittered_data
    return pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.random.uniform(0, 2 * np.pi)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return pointcloud @ rot_matrix.T

class Dataset(TorchDataset):
    def __init__(self, root, dataset_name='modelnet40', split='train',
                 random_rotate=True, random_jitter=True, random_translate=False):
        self.root = root
        self.split = split.lower()
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate

        self.data, self.label = self.load_data()

    def load_data(self):
        data, label = [], []
        for file in os.listdir(self.root):
            if self.split in file and file.endswith('.h5'):
                d, l = load_data_part_h5(os.path.join(self.root, file))
                data.append(d)
                label.append(l)
        if len(data) == 0:
            raise ValueError(f"No data found in: {self.root} with split: {self.split}")
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        return data, label

    def __getitem__(self, idx):
        pc = self.data[idx]  # [N, 3]
        label = self.label[idx]

        if self.random_rotate:
            pc = rotate_pointcloud(pc)
        if self.random_jitter:
            pc = jitter_pointcloud(pc)
        if self.random_translate:
            pc = translate_pointcloud(pc)

        pc = torch.from_numpy(pc.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))
        return pc, label

    def __len__(self):
        return self.data.shape[0]

