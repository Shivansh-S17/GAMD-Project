import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import random
from scipy.spatial.transform import Rotation
import h5py


def rotate_point_cloud(point_cloud):
    # Randomly choose an angle for rotation along the up-axis (y-axis)
    angle = random.uniform(0, 2 * np.pi)

    # Create a rotation matrix around the up-axis
    rotation_matrix = Rotation.from_euler('z', angle).as_matrix()

    # Apply the rotation to the entire point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)

    return rotated_point_cloud


def jitter_point_cloud(point_cloud, sigma=0.02):
    # Add Gaussian noise with zero mean and the specified standard deviation to each point
    jittered_point_cloud = point_cloud + \
        np.random.normal(0, sigma, point_cloud.shape)

    return jittered_point_cloud


class ModelNet_dataset(Dataset):
    def __init__(self, data_root, transform=False, train=True, conv=False):
        super(ModelNet_dataset, self).__init__()
        self.class_dir = glob.glob(data_root+'/'+'*')  # Class names
        self.data = []
        self.class_details = []
        self.class_len = []
        self.class_name = []
        self.train_bool = train
        self.transform = transform
        # Loading array of [img_path, img_class]
        if (data_root[-3:-1] == '40'):
            self.class_dir.remove(data_root+'metadata_modelnet40.csv')

        i = 0
        if train:
            for object in self.class_dir:
                img_paths = glob.glob(object + '/train/npy/*.npy')
                class_name = object.split("/")[-1]
                sample_len = 0
                for img in img_paths:
                    self.data.append([img, i])
                    sample_len = sample_len+1
                self.class_details.append([class_name, i, sample_len])
                self.class_name.append(class_name)
                self.class_len.append(sample_len)
                i = i+1
        else:
            for object in self.class_dir:
                img_paths = glob.glob(object + '/test/npy/*.npy')
                class_name = object.split("/")[-1]
                sample_len = 0
                for img in img_paths:
                    self.data.append([img, i])
                    sample_len = sample_len+1
                self.class_details.append([class_name, i, sample_len])
                self.class_name.append(class_name)
                self.class_len.append(sample_len)
                i = i+1
        if conv:  # if model uses conv instead of linear
            self.transpose = True
        else:
            self.transpose = False

    def class_detail(self):
        return self.class_details

    def label_len(self):
        return self.class_len

    def __len__(self):
        return len(self.data)

    def train(self) -> bool:
        return self.train_bool

    def __getitem__(self, index):

        point_cloud = np.load(self.data[index][0])

        if self.transform and random.choice([True, False]):
            point_cloud = jitter_point_cloud((rotate_point_cloud(point_cloud)))

        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        if self.transpose:
            point_cloud = point_cloud.transpose(0, 1)
        label = self.data[index][1]

        return point_cloud, label


class ShapeObj_dataset(Dataset):
    def __init__(self, data_root, transform=False, train=True, conv=False):
        super(ShapeObj_dataset, self).__init__()
        if transform and train:
            file = h5py.File(
                data_root+'/training_objectdataset_augmentedrot_scale75.h5', 'r')
        elif train:
            file = h5py.File(data_root+'/training_objectdataset.h5', 'r')

        elif transform:
            file = h5py.File(
                data_root+'/test_objectdataset_augmentedrot_scale75.h5', 'r')
        else:
            file = h5py.File(
                data_root+'/test_objectdataset.h5', 'r')

        self.data = file['data'][:]
        self.labels = file['label'][:]
        self.class_size = np.array([])

        for i in np.unique(self.labels):
            self.class_size = np.append(
                self.class_size, np.count_nonzero(self.labels == i))

        self.train_bool = train
        if conv:  # if model uses conv instead of linear
            self.transpose = True
        else:
            self.transpose = False

    def label_len(self):
        return self.class_size

    def __len__(self):
        return len(self.data)

    def train(self) -> bool:
        return self.train_bool

    def __getitem__(self, index):

        point_cloud = self.data[index]
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))

        if self.transpose:
            point_cloud = point_cloud.transpose(0, 1)
        label = self.labels[index]
        label = torch.from_numpy(np.asarray(label))
        label = label.type('torch.LongTensor')
        label = label.unsqueeze(0)
        return point_cloud, label

if __name__ =='main':
    dataset1 = ModelNet_dataset('dataset/modelnet10')
    dataset2 = ShapeObj_dataset('dataset/shapeobj')

