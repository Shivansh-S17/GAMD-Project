import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data

"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
work : https://github.com/antao97/dgcnn.pytorch/tree/master?tab=readme-ov-file#point-cloud-semantic-segmentation-on-the-s3dis-dataset
dataset : https://github.com/antao97/PointCloudDatasets/blob/master/dataset.py
"""

shapenetpart_cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12,
                                16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(
        pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix)  # random rotation (x,z)
    return pointcloud


class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', class_choice=None,
                 num_points=2048, split='train', load_name=True, load_file=True,
                 segmentation=False, random_rotate=False, random_jitter=False,
                 random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart',
                                        'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name not in ['shapenetpart'] and segmentation == True:
            raise AssertionError

        self.root = os.path.join(root, dataset_name + '_hdf5_2048')
        self.dataset_name = dataset_name
        self.class_choice = class_choice
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate

        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train', 'trainval', 'all']:
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val', 'trainval', 'all']:
                self.get_path('val')
        if self.split in ['test', 'all']:
            self.get_path('test')

        data, label, seg = self.load_h5py(self.path_h5py_all)
        print(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.name = np.array(self.load_json(
                self.path_name_all))    # load label name

        if self.load_file:
            self.file = np.array(self.load_json(
                self.path_file_all))    # load file name

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)
        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0)

        if self.class_choice != None:
            indices = (self.name == class_choice)
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.name = self.name[indices]
            if self.segmentation:
                self.seg = self.seg[indices]
                id_choice = shapenetpart_cat2id[class_choice]
                self.seg_num_all = shapenetpart_seg_num[id_choice]
                self.seg_start_index = shapenetpart_seg_start_index[id_choice]
            if self.load_file:
                self.file = self.file[indices]
        elif self.segmentation:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '%s*.h5' % type)
        paths = glob(path_h5py)
        paths_sort = [os.path.join(self.root, type + str(i) + '.h5')
                      for i in range(len(paths))]
        self.path_h5py_all += paths_sort
        if self.load_name:
            paths_json = [os.path.join(
                self.root, type + str(i) + '_id2name.json') for i in range(len(paths))]
            self.path_name_all += paths_json
        if self.load_file:
            paths_json = [os.path.join(
                self.root, type + str(i) + '_id2file.json') for i in range(len(paths))]
            self.path_file_all += paths_json
        return

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name
        if self.load_file:
            file = self.file[item]  # get file name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        
        label = label.squeeze(0)

        if self.segmentation:
            seg = self.seg[item]
            seg = torch.from_numpy(seg)
            return point_set, label, seg
        else:
            return point_set, label
        
    def __len__(self):
        return self.data.shape[0]


class _S3DISDataset(Dataset):
    def __init__(self, root, num_points, split='train', with_normalized_coords=True, holdout_area=5):
        """
        :param root: directory path to the s3dis dataset
        :param num_points: number of points to process for each scene
        :param split: 'train' or 'test'
        :param with_normalized_coords: whether include the normalized coords in features (default: True)
        :param holdout_area: which area to holdout (default: 5)
        """
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.num_points = num_points
        self.holdout_area = None if holdout_area is None else int(holdout_area)
        self.with_normalized_coords = with_normalized_coords
        # keep at most 20/30 files in memory
        self.cache_size = 20 if split == 'train' else 30
        self.cache = {}

        # mapping batch index to corresponding file
        areas = []
        if self.split == 'train':
            for a in range(1, 7):
                if a != self.holdout_area:
                    areas.append(os.path.join(self.root, f'Area_{a}'))
        else:
            areas.append(os.path.join(self.root, f'Area_{self.holdout_area}'))

        self.num_scene_windows, self.max_num_points = 0, 0
        index_to_filename, scene_list = [], {}
        filename_to_start_index = {}
        for area in areas:
            area_scenes = os.listdir(area)
            area_scenes.sort()
            for scene in area_scenes:
                current_scene = os.path.join(area, scene)
                scene_list[current_scene] = []
                for split in ['zero', 'half']:
                    current_file = os.path.join(current_scene, f'{split}_0.h5')
                    filename_to_start_index[current_file] = self.num_scene_windows
                    h5f = h5py.File(current_file, 'r')
                    num_windows = h5f['data'].shape[0]
                    self.num_scene_windows += num_windows
                    for i in range(num_windows):
                        index_to_filename.append(current_file)
                    scene_list[current_scene].append(current_file)
        self.index_to_filename = index_to_filename
        self.filename_to_start_index = filename_to_start_index
        self.scene_list = scene_list

    def __len__(self):
        return self.num_scene_windows

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        if filename not in self.cache.keys():
            h5f = h5py.File(filename, 'r')
            scene_data = h5f['data']
            scene_label = h5f['label_seg']
            scene_num_points = h5f['data_num']
            if len(self.cache.keys()) < self.cache_size:
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
            else:
                victim_idx = np.random.randint(0, self.cache_size)
                cache_keys = list(self.cache.keys())
                cache_keys.sort()
                self.cache.pop(cache_keys[victim_idx])
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
        else:
            scene_data, scene_label, scene_num_points = self.cache[filename]

        internal_pos = index - self.filename_to_start_index[filename]
        current_window_data = np.array(scene_data[internal_pos]).astype(np.float32)
        current_window_label = np.array(scene_label[internal_pos]).astype(np.int64)
        current_window_num_points = scene_num_points[internal_pos]

        choices = np.random.choice(current_window_num_points, self.num_points,
                                   replace=(current_window_num_points < self.num_points))
        data = current_window_data[choices, ...].transpose()
        label = current_window_label[choices]
        # data[9, num_points] = [x_in_block, y_in_block, z_in_block, r, g, b, x / x_room, y / y_room, z / z_room]
        if self.with_normalized_coords:
            return data, label
        else:
            return data[:-3, :], label


class S3DIS(dict):
    def __init__(self, root, num_points, split=None, with_normalized_coords=True, holdout_area=5):
        super().__init__()
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = _S3DISDataset(root=root, num_points=num_points, split=s,
                                    with_normalized_coords=with_normalized_coords, holdout_area=holdout_area)
if __name__ == '__main__':
    root = os.getcwd()

    # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
    dataset_name = 'shapenetcorev2'

    # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
    # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
    split = 'train'

    d = Dataset(root=root, dataset_name=dataset_name,
                num_points=2048, split=split)
    print("datasize:", d.__len__())

    item = 0
    ps, lb, n, f = d[item]
    print(ps.size(), ps.type(), lb.size(), lb.type(), n, f)

