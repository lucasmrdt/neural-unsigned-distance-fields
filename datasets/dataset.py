import torch as th
import numpy as np
from sklearn.neighbors import KDTree
from tqdm.auto import trange

import os


def voxelize(points, res):
    dim = points.shape[1]
    p_min, p_max = points.min(axis=0), points.max(axis=0)
    vox_size = (p_max - p_min) / (res-1)
    indices = np.indices((res,)*dim).reshape(dim, -1).T
    corners = indices*vox_size + p_min

    tree = KDTree(corners)
    idx = tree.query(points, k=1, return_distance=False)[:, 0]
    p_vox = np.zeros((res,)*dim)
    p_vox.flat[idx] = 1
    return p_vox


def sample_shape_from_corners(n_points, corners):
    w = np.linspace(0, 1, n_points//(len(corners)-1))
    points = np.concatenate([
        w*a[..., None] + (1-w)*b[..., None]
        for a, b in zip(corners[:-1], corners[1:])
    ], axis=1).T
    return points


def normalize_points(points):
    total_size = (points.max(axis=0) - points.min(axis=0)).max()
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points = (points - center) / total_size
    return points


def random_rotate_points(points):
    theta = np.random.uniform(0, 2*np.pi)
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    return points @ rot


class Dataset(th.utils.data.Dataset):
    def __init__(self, folder='data', mode='train'):
        idx_by_mode = np.load(f"{folder}/idx.npz")
        self.files = [f'{folder}/{idx}.npz' for idx in idx_by_mode[mode]]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return th.from_numpy(data['data']).float(), th.from_numpy(data['vox']).float(), th.from_numpy(data['dist']).float()

    @staticmethod
    def generate_dataset(n_points, res, unif_ratio, sigmas, sigmas_p, size=1000, gen_item_fct=None, folder='data', split=[.8, .1, .1]):
        assert sum(split) == 1
        assert len(split) == 3
        assert gen_item_fct is not None

        if not os.path.exists(folder):
            os.mkdir(folder)

        for i in trange(size, desc='Generating data'):
            item = gen_item_fct(n_points=n_points, res=res,
                                unif_ratio=unif_ratio, sigmas=sigmas, sigmas_p=sigmas_p)
            np.savez_compressed(f"{folder}/{i}.npz",
                                data=item['data'], vox=item['vox'], dist=item['dist'])

        idx = np.random.permutation(size)
        train_idx, test_idx, val_idx = np.split(idx,
                                                [int(size*split[0]), int(size*(split[0]+split[1]))])
        np.savez_compressed(f"{folder}/idx.npz",
                            train=train_idx, test=test_idx, val=val_idx)
