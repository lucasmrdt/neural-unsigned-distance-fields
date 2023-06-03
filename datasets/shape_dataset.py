import numpy as np
from sklearn.neighbors import KDTree

from .dataset import Dataset, normalize_points, random_rotate_points, sample_shape_from_corners, voxelize

square_corners = np.array([
    np.array([-1, -1]),
    np.array([-1, 1]),
    np.array([1, 1]),
    np.array([1, -1]),
    np.array([-1, -1]),
])

complex_quad_corners = np.array([
    np.array([-1, -1]),
    np.array([-1, 1]),
    np.array([1, -1]),
    np.array([1, 1]),
    np.array([-1, -1]),
])

triangle_corners = np.array([
    np.array([np.cos(0), np.sin(0)]),
    np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)]),
    np.array([np.cos(4*np.pi/3), np.sin(4*np.pi/3)]),
    np.array([np.cos(0), np.sin(0)]),
])

trapezoid_corners = np.array([
    np.array([-1, -1]),
    np.array([-1, 1]),
    np.array([0, 1]),
    np.array([1, -1]),
    np.array([-1, -1]),
])

regular_pentagon_corners = np.array([
    np.array([np.cos(0), np.sin(0)]),
    np.array([np.cos(2*np.pi/5), np.sin(2*np.pi/5)]),
    np.array([np.cos(4*np.pi/5), np.sin(4*np.pi/5)]),
    np.array([np.cos(6*np.pi/5), np.sin(6*np.pi/5)]),
    np.array([np.cos(8*np.pi/5), np.sin(8*np.pi/5)]),
    np.array([np.cos(0), np.sin(0)]),
])

regular_hexagon_corners = np.array([
    np.array([np.cos(0), np.sin(0)]),
    np.array([np.cos(2*np.pi/6), np.sin(2*np.pi/6)]),
    np.array([np.cos(4*np.pi/6), np.sin(4*np.pi/6)]),
    np.array([np.cos(6*np.pi/6), np.sin(6*np.pi/6)]),
    np.array([np.cos(8*np.pi/6), np.sin(8*np.pi/6)]),
    np.array([np.cos(10*np.pi/6), np.sin(10*np.pi/6)]),
    np.array([np.cos(0), np.sin(0)]),
])

cross_corners = np.array([
    np.array([-1, -1]),
    np.array([1, 1]),
    np.array([0, 0]),
    np.array([-1, 1]),
    np.array([1, -1]),
])

shapes = [
    square_corners,
    complex_quad_corners,
    triangle_corners,
    trapezoid_corners,
    regular_pentagon_corners,
    regular_hexagon_corners,
    # cross_corners, only for evaluation (not for training)
]


def sample_random_shape(n_points, b_min=-.5, b_max=.5):
    side = b_max - b_min
    center = (b_min + b_max) / 2

    idx = np.random.randint(0, len(shapes))
    points = sample_shape_from_corners(n_points, shapes[idx])
    points = random_rotate_points(points)
    points = normalize_points(points)*side + center
    corners = normalize_points(shapes[idx])*side + center
    return points, corners


class Shape2D(Dataset):
    def __init__(self, folder='data', mode='train'):
        super().__init__(folder, mode)

    @staticmethod
    def generate_open_shape(n_points=100, res=32, unif_ratio=.8, sigmas=[0.8, 0.02, 0.003, 0], sigmas_p=[.1, .3, .3, .3]):
        n_point_in_shape = int(n_points * (1-unif_ratio))

        # (n_point_in_shape, 2 dimensions) in [-1, 1]
        gt_data, corners = sample_random_shape(n_point_in_shape,
                                               b_min=-1, b_max=1)

        sig = np.random.choice(sigmas, size=len(gt_data), p=sigmas_p)
        noisy_data = gt_data + np.random.randn(*gt_data.shape)*sig[..., None]

        cloud = 2*np.random.rand(n_points - len(gt_data), 2) - 1  # in [-1, 1]

        data = np.concatenate([noisy_data, cloud], axis=0)

        tree = KDTree(gt_data)
        ind = tree.query(data, k=1, return_distance=False)
        dist = np.linalg.norm(data - gt_data[ind.squeeze()], axis=1)

        vox = voxelize(noisy_data, res)

        return {
            'data': data,
            'vox': vox,
            'dist': dist,
            'gt_data': gt_data,
            'noisy_data': noisy_data,
            'corners': corners,
        }

    @staticmethod
    def generate_dataset(**kwargs):
        Dataset.generate_dataset(
            **kwargs, gen_item_fct=Shape2D.generate_open_shape)
