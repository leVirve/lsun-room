import os

import numpy as np
import scipy.io as sio

import cv2
from datasets.lsun_room.label import mapping_func
from datasets.lsun_room.utils import load_image, load_mat, save_image


def load_mat(path):
    mat = sio.loadmat(path)
    mat = {k: v for k, v in mat.items() if not k.startswith('__')}
    return list(mat.values())[0]


def load_image(path):
    return cv2.imread(path)


def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.imwrite(path, img)


class Item():

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._image = None
        self._layout = None
        self.image_path = self.dataset.image % self.name
        self.layout_path = self.dataset.layout % self.name
        self.layout_mat_path = self.dataset.layout_mat % self.name

    @property
    def image(self):
        if self._image is None:
            self._image = load_image(self.image_path)
        return self._image

    @property
    def layout(self):
        if self._layout is None:
            self._layout = load_mat(self.layout_mat_path)
        return self._layout

    def remap_layout(self, verbose=False):
        mapping = mapping_func(self.type)(self)

        old_layout = load_mat(self.layout_mat_path)
        layout = np.zeros_like(old_layout)
        for new_label, point in mapping:
            old_label = old_layout[point[1], point[0]]
            layout[old_layout == old_label] = new_label
        return layout

    def save_layout(self, visualization=False):
        save_image(self.layout_path, self.layout)

    def __str__(self):
        return '<DataItem: %s>' % self.name


class DataItems():

    def __init__(self, root, phase):
        self.root = root
        self.image = os.path.join(root, 'images/', '%s.jpg')
        self.layout = os.path.join(root, 'layout_seg_images/', '%s.png')
        self.layout_mat = os.path.join(root, 'layout_seg/', '%s.mat')
        self.items = self._load(phase)

    def _load(self, phase):
        phase_map = {
            'train': 'training',
            'val': 'validation',
            'test': 'testing'
        }
        assert phase in phase_map.keys()

        path = os.path.join(self.root, '%s.mat' % phase_map[phase])
        meta = load_mat(path)[0]
        return [
            Item(
                name=m[0][0], scene=m[1][0], type=m[2][0][0],
                points=m[3], resolution=m[4][0],
                dataset=self)
            for m in meta
        ]
