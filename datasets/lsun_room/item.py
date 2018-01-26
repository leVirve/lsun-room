import os
import numpy as np

from datasets.lsun_room.label import mapping_func
from datasets.lsun_room.utils import load_mat, load_image, save_image


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
            if os.path.exists(self.layout_path):
                self._layout = load_image(self.layout_path)[:, :, :1]
            else:
                self._layout = np.expand_dims(load_mat(self.layout_mat_path), axis=2)
        return self._layout

    def remap_layout(self):
        mapping = mapping_func(self.type)(self)

        old_layout = np.array(self.layout)
        for new_label, point in mapping:
            old_label = old_layout[point[1], point[0]]
            self._layout[old_layout == old_label] = new_label

    def save_layout(self, visualization=False):
        save_image(self.layout_path, self.layout)

    def __str__(self):
        return '<DataItem: %s>' % self.name


class DataItems():

    def __init__(self, root_dir, phase):
        self.root_dir = root_dir
        self.image = os.path.join(root_dir, 'images/', '%s.jpg')
        self.layout = os.path.join(root_dir, 'layout_seg_images/', '%s.png')
        self.layout_mat = os.path.join(root_dir, 'layout_seg/', '%s.mat')
        self.items = self._load(phase)

    def _load(self, phase):
        phase_map = {
            'train': 'training',
            'val': 'validation',
            'test': 'testing'
        }
        assert phase in phase_map.keys()

        path = os.path.join(self.root_dir, '%s.mat' % phase_map[phase])
        meta = load_mat(path)[0]
        return [
            Item(
                name=m[0][0], scene=m[1][0], type=m[2][0][0],
                points=m[3], resolution=m[4][0],
                dataset=self)
            for m in meta
        ]
