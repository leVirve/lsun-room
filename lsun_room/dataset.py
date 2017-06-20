import numpy as np

import lsun_room
from lsun_room.label import mapping_func
from lsun_room.utils import load_mat, load_image, save_image


class Item():

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._image = None
        self._layout = None

    def layout_remap(self):
        mapping = mapping_func(self.type)(self)

        old_layout = np.array(self.layout)
        for new_label, point in mapping:
            old_label = old_layout[point[1], point[0]]
            self._layout[old_layout == old_label] = new_label

    def save_layout(self, visualization=False):
        path = lsun_room.layout_image_pattern % self.name
        if visualization:
            save_image(path, (self.layout * 51).astype('uint8'))
            return (self.layout * 51).astype('uint8')
        else:
            save_image(path, self.layout)

    @property
    def image(self):
        if self._image is None:
            path = lsun_room.image_pattern % self.name
            self._image = load_image(path)
        return self._image

    @property
    def layout(self):
        if self._layout is None:
            path = lsun_room.layout_pattern % self.name
            self._layout = load_mat(path)['layout']
        return self._layout


class Dataset():
    def __init__(self, state):
        self.items = self._load(state)

    def _load(self, state):
        assert state in lsun_room.data_path.keys()
        meta = self._load_meta(state)[state][0]
        return self._convert_to_item(meta)

    def _load_meta(self, state):
        path = lsun_room.data_path[state]
        return load_mat(path)

    def _convert_to_item(self, meta):
        return [
            Item(
                name=m[0][0], scene=m[1][0], type=m[2][0][0],
                points=m[3], resolution=m[4][0])
            for m in meta
        ]
