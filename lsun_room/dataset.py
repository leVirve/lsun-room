import os

import numpy as np
import cv2
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from lsun_room.label import mapping_func
from lsun_room.utils import load_mat, load_image, save_image


class ImageFolderDataset(dset.ImageFolder):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, root, target_size, phase):
        self.target_size = target_size
        self.dataset = Dataset(root_dir=root, phase=phase)
        self.filenames = [e.name for e in self.dataset.items]

    def __getitem__(self, index):
        return self.load(self.filenames[index])

    def load(self, name):
        image_path = os.path.join(self.dataset.image, '%s.jpg' % name)
        label_path = os.path.join(self.dataset.layout_image, '%s.png' % name)

        img = cv2.imread(image_path)
        lbl = cv2.imread(label_path, 0)

        img = cv2.resize(img, self.target_size, cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, self.target_size, cv2.INTER_NEAREST)

        img = self.transform(img)
        lbl = np.clip(lbl, 1, 5) - 1
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def __len__(self):
        return len(self.filenames)


class Item():

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._image = None
        self._layout = None
        self.image_path = self.image_pattern % self.name
        self.layout_path = self.layout_image_pattern % self.name

    @property
    def image(self):
        if self._image is None:
            path = self.image_pattern % self.name
            self._image = load_image(path)
        return self._image

    @property
    def layout(self):
        if self._layout is None:
            path = self.layout_image_pattern % self.name
            if os.path.exists(path):
                self._layout = load_image(path)[:, :, :1]
            else:
                path = self.layout_pattern % self.name
                self._layout = np.expand_dims(load_mat(path), axis=2)
        return self._layout

    def remap_layout(self):
        mapping = mapping_func(self.type)(self)

        old_layout = np.array(self.layout)
        for new_label, point in mapping:
            old_label = old_layout[point[1], point[0]]
            self._layout[old_layout == old_label] = new_label

    def save_layout(self, visualization=False):
        path = self.layout_image_pattern % self.name
        if visualization:
            save_image(path, (self.layout * 51).astype('uint8'))
            return (self.layout * 51).astype('uint8')
        else:
            save_image(path, self.layout)

    def __str__(self):
        return '<DataItem: %s>' % self.name


class Dataset():

    def __init__(self, root_dir, phase):
        self.root_dir = root_dir
        self.image = os.path.join(root_dir, 'images/')
        self.layout = os.path.join(root_dir, 'layout_seg/')
        self.layout_image = os.path.join(root_dir, 'layout_seg_images/')

        Item.image_pattern = self.image + '%s.jpg'
        Item.layout_pattern = self.layout + '%s.mat'
        Item.layout_image_pattern = self.layout_image + '%s.png'

        self.items = self._load(phase)

    def _load(self, phase):
        path = os.path.join(self.root_dir, '%s.mat' % phase)
        meta = load_mat(path)[0]
        return [
            Item(
                name=m[0][0], scene=m[1][0], type=m[2][0][0],
                points=m[3], resolution=m[4][0])
            for m in meta
        ]
