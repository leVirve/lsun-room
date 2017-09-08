import os

import numpy as np
import cv2
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from datasets.lsun_room.item import DataItems
from datasets.lsun_room.edge import mapping_func


class NamedSample():
    pass


class ImageFolderDataset(dset.ImageFolder):

    # TODO:
    # - move transform into main script
    # - make __getitem__ return well-structured object
    # - not use OpenCV

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, root, target_size, phase):
        self.target_size = target_size
        self.dataset = DataItems(root_dir=root, phase=phase)
        self.filenames = [e.name for e in self.dataset.items]
        self.num_classes = 5
        self._edge_width = 30

    def __getitem__(self, index):
        return self.load(self.filenames[index], index)

    @property
    def edge_width(self):
        return self._edge_width

    @edge_width.setter
    def edge_width(self, width):
        self._edge_width = int(width) if width > 2 else 2

    def _pil_load(self, name, index):
        image_path = os.path.join(self.dataset.image, '%s.jpg' % name)
        label_path = os.path.join(self.dataset.layout_image, '%s.png' % name)

        loader = dset.folder.default_loader

        universial_transform = [
            transforms.Scale((404, 404)),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose([
            *universial_transform,
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([
            universial_transform,
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(loader(image_path))
        layout = target_transform(loader(label_path))
        edge = torch.from_numpy(self.load_edge_map(index) / 255).float().sub_(1)

        return NamedSample(image, layout, edge)

    def load(self, name, index):
        image_path = os.path.join(self.dataset.image, '%s.jpg' % name)
        label_path = os.path.join(self.dataset.layout_image, '%s.png' % name)

        img = cv2.imread(image_path)[:, :, ::-1]
        lbl = cv2.imread(label_path, 0)
        edge = self.load_edge_map(index) / 255

        img = cv2.resize(img, self.target_size, cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, self.target_size, cv2.INTER_NEAREST)

        img = self.transform(img)
        lbl = np.clip(lbl, 1, 5) - 1
        lbl = torch.from_numpy(lbl).long()
        edge = torch.from_numpy(edge).float()

        return img, lbl, edge

    def load_edge_map(self, index):
        e = self.dataset.items[index]
        edge_map = mapping_func(e.type)
        return edge_map(e, image_size=self.target_size, width=self._edge_width)

    def __len__(self):
        return len(self.filenames)
