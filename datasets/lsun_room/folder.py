import os

import numpy as np
import cv2
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from datasets.lsun_room.item import DataItems
from datasets.lsun_room.edge import mapping_func


class ImageFolderDataset(dset.ImageFolder):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, root, target_size, phase):
        self.target_size = target_size
        self.dataset = DataItems(root_dir=root, phase=phase)
        self.filenames = [e.name for e in self.dataset.items]

    def __getitem__(self, index):
        return self.load(self.filenames[index], index)

    def load(self, name, index):
        image_path = os.path.join(self.dataset.image, '%s.jpg' % name)
        label_path = os.path.join(self.dataset.layout_image, '%s.png' % name)

        img = cv2.imread(image_path)[:, :, ::-1]
        lbl = cv2.imread(label_path, 0)

        img = cv2.resize(img, self.target_size, cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, self.target_size, cv2.INTER_NEAREST)

        img = self.transform(img)
        lbl = np.clip(lbl, 1, 5) - 1
        lbl = torch.from_numpy(lbl).long()

        edge = torch.from_numpy(self.load_edge_map(index) / 255).float()

        return img, lbl, edge

    def load_edge_map(self, index):
        e = self.dataset.items[index]
        edge_map = mapping_func(e.type)
        return edge_map(e, image_size=self.target_size, width=2)

    def __len__(self):
        return len(self.filenames)
