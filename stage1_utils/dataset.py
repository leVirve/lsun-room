import os
import numpy as np
import cv2
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


class DataItem():

    def __init__(self, root_dir, phase):
        self.image = os.path.join(root_dir, 'images/')
        self.label = os.path.join(root_dir, 'labels/')
        self.items = self._load(phase)

    def _load(self, phase):
        num_image = 5285 if phase == 'train' else 5050
        return ['img-%06d' % i for i in range(1, num_image + 1)]


def Phase(phase):
    if phase == 'validate':
        phase = 'train'
    return phase


class ImageFolderDataset(dset.ImageFolder):

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, root, target_size, phase):
        self.phase = Phase(phase)
        self.target_size = target_size
        self.dataset = DataItem(root, phase)
        self.filenames = self.dataset.items

    def __getitem__(self, index):
        return self.load(self.filenames[index])

    def load(self, filename):
        image_path = os.path.join(
            self.dataset.image, '{}/{}.jpg'.format(self.phase, filename))
        label_path = os.path.join(
            self.dataset.label, '{}/{}.png'.format(self.phase, filename))

        img = cv2.imread(image_path)
        lbl = cv2.imread(label_path, 0)

        img = cv2.resize(img, self.target_size, cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, self.target_size, cv2.INTER_NEAREST)

        img = self.transform(img)
        lbl = np.clip(lbl, 1, 37) - 1

        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def __len__(self):
        return len(self.filenames)
