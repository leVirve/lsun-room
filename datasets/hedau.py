import os
import pathlib

import torch
import numpy as np
import scipy.io as sio
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from onegan.io.loader import load_image, BaseDataset
from onegan.io.transform import SegmentationPair


class HedauDataset(BaseDataset):

    num_classes = 5

    def __init__(self, phase, args, **kwarges):
        self.phase = phase
        self.target_size = (args.image_size, args.image_size)

        root_path = pathlib.Path(args.folder)
        phase = 'test' if phase == 'val' else 'train'
        index_meta = self.get_index_meta(phase, root_path)
        images = sorted((root_path / 'image').glob('*.jpg'))
        labels = sorted((root_path / 'layout').glob('*.mat'))

        self.filenames = [(images[index], labels[index]) for index in index_meta]
        self.paired_transform = SegmentationPair(self.target_size, final_transform=True)
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])

    def get_index_meta(self, phase, path):
        return sio.loadmat(path / 'traintestind.mat')[f'{phase}ind'][0] - 1

    def __getitem__(self, index):
        image_path, label_path = self.filenames[index]

        image = load_image(image_path).convert('RGB')
        layout = Image.fromarray(sio.loadmat(label_path)['fields'])

        image = F.resize(image, self.target_size, interpolation=Image.BILINEAR)
        layout = F.resize(layout, self.target_size, interpolation=Image.NEAREST)

        image = self.image_transform(image)
        layout = np.array(layout)
        layout[layout == 6] = 0
        layout = torch.from_numpy(layout - 1).long()

        return {'image': image, 'label': layout, 'path': os.path.basename(image_path)}

    def __len__(self):
        return len(self.filenames)
