import os
import pathlib

import torch
from scipy.io import loadmat
import torchvision.transforms.functional as F
from PIL import Image


class HedauDataset(torch.utils.data.Dataset):

    def __init__(self, phase, folder, image_size):
        assert phase in ('training', 'validation')
        self.phase = phase
        self.root = pathlib.Path(folder)
        self.target_size = (image_size, image_size)

        index_meta = load_hedau_mat(
            self.root / 'traintestind.mat',
            phase='test' if phase == 'validation' else 'train')
        images = sorted((self.root / 'image').glob('*.jpg'))
        labels = sorted((self.root / 'layout').glob('*.mat'))
        self.filenames = [(images[index], labels[index]) for index in index_meta]

    def __getitem__(self, index):
        image_path, label_path = self.filenames[index]

        image = F.to_tensor(Image.open(image_path).convert('RGB'))
        label = torch.from_numpy(loadmat(label_path)['fields'])[None]

        image = F.resize(image, self.target_size, interpolation=Image.BILINEAR)
        label = F.resize(label, self.target_size, interpolation=Image.NEAREST)

        label[label == 6] = 0
        return {
            'image': F.normalize(image, mean=0.5, std=0.5),
            # make 0 into 255 as ignore index
            'label': (label[0] - 1).long(),
            'path': os.path.basename(image_path)
        }

    def __len__(self):
        return len(self.filenames)

    def to_loader(self, batch_size, num_workers=0):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=self.phase == 'training',
            pin_memory=True, num_workers=num_workers
        )


def load_hedau_mat(filepath: pathlib.Path, phase: str):
    # one-based -> zero-based
    return loadmat(filepath)[f'{phase}ind'].squeeze() - 1
