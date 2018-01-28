import os
import pathlib

from onegan.io.loader import load_image, BaseDataset
from onegan.io.transform import SegmentationPair


class SunRGBDDataset(BaseDataset):

    num_classes = 37

    def __init__(self, phase, args, **kwargs):
        self.phase = phase
        self.target_size = (args.image_size, args.image_size)

        phase = 'test' if phase == 'val' else 'train'
        root_path = pathlib.Path(args.folder)
        iamges = sorted((root_path / 'images' / phase).glob('*.jpg'))
        labels = sorted((root_path / 'labels' / phase).glob('*.png'))

        self.filenames = [(img, lbl) for img, lbl in zip(iamges, labels)]
        self.paired_transform = SegmentationPair(self.target_size, final_transform=True)

    def __getitem__(self, index):
        image_path, label_path = self.filenames[index]
        image, label = self.paired_transform(load_image(image_path), load_image(label_path))
        return {'image': image, 'label': label, 'path': os.path.basename(label_path)}

    def __len__(self):
        return len(self.filenames)
