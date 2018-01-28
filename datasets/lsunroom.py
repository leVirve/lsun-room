import glob
import os
import pathlib
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from lsun_room.label import Layout
from lsun_room.loader import LsunRoomDataset as _BaseDataset
from lsun_room.loader import get_meta


class LsunRoomDataset(_BaseDataset):

    def __init__(self, phase, args, **kwarges):
        self.phase = phase
        self.args = args
        self.root = pathlib.Path(args.folder)
        self.target_size = (args.image_size, args.image_size)
        self.meta = self.collect_meta(self.root, phase=phase)

        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])

    def collect_meta(self, root, phase):
        ''' metadata fold1: original dataset '''
        meta = [
            {'image_path': root / self.image_folder / f'{e["name"]}.jpg',
             'layout_path': root / self.layout_folder / f'{e["name"]}.png',
             'name': e['name'] + '.png', 'type': e['type']}
            for e in get_meta(dataset_root=root, phase=phase)
        ]
        counter = [0 for i in range(11)]
        for m in meta:
            counter[m['type']] += 1
        self._log.info('meta-fold1 -->' + '|'.join([str(c).rjust(4) for c in counter]))

        if phase == 'val' or self.args.datafold == 1:
            return meta

        ''' metadata fold2: 1-step degeneration laugmentated dataset '''
        counter = [0 for i in range(11)]
        aug_meta = []
        for i in range(11):
            imgs = sorted((root / 'aug_image' / f'type{i}').glob('*.jpg'))
            lbls = sorted((root / 'aug_layout' / f'type{i}').glob('*.png'))
            aug_meta += [
                {'image_path': img, 'layout_path': lbl, 'name': os.path.basename(lbl), 'type': i}
                for img, lbl in zip(imgs, lbls)]
            counter[i] = len(imgs)
        self._log.info('meta-fold2 -->' + '|'.join([str(c).rjust(4) for c in counter]))

        if self.args.datafold == 2:
            return meta + aug_meta

        ''' metadata fold3: 2-step degeneration laugmentated dataset '''
        counter = [0 for i in range(11)]
        augaug_meta = []
        for i in range(11):
            imgs = sorted(glob.glob(os.path.join(root, 'augaug_image', f'type{i}', '*.jpg')))
            lbls = sorted(glob.glob(os.path.join(root, 'augaug_layout', f'type{i}', '*.png')))
            augaug_meta += [
                {'image_path': img, 'layout_path': lbl, 'name': os.path.basename(lbl), 'type': i}
                for img, lbl in zip(imgs, lbls)]
            counter[i] = len(imgs)
        self._log.info('meta-fold3 -->' + '|'.join([str(c).rjust(4) for c in counter]))

        return meta + aug_meta + augaug_meta

    def __getitem__(self, index):
        e = self.meta[index]

        image = Image.open(e['image_path']).convert('RGB')
        layout = Image.open(e['layout_path']).convert('L')

        image = T.functional.resize(image, self.target_size, Image.BICUBIC)
        layout = T.functional.resize(layout, self.target_size, Image.NEAREST)

        if self.phase == 'train':
            image = self.color_jitter(image)
            image, layout = self.random_lr_flip(image, layout)

        layout = np.array(layout)
        edge = self.gen_edge_map(layout)

        item = {
            'image': self.image_transform(image),
            'label': torch.from_numpy(layout - 1).clamp_(0, 4).long(),
            'edge': torch.from_numpy(edge).clamp_(0, 1).float(),
            'type': int(e['type']), 'filename': e['name'],
        }

        if self.args.datafold == 1 and self.args.use_edge and self.args.use_corner:
            # only for datafold == 1
            item['edge'] = self.load_edge_map(index)
            item['corner'] = self.load_corner_map(index)

        return item

    def gen_edge_map(self, layout):
        import cv2
        lbl = cv2.GaussianBlur(layout.astype('uint8'), (3, 3), 0)
        edge = cv2.Laplacian(lbl, cv2.CV_64F)
        activation = cv2.dilate(np.abs(edge), np.ones((5, 5), np.uint8), iterations=1)
        activation[activation != 0] = 1
        return cv2.GaussianBlur(activation, (15, 15), 5)

    @classmethod
    def random_lr_flip(cls, image, layout):

        def layout_lr_swap(layout):
            layout = np.array(layout)
            old_layout = layout.copy()
            layout[old_layout == Layout.left.value] = Layout.right.value
            layout[old_layout == Layout.right.value] = Layout.left.value
            return layout

        if random.random() >= 0.5:
            image = T.functional.hflip(image)
            layout = layout_lr_swap(T.functional.hflip(layout))

        return image, layout

    @classmethod
    def random_rotate(cls, image, layout):
        angle = np.random.uniform(-5, 5)
        T.functional.rotate(image, angle, resample=False, expand=False, center=None)
        T.functional.rotate(layout, angle, resample=False, expand=False, center=None)
