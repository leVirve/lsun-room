import enum
import pathlib
import random
from collections import defaultdict, namedtuple
from typing import Sequence, Union

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms as T
from torchvision.transforms import functional as F

TensorCHW = Union[torch.Tensor, np.ndarray]
Scene = namedtuple('Scene', ['filename', 'scene_type', 'layout_type', 'keypoints', 'shape'])


class LsunRoomDataset(torch.utils.data.Dataset):

    def __init__(self, phase, folder, image_size):
        assert phase in ('training', 'validation', 'testing')
        self.root = pathlib.Path(folder)
        self.phase = phase
        self.metadata = load_lsun_mat(self.root / f'{phase}.mat')
        self.target_size = (image_size, image_size)
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        element = self.metadata[index]
        image_path = self.root / f'images/{element.filename}.jpg'
        label_path = self.root / f'layout_seg/{element.filename}.mat'

        image = F.to_tensor(Image.open(image_path).convert('RGB'))
        label = torch.from_numpy(loadmat(label_path)['layout'])[None]
        layout_type = int(element.layout_type)

        if self.phase == 'training':
            image = self.color_jitter(image)
            image, label = random_lr_flip(image, label)
            image, label, layout_type = random_layout_degradation(image, label, layout_type)

        image = F.resize(image, self.target_size, Image.BILINEAR)
        label = F.resize(label, self.target_size, Image.NEAREST)
        edge_map = generate_edge_map_from(label[0].numpy())

        item = {
            'image': F.normalize(image, mean=0.5, std=0.5),
            # make 0 into 255 as ignore index
            'label': (label[0] - 1).clamp_(0, 4).long(),
            'edge': torch.from_numpy(edge_map).clamp_(0, 1).float(),
            'type': layout_type,
        }
        return item

    def to_loader(self, batch_size, num_workers=0):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=self.phase == 'training',
            pin_memory=True, num_workers=num_workers
        )


def load_lsun_mat(filepath: pathlib.Path) -> Sequence[Scene]:
    data = loadmat(filepath)
    return [
        Scene(*(m.squeeze() for m in metadata))
        for metadata in data[filepath.stem].squeeze()
    ]


def generate_edge_map_from(label):
    lbl = cv2.GaussianBlur(label.astype('uint8'), (3, 3), 0)
    edge = cv2.Laplacian(lbl, cv2.CV_64F)
    activation = cv2.dilate(np.abs(edge), np.ones((5, 5), np.uint8), iterations=1)
    activation[activation != 0] = 1
    return cv2.GaussianBlur(activation, (15, 15), 5)


def random_lr_flip(image: TensorCHW, label: TensorCHW):
    if 0.5 < torch.rand(1):
        return image, label

    image = T.functional.hflip(image)
    label = T.functional.hflip(label)

    label = label.numpy()
    old_label = label.copy()
    label[old_label == Layout.left.value] = Layout.right.value
    label[old_label == Layout.right.value] = Layout.left.value
    label = torch.from_numpy(label)

    return image, label


def remove_ceiling(image: TensorCHW, label: TensorCHW):
    _, rows, _ = np.where(label == Layout.ceiling.value)
    if rows.size == 0:
        return image, label
    bound = rows.min()
    image = image[:, bound:, :]
    label = label[:, bound:, :]
    return image, label


def remove_floor(image: TensorCHW, label: TensorCHW):
    _, rows, _ = np.where(label == Layout.floor.value)
    if rows.size == 0:
        return image, label
    bound = rows.max()
    image = image[:, :bound, :]
    label = label[:, :bound, :]
    return image, label


def remove_right(image: TensorCHW, label: TensorCHW):
    _, _, cols = np.where(label == Layout.right.value)
    if cols.size == 0:
        return image, label
    bound = cols.min()
    image = image[:, :, :bound]
    label = label[:, :, :bound]
    if np.any(label == Layout.frontal.value):
        label[label == Layout.frontal.value] = Layout.right.value
    else:
        label[label == Layout.left.value] = Layout.frontal.value
    return image, label


def remove_left(image: TensorCHW, label: TensorCHW):
    _, _, cols = np.where(label == Layout.left.value)
    if cols.size == 0:
        return image, label
    bound = cols.max()
    image = image[:, :, bound:]
    label = label[:, :, bound:]
    if np.any(label == Layout.frontal.value):
        label[label == Layout.frontal.value] = Layout.left.value
    else:
        label[label == Layout.right.value] = Layout.frontal.value
    return image, label


def accept_aspect_ratio(x: TensorCHW):
    try:
        h, w = x.shape[1:]
        ratio = h / w if h > w else w / h
        return ratio < 16 / 9
    except (ZeroDivisionError, RuntimeWarning):
        return False


def random_layout_degradation(image, label, layout_type):
    if 0.5 < torch.rand(1) or layout_type > 7:
        return image, label, layout_type

    image_, label_, layout_type_ = image.numpy(), label.numpy(), layout_type
    degradation_paths = room_layout_degradation.random_paths(layout_type)

    for new_layout_type, degrade_fn in degradation_paths:
        new_image, new_label = degrade_fn(image_, label_)
        if not accept_aspect_ratio(new_image):
            return image, label, layout_type
        image_, label_, layout_type_ = new_image, new_label, new_layout_type

    image, label = torch.from_numpy(image_), torch.from_numpy(label_)
    layout_type = layout_type_

    return image, label, layout_type


class Layout(enum.Enum):
    frontal = 1
    left = 2
    right = 3
    floor = 4
    ceiling = 5


class RoomLayoutDegradation:
    DEGRADATION_GRAPH = {
        0: [(1, remove_ceiling), (2, remove_floor), (5, remove_right), (5, remove_left)],
        1: [(4, remove_right), (4, remove_left), (7, remove_floor)],
        2: [(3, remove_right), (3, remove_left), (7, remove_ceiling)],
        3: [(8, remove_right), (8, remove_left), (10, remove_ceiling)],
        4: [(9, remove_right), (9, remove_left), (10, remove_floor)],
        5: [(6, remove_right), (6, remove_left), (3, remove_floor), (4, remove_ceiling)],
        6: [(8, remove_floor), (9, remove_ceiling)],
        7: [(10, remove_right), (10, remove_left)],
    }

    def __init__(self) -> None:
        self.possible_degradations = defaultdict(list)
        self.initialize()

    def initialize(self):
        # all 11 type, #0~#10, but type #8, #9, #10 has no transform.
        for room_type in range(8):
            self.find_possible_degradations(
                room_type, [], results=self.possible_degradations[room_type])

    @classmethod
    def find_possible_degradations(cls, node, path, results):
        results.append([*path])
        for neighbor, operation in cls.DEGRADATION_GRAPH.get(node, []):
            path.append((neighbor, operation))
            cls.find_possible_degradations(neighbor, path, results)
            path.pop()

    def random_paths(self, room_type):
        return random.choice(self.possible_degradations[room_type])


room_layout_degradation = RoomLayoutDegradation()
