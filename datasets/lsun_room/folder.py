import torch
import torchvision.datasets as dset
from PIL import Image

from datasets.lsun_room.item import DataItems
from datasets.lsun_room.edge import mapping_func


def load_image(path):
    return Image.open(path)


class ImageFolderDataset(dset.ImageFolder):

    def __init__(self, root, target_size, phase,
                 input_transform=None, target_transform=None):
        self.target_size = target_size
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.num_classes = 5
        self._edge_width = 30

        self.dataset = DataItems(root_dir=root, phase=phase)
        self.filenames = [e.name for e in self.dataset.items]
        self.items = self.dataset.items
        self._edge_width = 30

    def __getitem__(self, index):
        return self.load(self.items[index], index)

    @property
    def edge_width(self):
        return self._edge_width

    @edge_width.setter
    def edge_width(self, width):
        self._edge_width = int(width) if width > 2 else 2

    def load(self, item, index):
        image = load_image(item.image_path).convert('RGB')
        layout = load_image(item.layout_path)
        image = self.input_transform(image)
        layout = self.target_transform(layout)
        edge = torch.from_numpy(self.load_edge_map(index) / 255).float()

        return {'image': image, 'layout': layout, 'edge': edge,
                'type': item.type}

    def load_edge_map(self, index):
        e = self.dataset.items[index]
        edge_map = mapping_func(e.type)
        return edge_map(e, image_size=self.target_size, width=self._edge_width)

    def __len__(self):
        return len(self.filenames)
