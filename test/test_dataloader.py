import pdb
import unittest

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from datasets.lsun_room.folder import ImageFolderDataset
from datasets.transform import ToLabel


class TestDataloader(unittest.TestCase):

    image_size = (404, 404)

    input_transform = transforms.Compose([
        transforms.Scale(image_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Scale(image_size, interpolation=Image.NEAREST),
        ToLabel()
    ])
    dataset = ImageFolderDataset(
        phase='train', root='../data/lsun_room', target_size=image_size,
        input_transform=input_transform,
        target_transform=target_transform)

    loader_args = {'num_workers': 1, 'pin_memory': True}
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, **loader_args)

    def test_data(self):
        data = self.dataset[87]
        edge = data['edge']
        edge = edge.cpu().numpy()
        h, w = edge.shape
        self.assertEqual(self.image_size[0], h)
        self.assertEqual(self.image_size[1], w)

    def test_change_edge_width(self):
        self.dataset.edge_width = 50
        data = self.dataset[87]
        edge = data['edge']
        edge = edge.cpu().numpy()
        h, w = edge.shape
        self.assertEqual(self.image_size[0], h)
        self.assertEqual(self.image_size[1], w)
        self.show(edge)

    def show(self, img):
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    unittest.main()
