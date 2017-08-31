import unittest

import matplotlib.pyplot as plt
from datasets.lsun_room.folder import ImageFolderDataset


class TestDataloader(unittest.TestCase):

    image_size = (404, 404)
    dataset = ImageFolderDataset(
        phase='val', root='../data/lsun_room', target_size=image_size)

    def test_data(self):
        img, lbl, edge = self.dataset[87]
        edge = edge.cpu().numpy()
        h, w = edge.shape
        self.assertEqual(self.image_size[0], h)
        self.assertEqual(self.image_size[1], w)

    def test_change_edge_width(self):
        self.dataset.edge_width = 50
        img, lbl, edge = self.dataset[87]
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
