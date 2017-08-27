import unittest

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.fcn import build_FCN, FCN


class TestFCN(unittest.TestCase):

    image_size = (404, 404)

    model = nn.DataParallel(FCN(input_size=image_size))
    bn_model = nn.DataParallel(build_FCN(base_net='vgg16_bn', input_size=image_size))

    def test_model(self):
        fake_img = torch.randn((1, 3, 404, 404))
        output = self.model(Variable(fake_img, requires_grad=False))
        self.assertEqual(output.size()[1], 5)
        self.assertEqual(output.size()[2], 404)
        self.assertEqual(output.size()[3], 404)

    def test_bn_model(self):
        fake_img = torch.randn((1, 3, 404, 404))
        output = self.bn_model(Variable(fake_img, requires_grad=False))
        self.assertEqual(output.size()[1], 5)
        self.assertEqual(output.size()[2], 404)
        self.assertEqual(output.size()[3], 404)


if __name__ == '__main__':
    unittest.main()
