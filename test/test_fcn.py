import unittest

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.fcn import FCN


class TestFCN(unittest.TestCase):

    image_size = (404, 404)
    fake_img = torch.randn((1, 3, *image_size)).cuda()

    def test_model(self):
        model = nn.DataParallel(
            FCN(input_size=self.image_size, base='vgg16')).cuda()
        output = model(Variable(self.fake_img, requires_grad=False))
        self._check_output_size(output)

    def test_bn_model(self):
        model = nn.DataParallel(
            FCN(input_size=self.image_size, base='vgg16_bn')).cuda()
        output = model(Variable(self.fake_img, requires_grad=False))
        self._check_output_size(output)

    def _check_output_size(self, output):
        self.assertEqual(output.size()[1], 5)
        self.assertEqual(output.size()[2], 404)
        self.assertEqual(output.size()[3], 404)


if __name__ == '__main__':
    unittest.main()
