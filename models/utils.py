import os
from collections import defaultdict

import torch
import skimage.io


class EpochHistory():

    def __init__(self, length):
        self.len = length
        self.loss_terms = defaultdict(float)
        self.accuracies = 0

    def add(self, losses, accuracy):
        self.accuracies += accuracy.data[0]
        for k, v in losses.items():
            self.loss_terms[k] += v.data[0]

        return {'accuracy': '%.04f' % accuracy.data[0],
                'loss': '%.04f' % losses["loss"].data[0]}

    def metric(self, prefix=''):
        terms = {prefix + 'accuracy': self.accuracies / self.len}
        terms.update({
            prefix + k: v / self.len for k, v in self.loss_terms.items()})
        return terms


class LayoutAccuracy():

    def __call__(self, output, target):
        return self.pixelwise_accuracy(output, target)

    def pixelwise_accuracy(self, output, target):
        _, output = torch.max(output, 1)
        return (output == target).float().mean()


def save_images(folder_name, named_tensor):
    root_folder = 'output/layout/%s/' % folder_name
    os.makedirs(root_folder, exist_ok=True)

    filenames, tensors = named_tensor
    for fname, img in zip(filenames, tensors):
        path = os.path.join(root_folder, '%s.png' % fname)
        skimage.io.imsave(path, img)
