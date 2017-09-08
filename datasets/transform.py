import torch
import numpy as np


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()


class Clamp:

    def __init__(self, label_min, label_max):
        self.label_min = label_min
        self.label_max = label_max

    def __call__(self, image):
        return image.clamp_(self.label_min, self.label_max).sub_(1)
