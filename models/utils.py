import torch
import numpy as np


class EpochHistory():

    def __init__(self, length):
        self.count = 0
        self.len = length
        self.loss_term = {'xent': None, 'l1': None, 'edge': None}
        self.losses = np.zeros(self.len)
        self.accuracies = np.zeros(self.len)

    def add(self, loss, loss_term, acc):
        self.losses[self.count] = loss.data[0]
        self.accuracies[self.count] = acc.data[0]

        for k, v in loss_term.items():
            if self.loss_term[k] is None:
                self.loss_term[k] = np.zeros(self.len)
            self.loss_term[k][self.count] = v.data[0]

        self.count += 1

    def metric(self, prefix=''):
        terms = {prefix + 'loss': self.losses.mean(),
                 prefix + 'accuracy': self.accuracies.mean()}
        terms.update({
            prefix + k: v.mean() for k, v in self.loss_term.items()
            if v is not None})
        return terms


class LayoutAccuracy():

    def __call__(self, output, target):
        return self.pixelwise_accuracy(output, target)

    def pixelwise_accuracy(self, output, target):
        _, output = torch.max(output, 1)
        return (output == target).float().mean()
