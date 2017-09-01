from collections import defaultdict

import numpy as np
from models.utils import to_numpy
np.seterr(divide='ignore', invalid='ignore')


class EpochHistory():

    def __init__(self, length):
        self.len = length
        self.losses = defaultdict(float)
        self.accuracies = defaultdict(float)

    def add(self, losses, accuracies):
        for k, v in accuracies.items():
            self.accuracies[k] += v
        for k, v in losses.items():
            self.losses[k] += v.data[0]

        return {'accuracy': '%.04f' % accuracies["pixel_accuracy"],
                'loss': '%.04f' % losses["loss"].data[0],
                'miou': '%.04f' % accuracies["miou"]}

    def metric(self):
        terms = {k: v / self.len for k, v in self.accuracies.items()}
        terms.update({k: v / self.len for k, v in self.losses.items()})
        return terms


class LayoutAccuracy():

    def __call__(self, output, target):
        return self.accuracy(output, target)

    # Refer from: https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_semantic_segmentation.py
    def accuracy(self, pred_labels, gt_labels):
        confusion = self.semantic_confusion(pred_labels, gt_labels)
        iou = self.semantic_iou(confusion)
        pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
        class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

        return {'miou': np.nanmean(iou),
                'pixel_accuracy': pixel_accuracy,
                'mean_class_accuracy': np.nanmean(class_accuracy)}

    def semantic_iou(self, confusion):
        iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                           - np.diag(confusion))
        iou = np.diag(confusion) / iou_denominator
        return iou

    def semantic_confusion(self, pred_labels, gt_labels, n_class=5):
        confusion = np.zeros((n_class, n_class), dtype=np.int64)

        for pred_label, gt_label in zip(pred_labels, gt_labels):
            pred_label = to_numpy(pred_label.view(-1).data)
            gt_label = to_numpy(gt_label.view(-1).data)

            mask = gt_label >= 0
            confusion += np.bincount(
                n_class * gt_label[mask].astype(int) + pred_label[mask],
                minlength=n_class**2).reshape((n_class, n_class))
        return confusion
