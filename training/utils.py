import os
from io import BytesIO
from collections import defaultdict

import scipy.misc
import numpy as np
import tensorflow as tf
np.seterr(divide='ignore', invalid='ignore')


def to_numpy(output):
    return output.cpu().numpy()


def save_batched_images(tensors, filenames=None, folder=None):
    root_folder = 'output/layout/%s/' % folder
    os.makedirs(root_folder, exist_ok=True)

    for fname, img in zip(filenames, tensors):
        path = os.path.join(root_folder, '%s.png' % fname)
        scipy.misc.imsave(path, img)


def shrink_edge_width(trainer, train, validate):
    if (trainer.epoch + 1) % 4:
        return
    w = train.dataset.edge_width
    train.dataset.edge_width = w * 2 / 3
    validate.dataset.edge_width = w * 2 / 3


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


class Logger(object):
    '''Code referenced from
    yunjey/pytorch-tutorial//tensorboard/logger.py
    '''

    def __init__(self, log_dir, name=None):
        """Create a summary writer logging to log_dir."""
        self.name = name
        self.writer = tf.summary.FileWriter(os.path.join(log_dir, name))

    def scalar(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image(self, tag, images, step, tag_count_base=0):
        """Log list of images."""

        def to_summary(img):
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")
            return tf.Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0], width=img.shape[1])

        img_summaries = [
            tf.Summary.Value(tag='%s/%d' % (tag, tag_count_base + i),
                             image=to_summary(img))
            for i, img in enumerate(images)
        ]
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histogram(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
