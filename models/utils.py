import os

import skimage.io


def to_numpy(output):
    return output.cpu().numpy()


def save_batched_images(tensors, filenames=None, folder=None):
    root_folder = 'output/layout/%s/' % folder
    os.makedirs(root_folder, exist_ok=True)

    for fname, img in zip(filenames, tensors):
        path = os.path.join(root_folder, '%s.png' % fname)
        skimage.io.imsave(path, img)


def shrink_edge_width(trainer, train, validate):
    if (trainer.epoch + 1) % 4:
        return
    w = train.dataset.edge_width
    train.dataset.edge_width = w * 2 / 3
    validate.dataset.edge_width = w * 2 / 3
