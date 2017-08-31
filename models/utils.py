import os

import skimage.io


def to_numpy_img(output):
    return output.cpu().numpy()


def save_batched_images(tensors, filenames=None, folder=None):
    root_folder = 'output/layout/%s/' % folder
    os.makedirs(root_folder, exist_ok=True)

    for fname, img in zip(filenames, tensors):
        path = os.path.join(root_folder, '%s.png' % fname)
        skimage.io.imsave(path, img)
