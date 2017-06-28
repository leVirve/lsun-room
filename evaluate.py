import os
import scipy
import numpy as np
import skimage.io
import skimage.color

import click
from lsun_room import Phase, Dataset
from fcn_models.fcn32s_vgg16 import fcn_vggbase


def labelcolormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


@click.command()
@click.argument('weight_path', type=click.Path(exists=True))
@click.option('--max_length', type=int, default=2048)
def main(weight_path, max_length):
    dataset_root = '../data'

    experiment_name = weight_path.split('/')[-2]
    images_folder = 'output_images/%s/' % experiment_name
    layout_folder = 'output_layout/%s/' % experiment_name
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(layout_folder, exist_ok=True)

    model = fcn_vggbase(pretrained_weights=weight_path, crop=False)

    cmap = labelcolormap(5)

    dataset = Dataset(root_dir=dataset_root, phase=Phase.VALIDATE)
    images = [e.image for e in dataset.items]

    for img, e in zip(images, dataset.items):
        h, w, _ = img.shape
        if h > w and h > max_length:
            h, w = max_length, int(w * float(max_length) / h)
        elif w > h and w > max_length:
            h, w = int(h * float(max_length) / w), max_length

        img = scipy.misc.imresize(img, (h, w))

        batched_img = np.expand_dims(img, axis=0)
        pred = model.predict(batched_img)[0, ...]

        pred_img = np.argmax(pred, axis=2)
        print('... origin >> (%d, %d)' % (h, w))
        print('... predict >>', pred_img.shape)
        x = (pred_img.shape[0] - h) // 2
        y = (pred_img.shape[1] - w) // 2
        pred_img = pred_img[x:x + h, y:y + w]
        print('... cropped >>', pred_img.shape)

        out = skimage.color.label2rgb(pred_img, colors=cmap[1:], bg_label=0)

        pred_img = scipy.misc.imresize(pred_img, (h, w))
        out = scipy.misc.imresize(out, (h, w), interp='nearest')
        skimage.io.imsave(images_folder + '%s.png' % e.name, out)
        skimage.io.imsave(layout_folder + '%s.png' % e.name, pred_img)

        print('--> Done', e.name)


if __name__ == '__main__':
    main()
