import os
import numpy as np

from lsun_room import Phase
from dataset import DataGenerator

from fcn_models.fcn32s_vgg16 import fcn_vggbase


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def labelcolormap(N=256):
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


def main():
    experiment_name = 'vggbase_go_adam_lr1e-4'
    weights_name = 'weights.{epoch:02d}.hdf5'.format(epoch=18)

    dataset_root = '../data'
    size = 512
    workers = 16
    batch_size = 8

    data_gen = DataGenerator()
    validation_generator = data_gen.flow_from_directory(
        directory=dataset_root, phase=Phase.VALIDATE,
        target_size=(size, size),
        batch_size=batch_size)

    model = fcn_vggbase(
        input_shape=(size, size, 3),
        pretrained_weights=os.path.join(experiment_name, weights_name))

    result = model.predict_generator(
        validation_generator,
        steps=(validation_generator.samples + 1) // batch_size,
        workers=workers
    )

    for e in result:
        import skimage.io
        import skimage.color
        cmap = labelcolormap(5)

        out = skimage.color.label2rgb(np.argmax(e, axis=2), colors=cmap[1:], bg_label=0)
        skimage.io.imsave('e-out.png', np.argmax(e, axis=2))
        skimage.io.imsave('out.png', out)
        break


if __name__ == '__main__':
    main()
