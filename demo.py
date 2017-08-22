import time
from functools import wraps
import cv2
import click
import torch
import numpy as np
import matplotlib.pyplot as plt
from net import Stage_Net

torch.backends.cudnn.benchmark = True


def timeit(f):

    @wraps(f)
    def wrap(*args, **kw):
        s = time.time()
        result = f(*args, **kw)
        e = time.time()
        print('--> %s(), cost %2.4f sec' % (f.__name__, e - s))
        return result

    return wrap


def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


@click.command()
@click.option('--name', type=str)
@click.option('--image_size', default=(404, 404), type=(int, int))
@click.option('--workers', default=6, type=int)
def main(name, image_size, workers):
    net = load_model()
    path = '../data/images/0031cb53219d43468b723a729e25384464593a33.jpg'
    output, raw = net_feed(path, image_size, net)
    label = convet_label(image_size, output, raw)

    plt.imshow(raw * 0.5 + label)
    plt.show()


@timeit
def load_model():
    net = Stage_Net(name='stage2_ResFCN', pretrained=False, stage_2=True,
                    joint_class=True, type_portion=0.1, edge_portion=0.1)
    net.model.load_state_dict(torch.load('output/weight/stage2_ResFCN/20.pth'))
    return net


@timeit
def convet_label(image_size, output, raw):
    cmap = label_colormap(5)
    label = np.zeros((*image_size, 3))
    for lbl in range(5):
        label[output == lbl] = cmap[lbl]
    label = cv2.resize(label, (raw.shape[1], raw.shape[0]))
    return label


@timeit
def net_feed(path, image_size, net):

    @timeit
    def preprocess():
        raw = cv2.imread(path).astype('float') / 255
        img = cv2.resize(raw, image_size).transpose((2, 0, 1))
        batched_img = torch.from_numpy(np.expand_dims(img, 0)).float()
        return raw, batched_img

    @timeit
    def feed():
        return net.predict_each(batched_img)

    raw, batched_img = preprocess()
    output = feed()

    return output, raw


if __name__ == '__main__':
    main()
