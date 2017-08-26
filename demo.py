import time
from functools import wraps
import cv2
import click
import torch
import numpy as np
from net import StageNet

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


class Demo():

    stage1_weight = 'output/weight/stage1_ResFCN/mike20.pth'
    stage2_weight = 'output/weight/stage2_ResFCN/mike30.pth'

    def __init__(self, stage, input_size):
        self.input_size = input_size
        self.net = self.load_model(stage)
        self.cmap = self.create_camp(stage)

    def create_camp(self, stage):
        self.num_class = 37 if stage == 2 else 5
        return label_colormap(self.num_class + 1)[1:]

    @timeit
    def load_model(self, stage):

        if stage == 1:
            net = StageNet(name='stage1_ResFCN')
            pretrain = torch.load(self.stage1_weight)
        elif stage == 2:
            net = StageNet(name='stage2_ResFCN', stage_2=True, joint_roomtype=True)
            pretrain = torch.load(self.stage2_weight)

        model_dict = net.model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net.model.load_state_dict(model_dict)

        return net

    @timeit
    def process(self, raw):

        def output_label(output):
            label = np.zeros((*self.input_size, 3))
            for lbl in range(self.num_class):
                label[output == lbl] = self.cmap[lbl]
            return label

        img = np.subtract(raw, np.array([.485, .456, .406])) / np.array([.229, .224, .225])
        img = cv2.resize(img, self.input_size).transpose((2, 0, 1))
        batched_img = torch.from_numpy(np.expand_dims(img, 0)).float()

        output = self.net.predict_each(batched_img)
        label = output_label(output)

        raw = cv2.resize(raw, (raw.shape[1], raw.shape[0]))
        label = cv2.resize(label, (raw.shape[1], raw.shape[0]))

        return raw * 0.5 + label


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
@click.option('--stage', default=2)
@click.option('--input_size', default=(404, 404), type=(int, int))
def main(name, stage, input_size):

    demo = Demo(stage, input_size)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        raw_image = frame.astype('float') / 255
        output = demo.process(raw_image)

        cv2.imshow('layout', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
