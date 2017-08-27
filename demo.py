import cv2
import click
import torch
import torch.nn as nn
import numpy as np
from models.fcn import FCN
from models.saver import Predictor

from tools import timeit

torch.backends.cudnn.benchmark = True


class Demo():

    weight = 'output/weight/vgg/29.pth'

    def __init__(self, input_size):
        self.input_size = input_size
        self.num_class = 5
        self.predictor = self.load_model()
        self.cmap = self.create_camp()

    def create_camp(self):
        return label_colormap(self.num_class + 1)[1:]

    @timeit
    def load_model(self):
        model = nn.DataParallel(
                FCN(num_classes=5, pretrained=False, input_size=self.input_size)
            ).cuda()
        predictor = Predictor(model)
        predictor.model.load_state_dict(torch.load(self.weight))
        return predictor

    @timeit
    def process(self, raw):

        def output_label(output):
            label = np.zeros((*self.input_size, 3))
            for lbl in range(self.num_class):
                label[output.squeeze() == lbl] = self.cmap[lbl]
            return label

        img = np.subtract(raw, np.array([.485, .456, .406])) / np.array([.229, .224, .225])
        img = cv2.resize(img, self.input_size).transpose((2, 0, 1))
        batched_img = torch.from_numpy(np.expand_dims(img, 0)).float()

        output = self.predictor.forward(batched_img)
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
@click.option('--input_size', default=(404, 404), type=(int, int))
def main(name, input_size):

    demo = Demo(input_size)
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
