import cv2
import click
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from models.fcn import FCN
from tools import timeit, label_colormap

torch.backends.cudnn.benchmark = True


class Demo():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, input_size, weight=None):
        self.input_size = input_size
        self.num_class = 5
        self.model = self.load_model(weight)
        self.cmap = self.create_camp()

    def create_camp(self):
        return label_colormap(self.num_class + 1)[1:]

    @timeit
    def load_model(self, weight):
        model = nn.DataParallel(
                FCN(num_classes=5, input_size=self.input_size,
                    pretrained=False)).cuda()
        model.load_state_dict(torch.load(weight))
        return model

    @timeit
    def process(self, raw):

        def output_label(output):
            label = np.zeros((*self.input_size, 3))
            for lbl in range(self.num_class):
                label[output.squeeze() == lbl] = self.cmap[lbl]
            return label

        img = cv2.resize(raw, self.input_size)
        batched_img = self.transform(img).unsqueeze(0).cuda()

        output = self.model.module.predict(batched_img)
        label = output_label(output)

        label = cv2.resize(label, (raw.shape[1], raw.shape[0]))

        return raw.astype(np.float32) / 255 * 0.5 + label


@click.command()
@click.option('--device', default=0)
@click.option('--video', default='')
@click.option('--weight', default='output/weight/vgg_bn_new/24.pth')
@click.option('--input_size', default=(404, 404), type=(int, int))
def main(device, video, weight, input_size):

    demo = Demo(input_size, weight=weight)

    reader = video if video else device
    cap = cv2.VideoCapture(reader)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        output = demo.process(frame[:, :, ::-1])

        cv2.imshow('layout', output[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
