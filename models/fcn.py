import types

import torchvision
import torch.nn as nn
import torch.nn.init as weight_init
import torchvision.models as models

from tools import timeit


@timeit
def build_FCN(num_classes=5, base_net='vgg16', input_size=None, pretrained=True):
    base = getattr(models, base_net)
    vgg = base(pretrained=pretrained)
    classifier, upscore = score_modules(num_classes)

    initialize_module(vgg, classifier)

    ''' Model assmbly '''
    vgg.output_size = input_size
    vgg.classifier = classifier
    vgg.upscore = upscore
    vgg.forward = types.MethodType(gg_forward, vgg)

    return vgg


def gg_forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    x = self.upscore(x)

    _, _, h, w = x.size()
    oh, ow = self.output_size
    sh, sw = (h - oh) // 2, (w - ow) // 2

    x = x[:, :, sh:-sh, sw:-sw].contiguous()

    return x


def score_modules(num_classes):
    classifier = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # score_fr
            nn.Conv2d(4096, num_classes, 1),
        )
    upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
                                 bias=False)
    return classifier, upscore


def initialize_module(vgg, classifier):
    vgg.features[0].padding = (100, 100)

    for m in vgg.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data = weight_init.kaiming_normal(m.weight.data)
        if isinstance(m, nn.MaxPool2d):
            m.ceil_mode = True
    for i in [0, 3]:
        a, b = vgg.classifier[i], classifier[i]
        b.weight.data = a.weight.data.view(b.weight.size())
        b.bias.data = a.bias.data.view(b.bias.size())


class FCN(nn.Module):

    def __init__(self, num_classes=5, input_size=None, pretrained=True):
        super(FCN, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )
        self.classifier = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # score_fr
            nn.Conv2d(4096, num_classes, 1),
        )
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
                                          bias=False)
        self.output_size = input_size
        if pretrained:
            self._initialize_weights()

    @timeit
    def _initialize_weights(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = weight_init.kaiming_normal(m.weight.data)
        for a, b in zip(vgg16.features, self.features):
            if (isinstance(a, nn.Conv2d) and isinstance(b, nn.Conv2d)):
                b.weight.data = a.weight.data
                b.bias.data = a.bias.data
        for i in [0, 3]:
            a, b = vgg16.classifier[i], self.classifier[i]
            b.weight.data = a.weight.data.view(b.weight.size())
            b.bias.data = a.bias.data.view(b.bias.size())

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.upscore(x)

        _, _, h, w = x.size()
        oh, ow = self.output_size
        sh, sw = (h - oh) // 2, (w - ow) // 2

        x = x[:, :, sh:-sh, sw:-sw].contiguous()
        return x
