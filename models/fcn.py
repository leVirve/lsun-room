import torch.nn as nn
import torch.nn.init as weight_init
import torchvision.models as models

from tools import timeit


class FCN(nn.Module):

    def __init__(self, num_classes=5, input_size=None,
                 pretrained=True, base='vgg16_bn'):
        super().__init__()
        self.output_size = input_size

        BaseNet = getattr(models, base)
        self.vgg = BaseNet(pretrained=pretrained)

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
        self._initialize_module(pretrained)
        self._model_sugury()

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.classifier(x)
        x = self.upscore(x)

        _, _, h, w = x.size()
        oh, ow = self.output_size
        sh, sw = (h - oh) // 2, (w - ow) // 2

        x = x[:, :, sh:-sh, sw:-sw].contiguous()

        return x

    @timeit
    def _initialize_module(self, pretrained):
        if not pretrained:
            return

        for i in [0, 3]:
            a, b = self.vgg.classifier[i], self.classifier[i]
            b.weight.data = a.weight.data.view(b.weight.size())
            b.bias.data = a.bias.data.view(b.bias.size())

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = weight_init.kaiming_normal(m.weight.data)

    def _model_sugury(self):
        ''' <Black magic> Hack the netwrok structure '''
        for m in self.vgg.modules():
            if isinstance(m, nn.MaxPool2d):
                m.ceil_mode = True

        self.vgg.features[0].padding = (100, 100)
        self.vgg.classifier = self.classifier
