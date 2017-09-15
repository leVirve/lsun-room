import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.autograd import Variable
import torchvision.models as models

from training.utils import to_numpy
from tools import timeit


class VggFCN(nn.Module):

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

    def predict(self, img):
        self.eval()
        output = self.forward(Variable(img, volatile=True).cuda())
        _, pred = torch.max(output, 1)
        return to_numpy(pred.data)

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


class ResFCN(nn.Module):

    def __init__(self, num_classes=5, input_size=None,
                 pretrained=True, base='resnet101'):
        self.inplanes = 64
        super().__init__()
        self.output_size = input_size

        BaseNet = getattr(models, base)
        self.resnet = BaseNet(pretrained=pretrained)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=100,
                               bias=False)
        self.layer3 = self._make_variant_layer(VariantBottleneck, 256, 23)
        self.layer4 = self._make_variant_layer(VariantBottleneck, 512, 3)

        self.fc = nn.Conv2d(512 * VariantBottleneck.expansion, num_classes, 1)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64,
                                          stride=32, bias=False)
        self.drop1 = nn.Dropout2d()
        self.drop2 = nn.Dropout2d()
        self._model_sugury()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.drop1(x)
        x = self.resnet.avgpool(x)
        x = self.drop2(x)
        x = self.resnet.fc(x)
        x = self.upscore(x)

        _, _, h, w = x.size()
        oh, ow = self.output_size
        sh, sw = (h - oh) // 2, (w - ow) // 2

        x = x[:, :, sh:sh + h, sw:sw + w].contiguous()

        return x

    def _make_variant_layer(self, block, planes, blocks):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = [block(self.inplanes, planes,
                        stride=1, dilation=1, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=2))

        return nn.Sequential(*layers)

    def _model_sugury(self):
        self.conv1.weight.data = self.resnet.conv1.weight.data
        for a, b in zip(
                self.resnet.layer3.modules(),
                self.layer3.modules()):
            if isinstance(a, nn.Conv2d) or isinstance(a, nn.BatchNorm2d):
                b.weight.data = a.weight.data
        for a, b in zip(
                self.resnet.layer4.modules(),
                self.layer4.modules()):
            if isinstance(a, nn.Conv2d) or isinstance(a, nn.BatchNorm2d):
                b.weight.data = a.weight.data

        self.resnet.conv1 = self.conv1
        self.resnet.layer3 = self.layer3
        self.resnet.layer4 = self.layer4
        self.resnet.fc = self.fc


class VariantBottleneck(models.resnet.Bottleneck):

    def __init__(self, inplanes, planes,
                 stride=1, dilation=1, downsample=None):
        super().__init__(inplanes, planes, stride, downsample)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
