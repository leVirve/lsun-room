import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torchvision.models as models
from onegan.external import fcn


class LaFCN(nn.Module):

    def __init__(self, num_classes=5, input_size=(320, 320), pretrained=True, base='vgg16_bn'):
        super().__init__()
        self.output_size = input_size

        BaseNet = getattr(models, base)
        vgg = BaseNet(pretrained=pretrained)
        self.features = vgg.features

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
        # self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=8, padding=12, bias=False)
        self.edge = nn.ConvTranspose2d(num_classes, 1, kernel_size=8, stride=4, padding=2, bias=False)
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(num_classes, 32, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(4),
        #     # fc
        #     nn.Conv2d(32, 64, kernel_size=5, padding=2),
        #     nn.Conv2d(64, 128, kernel_size=1),  # 1x1
        # )
        # self.edge = nn.Sequential(
        #     nn.ConvTranspose2d(128, 32, kernel_size=8, stride=4, padding=2, bias=False),
        #     nn.ConvTranspose2d(32, 1, kernel_size=16, stride=8, padding=4, bias=False),
        # )
        self.features[0].padding = (100, 100)
        self._initialize_module(vgg, pretrained)

    def forward(self, x):
        # x = 320x320
        x = self.features(x)
        b = self.classifier(x)  # 10x10
        x = self.upscore(b)     # 80x80

        # c = self.downsample(b)  # 1x1
        # c = nn.functional.upsample(c, scale_factor=4, mode='bilinear')
        c = self.edge(x)
        x = nn.functional.upsample(x, scale_factor=4, mode='bilinear')

        return x, c

    def _initialize_module(self, vgg, pretrained):
        if not pretrained:
            return

        for i in [0, 3]:
            a, b = vgg.classifier[i], self.classifier[i]
            b.weight.data = a.weight.data.view(b.weight.size())
            b.bias.data = a.bias.data.view(b.bias.size())

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = weight_init.kaiming_normal(m.weight.data)


class VggFCN(nn.Module):

    def __init__(self, num_class=5, input_size=(320, 320), pretrained=True, base='vgg16_bn'):
        super().__init__()
        self.output_size = input_size

        BaseNet = getattr(models, base)
        vgg = BaseNet(pretrained=pretrained)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
                # fc6
                nn.Conv2d(512, 4096, kernel_size=7),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                # fc7
                nn.Conv2d(4096, 4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                # score_fr
                nn.Conv2d(4096, num_class, kernel_size=1),
            )
        self.upscore = nn.ConvTranspose2d(num_class, num_class, 64, stride=32, bias=False)
        self.upscore.weight.data = fcn.get_upsampling_weight(num_class, num_class, 64)
        self._initialize_module(pretrained, classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.upscore(x)

        _, _, h, w = x.size()
        oh, ow = self.output_size
        sh, sw = (h - oh) // 2, (w - ow) // 2

        x = x[:, :, sh:-sh, sw:-sw].contiguous()

        return x, None

    def _initialize_module(self, pretrained, classifier):
        if not pretrained:
            return

        for i in [0, 3]:
            a, b = classifier[i], self.classifier[i]
            b.weight.data = a.weight.data.view(b.weight.size())
            b.bias.data = a.bias.data.view(b.bias.size())


class ResFCN(nn.Module):

    def __init__(self, num_classes=5, num_room_types=11, pretrained=True, base='resnet101'):
        self.inplanes = 64
        super().__init__()

        BaseNet = getattr(models, base)
        resnet = BaseNet(pretrained=pretrained)

        self.conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # room type classification
        self.avgpool = nn.AvgPool2d(10)
        self.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, num_room_types)

        # planar segmentation
        self.fc_conv = nn.Conv2d(512 * models.resnet.Bottleneck.expansion, num_classes, kernel_size=1, bias=False)
        self.dec1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)

        def weights_init_kaiming(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                weight_init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Linear') != -1:
                weight_init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif classname.find('BatchNorm2d') != -1:
                weight_init.uniform(m.weight.data, 1.0, 0.02)
                weight_init.constant(m.bias.data, 0.0)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = weight_init.kaiming_normal(m.weight.data)
        self.apply(weights_init_kaiming)

    def _initialize_module(self, pretrained):
        if not pretrained:
            return

        def init_module_weight(target, source):
            for target_block, source_block in zip(target, source):
                for t, s in zip(target_block.children(), source_block.children()):
                    if hasattr(s, 'weight'):
                        t.weight.data = s.weight.data

        resnet = models.resnet101(pretrained=True)
        init_module_weight(self.layer1, resnet.layer1)
        init_module_weight(self.layer2, resnet.layer2)
        init_module_weight(self.layer3, resnet.layer3)
        init_module_weight(self.layer4, resnet.layer4)

    def forward(self, x):
        e1 = self.conv(x)      # /2, /2, 64 --*
        e2 = self.maxpool(e1)  # /4, /4, 64 -*

        h = self.layer1(e2)    # /4, /4, 64
        e3 = self.layer2(h)    # /8, /8, 128
        h = self.layer3(e3)    # /16, /16, 256
        h = self.layer4(h)     # /32, /32, 512

        c = self.fc_conv(h)    # /32, /32, 2048
        d1 = self.dec1(c)      # /16, /16, num_classes -*
        d2 = self.dec2(d1)     # /8, /8, num_classes --*
        d2 = nn.functional.upsample_bilinear(d2, scale_factor=2)
        d3 = self.dec3(d2)     # /2, /2, num_classes
        d3 = nn.functional.upsample_bilinear(d3, scale_factor=2)

        # room type
        t = self.avgpool(h)
        t = t.view(t.size(0), -1)
        t = self.fc(t)

        return d3, t


class DilatedResFCN(nn.Module):

    def __init__(self, num_classes=5, num_room_types=11, pretrained=True, base='resnet101'):
        self.inplanes = 64
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(models.resnet.Bottleneck, 64, 3)
        self.layer2 = self._make_layer(models.resnet.Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_variant_layer(VariantBottleneck, 256, 23)
        self.layer4 = self._make_variant_layer(VariantBottleneck, 512, 3)

        # room type classification
        self.avgpool = nn.AvgPool2d(40)
        self.fc = nn.Linear(512 * VariantBottleneck.expansion, num_room_types)

        # planar segmentation
        self.fc_conv = nn.Conv2d(512 * VariantBottleneck.expansion, 2048, kernel_size=1, bias=False)
        self.dec1 = nn.ConvTranspose2d(2048, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.fuse1 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.fuse2 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1, bias=False)

        def weights_init_kaiming(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                weight_init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Linear') != -1:
                weight_init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif classname.find('BatchNorm2d') != -1:
                weight_init.uniform(m.weight.data, 1.0, 0.02)
                weight_init.constant(m.bias.data, 0.0)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = weight_init.kaiming_normal(m.weight.data)

        self.apply(weights_init_kaiming)
        self._initialize_module(pretrained)

    def _initialize_module(self, pretrained):
        if not pretrained:
            return

        def init_module_weight(target, source):
            for target_block, source_block in zip(target, source):
                for t, s in zip(target_block.children(), source_block.children()):
                    if hasattr(s, 'weight'):
                        t.weight.data = s.weight.data

        resnet = models.resnet101(pretrained=True)
        init_module_weight(self.layer1, resnet.layer1)
        init_module_weight(self.layer2, resnet.layer2)
        init_module_weight(self.layer3, resnet.layer3)
        init_module_weight(self.layer4, resnet.layer4)
        print('---> init with ResNet101')

    def forward(self, x):
        e1 = self.conv(x)      # /2, /2, 64 --*
        e2 = self.maxpool(e1)  # /4, /4, 64 -*

        h = self.layer1(e2)    # /8, /8, 64
        e3 = self.layer2(h)    # /8, /8, 128
        h = self.layer3(e3)    # /8, /8, 256
        h = self.layer4(h)     # /8, /8, 512

        c = self.fc_conv(h)    # /8, /8, 2048

        d1 = self.dec1(c)      # /4, /4, 64 -*
        f1 = self.fuse1(torch.cat((d1, e2), dim=1))  # /4, /4, 64 + 64
        d2 = self.dec2(f1)     # /2, /2, 64 --*
        f2 = self.fuse2(torch.cat((d2, e1), dim=1))  # /2, /2, 64 + 64
        d3 = self.dec3(f2)     # /, /, num_classes

        # room type
        t = self.avgpool(h)
        t = t.view(t.size(0), -1)
        t = self.fc(t)

        return d3, t

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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class VariantBottleneck(models.resnet.Bottleneck):

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__(inplanes, planes, stride, downsample)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
