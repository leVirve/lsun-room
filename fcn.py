import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

from logger import Logger


class LayoutNet():

    def __init__(self, weight=None):
        self.log = Logger('./logs')
        self.model = FCN(num_classes=5).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.cross_entropy_criterion = nn.NLLLoss2d(weight=weight)

    def pixelwise_loss(self, pred, target):
        log_pred = F.log_softmax(pred)
        xent_loss = self.cross_entropy_criterion(log_pred, target)
        return xent_loss

    def pixelwise_accuracy(self, pred, target):
        _, pred = torch.max(pred, 1)
        return (pred == target).float().mean()

    def train(self, train_data, epochs):
        for epoch in range(1, epochs + 1):
            progress = tqdm.tqdm(train_data)

            for img, lbl in progress:
                img, lbl = Variable(img).cuda(), Variable(lbl).cuda()

                self.optimizer.zero_grad()
                pred = self.model(img)
                loss = self.pixelwise_loss(pred, lbl)
                loss.backward()
                self.optimizer.step()

                accuracy = self.pixelwise_accuracy(pred, lbl)

                loss = loss.data[0]
                accuracy = accuracy.data[0]

                progress.set_description('Epoch#%i' % epoch)
                progress.set_postfix(
                    loss='%.02f' % loss,
                    accuracy='%.02f' % accuracy)

            print('===> Epoch#{} val_loss: {:.4f}, val_accuracy={:.2f}'.format(
                  epoch, 0, 0))

    def evaluate(self):
        pass


class FCN(nn.Module):

    def __init__(self, num_classes=5):
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
        self._initialize_weights()

    def _initialize_weights(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
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
        x = x[:, :, 44:44 + x.size()[2], 44:44 + x.size()[3]].contiguous()
        return x
