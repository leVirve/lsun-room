import os
import skimage
import tqdm
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

from tools import Logger, timeit


class LayoutNet():

    def __init__(self, name, criterion):
        self.name = name
        self.model = nn.DataParallel(FCN(num_classes=5)).cuda()
        self.criterion = criterion
        self.accuracy = LayoutAccuracy()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.tf_summary = Logger('./logs', name=name)

    def train(self, train_loader, validate_loader, epochs):
        for epoch in range(1, epochs + 1):

            self.model.train()
            self.epoch = epoch
            history = EpochHistory(length=len(train_loader))
            progress = tqdm.tqdm(train_loader)

            for image, target in progress:
                self.optimizer.zero_grad()
                loss, acc = self.forward(image, target)
                loss.backward()
                self.optimizer.step()

                history.add(loss, acc)
                progress.set_description('Epoch#%i' % epoch)
                progress.set_postfix(
                    loss='%.04f' % loss.data[0],
                    accuracy='%.04f' % acc.data[0])

            metrics = dict(**history.metric(),
                           **self.evaluate(validate_loader, prefix='val_'))
            print('---> Epoch#{} loss: {loss:.4f}, accuracy={accuracy:.4f}'
                  ' val_loss: {val_loss:.4f}, val_accuracy={val_accuracy:.4f}'
                  .format(self.epoch, **metrics))

            self.summary_scalar(metrics)
            self.save_model()

    @timeit
    def evaluate(self, data_loader, prefix=''):
        self.model.eval()
        history = EpochHistory(length=len(data_loader))
        for i, (image, target) in enumerate(data_loader):
            loss, acc, output = self.forward(image, target, is_eval=True)
            history.add(loss, acc)
            if i == 0:
                self.summary_image(output.data, target, prefix)
        return history.metric(prefix=prefix)

    def predict(self, data_loader, name):
        self.model.eval()
        layout_folder = 'output/layout/%s/' % name
        os.makedirs(layout_folder, exist_ok=True)
        for i, (image, _) in enumerate(data_loader):
            output = self.model(Variable(image, volatile=True).cuda())
            _, output = torch.max(output, 1)
            fn = data_loader.dataset.filenames[i]
            skimage.io.imsave(layout_folder + '%s.png' % fn, output)

    def forward(self, image, target, is_eval=False):

        def to_var(t):
            return Variable(t, volatile=is_eval).cuda()

        image, target = to_var(image), to_var(target)
        output = self.model(image)
        loss = self.criterion(output, target)
        acc = self.accuracy(output, target)
        return (loss, acc, output) if is_eval else (loss, acc)

    def summary_scalar(self, metrics):
        for tag, value in metrics.items():
            self.tf_summary.scalar(tag, value, self.epoch)

    def summary_image(self, output, target, prefix):

        def to_numpy(imgs):
            return imgs.squeeze().cpu().numpy()

        _, output = torch.max(output, 1)
        self.tf_summary.image(prefix + 'output', to_numpy(output), self.epoch)
        self.tf_summary.image(prefix + 'target', to_numpy(target), self.epoch)

    def load_model(self, path):
        self.model = torch.load(path)

    def save_model(self):
        folder = 'output/weight/%s' % self.name
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model.state_dict(), folder + '/%d.pth' % self.epoch)


class EpochHistory():

    def __init__(self, length):
        self.count = 0
        self.len = length
        self.losses = np.zeros(self.len)
        self.accuracies = np.zeros(self.len)

    def add(self, loss, acc):
        self.losses[self.count] = loss.data[0]
        self.accuracies[self.count] = acc.data[0]
        self.count += 1

    def mean(self):
        return self.losses.mean(), self.accuracies.mean()

    def metric(self, prefix=''):
        loss, accuracy = self.mean()
        return {prefix + 'loss': loss,
                prefix + 'accuracy': accuracy}


class LayoutAccuracy():

    def __call__(self, output, target):
        return self.pixelwise_accuracy(output, target)

    def pixelwise_accuracy(self, output, target):
        _, output = torch.max(output, 1)
        return (output == target).float().mean()


class LayoutLoss():

    def __init__(self, λ=0.1, weight=None):
        self.λ = λ
        self.cross_entropy_criterion = nn.NLLLoss2d(weight=weight).cuda()
        self.l1_criterion = nn.L1Loss().cuda()

    def __call__(self, pred, target):
        return self.pixelwise_loss(pred, target)

    def pixelwise_loss(self, pred, target):
        log_pred = F.log_softmax(pred)
        xent_loss = self.cross_entropy_criterion(log_pred, target)

        onehot_target = (
            torch.FloatTensor(pred.size())
            .zero_().cuda()
            .scatter_(1, target.data.unsqueeze(1), 1))
        l1_loss = self.l1_criterion(pred, Variable(onehot_target))
        return xent_loss + self.λ * l1_loss


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
        x = x[:, :, 44:44 + x.size()[2], 44:44 + x.size()[3]].contiguous()
        return x
