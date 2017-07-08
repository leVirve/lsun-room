import os
import skimage
import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

import config as cfg
from logger import Logger


class LayoutNet():

    def __init__(self, weight=None):
        self.tf_summary = Logger('./logs', name='pytorch')
        self.model = FCN(num_classes=5).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.cross_entropy_criterion = nn.NLLLoss2d(weight=weight)
        self.l1_criterion = nn.L1Loss()

    def train(self, train_loader, validate_loader, epochs):
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            self.model.train()
            progress = tqdm.tqdm(train_loader)
            for image, target in progress:
                loss, accuracy = self.train_on_batch(to_var(image), to_var(target))
                self.on_batch_end(progress, loss, accuracy)
            self.on_epoch_end(train_loader, validate_loader)

    def train_on_batch(self, image, target):
        self.optimizer.zero_grad()
        output = self.model(image)
        loss = self.pixelwise_loss(output, target)
        loss.backward()
        self.optimizer.step()
        accuracy = self.pixelwise_accuracy(output, target)
        return loss, accuracy

    def on_epoch_end(self, train_loader, validate_loader):
        metrics = dict(**self.evaluate(train_loader),
                       **self.evaluate(validate_loader, prefix='val_'))
        for tag, value in metrics.items():
            self.tf_summary.scalar(tag, value, self.epoch)
        print('---> Epoch#{} loss: {loss:.4f}, accuracy={accuracy:.4f}'
              ' val_loss: {val_loss:.4f}, val_accuracy={val_accuracy:.4f}'
              .format(self.epoch, **metrics))
        self.save_model()

    def on_batch_end(self, progress, loss, accuracy):
        progress.set_description('Epoch#%i' % self.epoch)
        progress.set_postfix(
            loss='%.04f' % loss.data[0],
            accuracy='%.04f' % accuracy.data[0])

    def evaluate(self, data_loader, prefix=''):
        self.model.eval()
        loss, accuracy = 0, 0
        for i, (image, target) in enumerate(data_loader):
            image, target = to_var(image), to_var(target)
            output = self.model(image)
            loss += self.pixelwise_loss(output, target).data[0]
            accuracy += self.pixelwise_accuracy(output, target).data[0]
            if i == 0:
                self.summay_output(output, target, prefix)
        return {prefix + 'loss': loss / len(data_loader),
                prefix + 'accuracy': accuracy / len(data_loader)}

    def predict(self, data_loader, name):
        self.model.eval()
        layout_folder = 'output/layout/%s/' % name
        os.makedirs(layout_folder, exist_ok=True)
        for i, (image, target) in enumerate(data_loader):
            fn = data_loader.dataset.filenames[i]
            image, target = to_var(image), to_var(target)
            output = self.model(image)
            _, output = torch.max(output, 1)
            skimage.io.imsave(layout_folder + '%s.png' % fn, output)

    def summay_output(self, output, target, prefix):
        _, output = torch.max(output, 1)
        self.tf_summary.image(prefix + 'output', output.data, self.epoch)
        self.tf_summary.image(prefix + 'target', target.data, self.epoch)

    def pixelwise_loss(self, pred, target):
        log_pred = F.log_softmax(pred)
        xent_loss = self.cross_entropy_criterion(log_pred, target)

        onehot_target = (
            torch.FloatTensor(pred.size())
            .zero_().cuda()
            .scatter_(1, target.data.unsqueeze(1), 1))
        l1_loss = self.l1_criterion(pred, Variable(onehot_target))

        return xent_loss + cfg.λ * l1_loss

    def pixelwise_accuracy(self, output, target):
        _, output = torch.max(output, 1)
        return (output == target).float().mean()

    def load_model(self, path):
        self.model = torch.load(path)

    def save_model(self):
        os.makedirs('output/weight', exist_ok=True)
        torch.save(self.model.state_dict(), 'output/weight/%d.pth' % self.epoch)


def to_var(tensor):
    return Variable(tensor).cuda()


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
