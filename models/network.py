import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.saver import Saver, Predictor
from models.utils import LayoutAccuracy, EpochHistory, save_images
from models.logger import Logger
from tools import timeit


class LayoutNet():

    def __init__(self, name, model, optimizer, criterion, scheduler):
        self.name = name
        self.model = nn.DataParallel(model).cuda()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accuracy = LayoutAccuracy()
        self.tf_summary = Logger('./logs', name=name)
        self.saver = Saver(network=self)

    def train(self, train_loader, validate_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
            self.epoch = epoch
            history = EpochHistory(length=len(train_loader))
            progress = tqdm.tqdm(train_loader)

            for image, *target in progress:
                self.optimizer.zero_grad()
                losses, accuracy = self.forward(image, target)
                losses['loss'].backward()
                self.optimizer.step()

                progress.set_description('Epoch#%d' % epoch)
                progress.set_postfix(history.add(losses, accuracy))

            metrics = dict(**history.metric(),
                           **self.evaluate(validate_loader, prefix='val_'))
            print('---> Epoch#%d val_loss: %.4f, val_accuracy: %.4f' % (
                    epoch + 1, metrics['val_loss'], metrics['val_accuracy']))
            self.scheduler.step(metrics['val_loss'])
            self.summary_scalar(metrics)
            self.saver.save()

    @timeit
    def evaluate(self, data_loader, prefix=''):
        self.model.eval()
        history = EpochHistory(length=len(data_loader))
        for i, (image, *target) in enumerate(data_loader):
            losses, accuracy, output = self.forward(image, target, is_eval=True)
            history.add(losses, accuracy)
            if i == 0:
                self.summary_image(output.data, target, prefix)
        return history.metric(prefix=prefix)

    def predict(self, data_loader, name):
        predictor = Predictor(self.model)
        for i, (image, *_) in enumerate(data_loader):
            output = predictor.forward(image)
            filenames = [
                data_loader.dataset.filenames[i * len(image)]
                for e in range(len(image))
            ]
            save_images(name, (filenames, output))

    def forward(self, image, target, is_eval=False):

        def to_var(t):
            return Variable(t, volatile=is_eval).cuda()

        label, edge_map = target
        image, label = to_var(image), to_var(label)
        output = self.model(image)
        losses = self.criterion(output, label, edge_map)
        acc = self.accuracy(output, label)
        return (losses, acc, output) if is_eval else (losses, acc)

    def summary_scalar(self, metrics):
        for tag, value in metrics.items():
            self.tf_summary.scalar(tag, value, self.epoch)

    def summary_image(self, output, target, prefix):

        def to_numpy(imgs):
            return imgs.squeeze().cpu().numpy()

        label, _ = target
        _, pred = torch.max(output, 1)
        self.tf_summary.image(prefix + 'pred', to_numpy(pred), self.epoch)
        self.tf_summary.image(prefix + 'label', to_numpy(label), self.epoch)
