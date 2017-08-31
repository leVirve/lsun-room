import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.evaluate import LayoutAccuracy, EpochHistory
from models.saver import Saver
from models.utils import to_numpy, shrink_edge_width
from models.logger import Logger
from tools import timeit


class Trainer():

    max_summary_image = 20

    def __init__(self, name, model, optimizer, criterion, scheduler):
        self.name = name
        self.model = nn.DataParallel(model).cuda()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accuracy = LayoutAccuracy()
        self.tf_summary = Logger('./logs', name=name)
        self.saver = Saver(network=self)

        self.dataset_hook = shrink_edge_width
        self.summary_img = False
        self.evaluated_images = 0

        self.criterion.register_trainer(self)
        self.dataset_hook.register_trainer(self)

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
            print(('--> Epoch#%d'
                   ' val_loss: %.4f, val_accuracy: %.4f, val_iou: %.4f') % (
                    epoch + 1, metrics['val_loss'],
                    metrics['val_pixel_accuracy'], metrics['val_miou']))
            self.scheduler.step(metrics['val_loss'])
            self.dataset_hook(self, train_loader, validate_loader)
            self.summary_scalar(metrics)
            self.saver.save()

    @timeit
    def evaluate(self, data_loader, prefix='', callback=None):
        self.model.eval()
        history = EpochHistory(length=len(data_loader))

        self.evaluated_images = 0
        for i, (image, *target) in enumerate(data_loader):
            self.summary_img = self.evaluated_images < self.max_summary_image

            losses, accuracy, pred = self.forward(image, target, is_eval=True)
            history.add(losses, accuracy)
            if callback:
                callback(i, to_numpy(pred))
            elif self.summary_img:
                self.summary_image(pred, target, prefix)

            self.evaluated_images += image.size(0)

        return history.metric(prefix=prefix)

    def forward(self, image, target, is_eval=False):

        def to_var(t):
            return Variable(t, volatile=is_eval).cuda()

        label, edge_map = target
        image, label, edge_map = to_var(image), to_var(label), to_var(edge_map)

        output = self.model(image)
        _, pred = torch.max(output, 1)

        losses = self.criterion(output, pred, label, edge_map)
        acc = self.accuracy(pred, label)

        return (losses, acc, pred.data) if is_eval else (losses, acc)

    def summary_scalar(self, metrics):
        for tag, value in metrics.items():
            self.tf_summary.scalar(tag, value, self.epoch)

    def summary_image(self, pred, target, prefix):
        label, _ = target
        self.tf_summary.image(prefix + 'pred', to_numpy(pred.squeeze()),
                              self.epoch, tag_count_base=self.evaluated_images)
        self.tf_summary.image(prefix + 'label', to_numpy(label.squeeze()),
                              self.epoch, tag_count_base=self.evaluated_images)
