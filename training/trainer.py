import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from training.saver import Checkpoint
from training.utils import EpochHistory, to_numpy
from tools import timeit


class Trainer():

    max_summary_image = 20

    def __init__(self, model, optimizer, criterion, accuracy, scheduler, logger):
        self.model = nn.DataParallel(model).cuda()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accuracy = accuracy
        self.logger = logger
        self.saver = Checkpoint()

        self.start_epoch = 0
        self.dataset_hook = None
        self.summary = False
        self.evaluated_images = 0

        self.criterion.register_trainer(self)
        self.saver.register_trainer(self)

    def train(self, train_loader, validate_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
            self.epoch = self.start_epoch + epoch
            history = EpochHistory(length=len(train_loader))
            progress = tqdm.tqdm(train_loader)

            for data in progress:
                self.optimizer.zero_grad()
                losses, accuracies = self.forward(data)
                losses['loss'].backward()
                self.optimizer.step()

                progress.set_description('Epoch#%d' % (self.epoch + 1))
                progress.set_postfix(history.add(losses, accuracies))

            train_metrics = history.metric()
            valid_metrics = self.evaluate(validate_loader)

            if self.dataset_hook:
                self.dataset_hook(self, train_loader, validate_loader)
            self.scheduler.step(valid_metrics['loss'])
            self.summary_scalar(train_metrics)
            self.summary_scalar(valid_metrics, prefix='val_')
            self.saver.save()

    @timeit
    def evaluate(self, data_loader, callback=None):
        self.model.eval()
        history = EpochHistory(length=len(data_loader))

        self.evaluated_images = 0
        for i, data in enumerate(data_loader):
            self.summary = self.evaluated_images < self.max_summary_image
            losses, accuracies = self.forward(
                data,
                hook=(lambda x: callback(i, to_numpy(x))) if callback else None)
            history.add(losses, accuracies)
            self.evaluated_images += len(data)

        metrics = history.metric()
        print('--> Epoch#%d' % (self.epoch + 1), end=' ')
        print('val_loss: %.4f, val_accuracy: %.4f, val_iou: %.4f' % (
              metrics['loss'], metrics['pixel_accuracy'], metrics['miou']))
        return metrics

    def forward(self, item, hook=None):

        def to_var(t):
            return Variable(t, volatile=not self.model.training).cuda()

        def summarize(pred_edge=None):
            if not self.summary or hook:
                return
            data = {'val_image': item['image'],
                    'val_pred_label': pred.data.squeeze(),
                    'val_label': item['label'].squeeze()}
            if pred_edge is not None:
                data.update({'val_pred_edge': pred_edge.data,
                             'val_edge': item['edge']})
            self.summary_image(data)

        label = to_var(item['label'])
        score = self.model(to_var(item['image']))
        _, pred = torch.max(score, 1)

        losses = self.criterion(score, pred, label, item, end_hook=summarize)
        accuracies = self.accuracy(pred, label)

        if hook:
            hook(pred.data)
        return losses, accuracies

    def summary_scalar(self, metrics, prefix=''):
        for tag, value in metrics.items():
            self.logger.scalar(prefix + tag, value, self.epoch)

    def summary_image(self, images, prefix=''):
        for tag, image in images.items():
            self.logger.image(prefix + tag, to_numpy(image), self.epoch,
                              tag_count_base=self.evaluated_images)
