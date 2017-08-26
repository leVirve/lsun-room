import os
import skimage
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.fcn import FCN
from models.utils import LayoutAccuracy, EpochHistory
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

            for image, target, edge_map in progress:
                self.optimizer.zero_grad()
                loss, loss_term, acc = self.forward(image, target, edge_map)
                loss.backward()
                self.optimizer.step()

                history.add(loss, loss_term, acc)
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
        for i, (image, target, edge_map) in enumerate(data_loader):
            loss, loss_term, acc, output = self.forward(image, target, edge_map, is_eval=True)
            history.add(loss, loss_term, acc)
            if i == 0:
                self.summary_image(output.data, target, prefix)
        return history.metric(prefix=prefix)

    def predict(self, data_loader, name):
        self.model.eval()
        layout_folder = 'output/layout/%s/' % name
        os.makedirs(layout_folder, exist_ok=True)
        for i, (image, _, _) in enumerate(data_loader):
            output = self.model(Variable(image, volatile=True).cuda())
            _, output = torch.max(output, 1)
            fn = data_loader.dataset.filenames[i]
            skimage.io.imsave(layout_folder + '%s.png' % fn, output)

    def forward(self, image, target, edge_map, is_eval=False):

        def to_var(t):
            return Variable(t, volatile=is_eval).cuda()

        image, target = to_var(image), to_var(target)
        output = self.model(image)
        loss, loss_term = self.criterion(output, target, edge_map)
        acc = self.accuracy(output, target)
        return (loss, loss_term, acc, output) if is_eval else (loss, loss_term, acc)

    def summary_scalar(self, metrics):
        for tag, value in metrics.items():
            self.tf_summary.scalar(tag, value, self.epoch - 1)

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
