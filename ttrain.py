import click
import numpy as np

import config as cfg
import torch
import torch.nn.functional as F
from dataset import ImageFolderDataset
from fcn import FCN
from lsun_room import Phase
from torch.autograd import Variable
from tqdm import tqdm


def cross_entropy2d(pred, target, weight=None, size_average=True):
    n, c, h, w = pred.size()

    log_p = F.log_softmax(pred)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.transpose(1, 2).transpose(2, 3).contiguous().view(-1)

    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)

    if size_average:
        loss /= (h * w * n)
    return loss


def sparse_pixelwise_accuracy(pred, target):
    n, num_classes, h, w = pred.size()

    _pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, num_classes)
    _target = target.view(-1)

    _pred = _pred.data.max(1)[1]
    correct = _pred.eq(_target).sum()

    return correct / (h * w * n)


@click.command()
@click.option('--resume', type=click.Path(exists=True))
def main(resume):

    print('===> Loading dataset')
    train_dataset = ImageFolderDataset(
        root=cfg.dataset_root,
        target_size=(cfg.size, cfg.size),
        phase=Phase.TRAIN)

    print('===> Prepare data loader')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=10,
        num_workers=4,
        pin_memory=True,
        shuffle=True)

    print('===> Prepare model')
    model = FCN(num_classes=5)

    print('===> Compile model')
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print('===> Start training')
    losses, accuracies = np.zeros(len(train_loader)), np.zeros(len(train_loader))
    for epoch in range(1, cfg.epochs + 1):
        epoch_range = tqdm(train_loader)
        for i, (img, lbl) in enumerate(epoch_range):
            optimizer.zero_grad()

            img = Variable(img).cuda()

            pred = model(img)
            loss = cross_entropy2d(pred, Variable(lbl).cuda())

            loss.backward()
            optimizer.step()

            accuracy = sparse_pixelwise_accuracy(pred, lbl.cuda())

            losses[i], accuracies[i] = loss.data[0], accuracy
            avg_loss = np.mean(losses[losses > 0])
            avg_accuracy = np.mean(accuracies[accuracies > 0])

            epoch_range.set_description('Epoch# %i' % epoch)
            epoch_range.set_postfix(
                loss=loss.data[0], accuracy=accuracy,
                avg_loss=avg_loss, avg_accuracy=avg_accuracy)


if __name__ == '__main__':
    main()
