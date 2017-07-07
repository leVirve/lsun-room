import click
import numpy as np

import torch
from torch.autograd import Variable
from tqdm import tqdm

import config as cfg
from lsun_room import Phase
from dataset import ImageFolderDataset
from fcn import FCN, cross_entropy2d, sparse_pixelwise_accuracy


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
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=True,
        shuffle=True)

    print('===> Prepare model')
    model = FCN(num_classes=5).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print('===> Start training')
    losses = np.zeros(len(train_loader))
    accuracies = np.zeros(len(train_loader))
    for epoch in range(1, cfg.epochs + 1):
        epoch_range = tqdm(train_loader)
        for i, (img, lbl) in enumerate(epoch_range):
            optimizer.zero_grad()

            pred = model(Variable(img).cuda())
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
