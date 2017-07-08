import click

import torch

import config as cfg
from lsun_room import Phase
from dataset import ImageFolderDataset
from fcn import LayoutNet


@click.command()
@click.option('--resume', type=click.Path(exists=True))
def main(resume):

    print('===> Prepare data loader')
    train_loader = torch.utils.data.DataLoader(
        dataset=ImageFolderDataset(
                    root=cfg.dataset_root,
                    target_size=(cfg.size, cfg.size),
                    phase=Phase.TRAIN),
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=True,
        shuffle=True)

    print('===> Prepare model')
    net = LayoutNet()

    print('===> Start training')
    net.train(train_loader, epochs=cfg.epochs)


if __name__ == '__main__':
    main()
