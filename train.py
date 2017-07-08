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
    dataset_args = {'root': cfg.dataset_root, 'target_size': cfg.image_size}
    loader_args = {'num_workers': cfg.workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        dataset=ImageFolderDataset(phase=Phase.TRAIN, **dataset_args),
        batch_size=cfg.batch_size, shuffle=True, **loader_args)
    validate_loader = torch.utils.data.DataLoader(
        dataset=ImageFolderDataset(phase=Phase.VALIDATE, **dataset_args),
        batch_size=cfg.batch_size, **loader_args)

    print('===> Prepare model')
    net = LayoutNet()

    print('===> Start training')
    net.train(
        train_loader=train_loader,
        validate_loader=validate_loader,
        epochs=cfg.epochs)


if __name__ == '__main__':
    main()
