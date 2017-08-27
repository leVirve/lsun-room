import click
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.fcn import FCN
from models.network import LayoutNet
from models.loss import LayoutLoss
from datasets.lsun_room.folder import ImageFolderDataset

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--name', type=str)
@click.option('--dataset_root', default='../data/lsun_room/')
@click.option('--image_size', default=(404, 404), type=(int, int))
@click.option('--epochs', default=50, type=int)
@click.option('--batch_size', default=4, type=int)
@click.option('--workers', default=8, type=int)
@click.option('--l1_weight', default=0.1, type=float)
@click.option('--edge_weight', default=0.1, type=float)
@click.option('--resume', type=click.Path(exists=True))
def main(name, dataset_root,
         image_size, epochs, batch_size, workers,
         l1_weight, edge_weight, resume):

    print('===> Prepare data loader')
    dataset_args = {'root': dataset_root, 'target_size': image_size}
    loader_args = {'num_workers': workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        dataset=ImageFolderDataset(phase='train', **dataset_args),
        batch_size=batch_size, shuffle=True, **loader_args)
    validate_loader = torch.utils.data.DataLoader(
        dataset=ImageFolderDataset(phase='val', **dataset_args),
        batch_size=batch_size, **loader_args)

    print('===> Prepare model')
    model = FCN(num_classes=5, input_size=image_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    net = LayoutNet(
            name,
            model,
            optimizer=optimizer,
            criterion=LayoutLoss(l1_λ=l1_weight, edge_λ=edge_weight),
            scheduler=ReduceLROnPlateau(
                optimizer, patience=2, mode='min', min_lr=1e-12, verbose=True)
        )

    print('===> Start training')
    net.train(
        train_loader=train_loader,
        validate_loader=validate_loader,
        epochs=epochs)
    net.predict(validate_loader, name=name)


if __name__ == '__main__':
    main()
