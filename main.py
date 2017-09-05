import click
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from models.utils import save_batched_images
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
    model = models.fcn.ResFCN(num_classes=5, input_size=image_size, base='resnet50')

    criterion = models.loss.LayoutLoss(l1_λ=l1_weight, edge_λ=edge_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, mode='min',
                                     factor=0.5, min_lr=1e-8, verbose=True)

    trainer = models.network.Trainer(
        name, model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=lr_scheduler)

    print('===> Start training')
    trainer.train(
        train_loader=train_loader,
        validate_loader=validate_loader,
        epochs=epochs)

    print('===> Generate validation results')
    fnames = validate_loader.dataset.filenames

    def save_prediction(batch_id, imgs):
        s, e = batch_id * batch_size, (batch_id + 1) * batch_size
        save_batched_images(imgs, filenames=fnames[s:e], folder=name)

    trainer.evaluate(
        validate_loader,
        callback=save_prediction)


if __name__ == '__main__':
    main()
