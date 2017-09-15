import click
import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image

import training
from training.utils import Logger, save_batched_images, shrink_edge_width
from datasets.transform import ToLabel, Clamp

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(9487)


@click.command()
@click.option('--name', type=str)
@click.option('--dataset', default='lsun_room', type=str)
@click.option('--dataset_root', default='../data/lsun_room/')
@click.option('--log_dir', default='logs', type=str)
@click.option('--image_size', default=(404, 404), type=(int, int))
@click.option('--epochs', default=50, type=int)
@click.option('--batch_size', default=4, type=int)
@click.option('--workers', default=8, type=int)
@click.option('--l1_weight', default=0.1, type=float)
@click.option('--edge_weight', default=0.1, type=float)
@click.option('--resume', type=click.Path(exists=True))
def main(name, dataset, dataset_root, log_dir,
         image_size, epochs, batch_size, workers,
         l1_weight, edge_weight, resume):

    def get_dataset(dataset_name, **kwargs):
        if dataset_name == 'lsun_room':
            from datasets.lsun_room.folder import ImageFolderDataset
        elif dataset_name == 'sun_rgbd':
            pass
        elif dataset_name == 'lip':
            from datasets.lip.folder import ImageFolderDataset
        if len(kwargs) == 0:
            return ImageFolderDataset.num_classes
        return ImageFolderDataset(**kwargs)

    print('===> Prepare data loader')
    num_classes = get_dataset(dataset)
    input_transform = transforms.Compose([
        transforms.Scale(image_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Scale(image_size, interpolation=Image.NEAREST),
        ToLabel(),
        Clamp(1, label_max=num_classes)
    ])

    dataset_args = {'root': dataset_root, 'target_size': image_size,
                    'input_transform': input_transform,
                    'target_transform': target_transform}
    loader_args = {'num_workers': workers, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(dataset, phase='train', **dataset_args),
        batch_size=batch_size, shuffle=True, **loader_args)
    validate_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(dataset, phase='val', **dataset_args),
        batch_size=batch_size, **loader_args)

    print('===> Prepare model')
    if dataset == 'lsun_room':
        Criterion = training.criterion.LayoutLoss
    else:
        Criterion = training.criterion.SegmentLoss

    logger = Logger(log_dir, name=name)
    model = training.models.VggFCN(num_classes=num_classes, input_size=image_size)

    accuracy = training.criterion.Accuracy(num_classes=num_classes)
    criterion = Criterion(l1_λ=l1_weight, edge_λ=edge_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, mode='min',
                                     factor=0.5, min_lr=1e-8, verbose=True)

    agent = training.trainer.Trainer(
        model,
        optimizer=optimizer,
        criterion=criterion,
        accuracy=accuracy,
        scheduler=lr_scheduler,
        logger=logger)

    if dataset == 'lsun_room':
        agent.dataset_hook = shrink_edge_width

    if resume:
        agent.saver.load(resume)
        print('==> load checkpoint at', resume)

    print('===> Start training')
    agent.train(
        train_loader=train_loader,
        validate_loader=validate_loader,
        epochs=epochs)

    print('===> Generate validation results')
    fnames = validate_loader.dataset.filenames

    def save_prediction(batch_id, imgs):
        s, e = batch_id * batch_size, (batch_id + 1) * batch_size
        save_batched_images(imgs, filenames=fnames[s:e], folder=name)

    agent.evaluate(
        validate_loader,
        callback=save_prediction)


if __name__ == '__main__':
    main()
