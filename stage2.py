
import click
import torch

from stage2_utils import Phase
from stage2_utils.lsun_dataset import ImageFolderDataset
from net import StageNet

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--name', type=str)
@click.option('--dataset_root', default='../data')
@click.option('--image_size', default=(404, 404), type=(int, int))
@click.option('--epochs', default=20, type=int)
@click.option('--batch_size', default=8, type=int)
@click.option('--workers', default=6, type=int)
@click.option('--resume', type=click.Path(exists=True))
def main(name, dataset_root, image_size, epochs, batch_size, workers, resume):

    print('===> Prepare data loader')
    dataset_args = {'root': dataset_root, 'target_size': image_size}
    loader_args = {'num_workers': workers, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        dataset=ImageFolderDataset(phase=Phase.TRAIN, **dataset_args),
        batch_size=batch_size, shuffle=True, **loader_args
    )
    validate_loader = torch.utils.data.DataLoader(
        dataset=ImageFolderDataset(phase=Phase.VALIDATE, **dataset_args),
        batch_size=batch_size, **loader_args
    )

    print('===> Prepare model')
    net = StageNet(name='stage2_ResFCN', stage_2=True, joint_roomtype=True,
                   roomtype_weight=0.1, edge_weight=0.1)

    pretrain_dict = torch.load('output/weight/stage1_ResFCN/20.pth')
    model_dict = net.model.state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.model.load_state_dict(model_dict)

    print('===> Start training')
    net.train(train_loader=train_loader,
              validate_loader=validate_loader,
              epochs=epochs)
    net.evaluate(data_loader=validate_loader, prefix='')
    net.predict(data_loader=validate_loader, name='FCN_seg_layout')


if __name__ == '__main__':
    main()
