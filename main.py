import argparse
import importlib
from pprint import pprint

import onegan
import pytorch_lightning as pl

from trainer import core


def create_dataset(args):
    assert args.batch_size > 1

    module = importlib.import_module(f'datasets.{args.dataset}')
    Dataset = getattr(module, {
        'sunrgbd': 'SunRGBDDataset',
        'lsunroom': 'LsunRoomDataset',
        'hedau': 'HedauDataset',
    }[args.dataset])
    # args.num_class = Dataset.num_classes
    kwargs = {'collate_fn': onegan.io.universal_collate_fn}

    return (Dataset(phase, args=args).to_loader(**kwargs)
            for phase in ['train', 'val'])


def hyperparams_search(args):
    search_hyperparams = {
        'arch': ['vgg', 'mike'],
        'image_size': [320],
        'edge_factor': [0, 0.2, 0.4],
    }

    import itertools
    for i, params in enumerate(itertools.product(*search_hyperparams.values())):
        for key, val in zip(search_hyperparams.keys(), params):
            args[key] = val
        args.name = '{}{}_e{}_g{}'.format(*params)
        print(f'Experiment#{i + 1}:', args.name)
        main(args)


def main(args):
    pprint(args)

    train_loader, val_loader = create_dataset(args)
    if args.phase == 'train':
        model = core.LayoutSeg(
            lr=args.lr, backbone=args.backbone,
            l1_factor=args.l1_factor, l2_factor=args.l2_factor, edge_factor=args.edge_factor)
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=args.epoch,
            weights_save_path=f'ckpts/{args.name}',
            logger=pl.loggers.TensorBoardLogger('ckpts/tb_logs', name=args.name)
        )
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Indoor room corner detection')
    parser.add_argument('--name', help='experiment name')
    parser.add_argument('--folder', default='data/lsun_room', help='where is the dataset')
    parser.add_argument('--dataset', default='lsunroom', choices=['lsunroom', 'hedau', 'sunrgbd'])
    parser.add_argument('--phase', default='eval', choices=['train', 'eval', 'eval_search'])
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)

    # data
    parser.add_argument('--image_size', default=320, type=int)
    parser.add_argument('--use_edge', action='store_true')
    parser.add_argument('--use_corner', action='store_true')
    parser.add_argument('--datafold', type=int, default=1)

    # outout
    parser.add_argument('--tri_visual', action='store_true')

    # network
    parser.add_argument('--arch', default='resnet')
    parser.add_argument('--backbone', default='resnet101')
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--disjoint_class', action='store_true')
    parser.add_argument('--pretrain_path', default='')

    # hyper-parameters
    parser.add_argument('--l1_factor', type=float, default=0.0)
    parser.add_argument('--l2_factor', type=float, default=0.0)
    parser.add_argument('--edge_factor', type=float, default=0.0)
    parser.add_argument('--type_factor', type=float, default=0.0)
    parser.add_argument('--focal_gamma', type=float, default=0)
    args = parser.parse_args()

    main(args)
