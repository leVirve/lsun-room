import importlib
import logging

import torch
import onegan

from trainer import core
from trainer.賣扣老師 import build_resnet101_FCN
from trainer.model import ResFCN, VggFCN, LaFCN, DilatedResFCN


def create_dataset(args):
    assert args.batch_size > 1

    module = importlib.import_module(f'datasets.{args.dataset}')
    Dataset = getattr(module, {
        'sunrgbd': 'SunRGBDDataset',
        'lsunroom': 'LsunRoomDataset',
        'hedau': 'HedauDataset',
    }[args.dataset])
    args.num_class = Dataset.num_classes

    return Dataset('train', args=args).to_loader(args=args), Dataset('val', args=args).to_loader(args=args)


def create_model(args):
    return {
        'vgg': lambda: VggFCN(
            num_class=args.num_class, input_size=(args.image_size, args.image_size), pretrained=True, base='vgg16_bn'),
        'lavgg': lambda: LaFCN(
            num_classes=args.num_class, input_size=None, pretrained=True, base='vgg16_bn'),
        'resnet': lambda: ResFCN(
            num_classes=args.num_class, num_room_types=11, pretrained=True, base='resnet101'),
        'drn': lambda: DilatedResFCN(
            num_classes=args.num_class, num_room_types=11, pretrained=True, base='resnet101'),
        'mike': lambda: build_resnet101_FCN(
            pretrained=True, nb_classes=37, stage_2=True, joint_class=not args.disjoint_class)
    }[args.arch]()


def create_optim(args, model, optim='sgd'):
    return {
        'adam': lambda: torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999)),
        'sgd': lambda: torch.optim.SGD(model.parameters(), lr=args.lr / 10, momentum=0.9)
    }[optim]()


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
    log = logging.getLogger('room')
    log.info(''.join([f'\n-- {k}: {v}' for k, v in vars(args).items()]))

    train_loader, val_loader = create_dataset(args)
    model = create_model(args)

    if args.phase == 'train':
        training_estimator = core.training_estimator(
            torch.nn.DataParallel(model.cuda()),
            create_optim(args, model, optim=args.optim), args)
        training_estimator(train_loader, val_loader, epochs=args.epoch)

    if args.phase in ['eval', 'eval_search']:
        core_fn = core.evaluation_estimator if args.phase == 'eval' else core.weights_estimator
        evaluate_estimator = core_fn(torch.nn.DataParallel(model.cuda()), args)
        evaluate_estimator(val_loader)


if __name__ == '__main__':
    parser = onegan.option.Parser(description='Indoor room corner detection', config='./config.yml')
    parser.add_argument('--name', help='experiment name')
    parser.add_argument('--folder', help='where\'s the dataset')
    parser.add_argument('--dataset', default='lsunroom', choices=['lsunroom', 'hedau', 'sunrgbd'])
    parser.add_argument('--phase', default='eval', choices=['train', 'eval', 'eval_search'])

    # data
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--use_edge', action='store_true')
    parser.add_argument('--use_corner', action='store_true')
    parser.add_argument('--datafold', type=int, default=1)

    # outout
    parser.add_argument('--tri_visual', action='store_true')

    # network
    parser.add_argument('--arch', default='resnet')
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--disjoint_class', action='store_true')
    parser.add_argument('--pretrain_path', default='')

    # hyper-parameters
    parser.add_argument('--l1_factor', type=float, default=0.0)
    parser.add_argument('--l2_factor', type=float, default=0.0)
    parser.add_argument('--edge_factor', type=float, default=0.0)
    parser.add_argument('--focal_gamma', type=float, default=0)
    args = parser.parse()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.StreamHandler(), ])
    main(args)
