import logging
from functools import partial

import torch
import onegan
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from onegan.estimator import OneEstimator
from onegan.utils import to_var


def training_estimator(model, optimizer, args):

    def layout_gradient(output, σ=5.0):
        grad_x = sobel_y_conv2d(output.unsqueeze(1).float()).squeeze()
        grad_y = sobel_x_conv2d(output.unsqueeze(1).float()).squeeze()
        return 1 - torch.exp(-torch.sqrt(grad_x ** 2 + grad_y ** 2) / σ)

    def objective(score, prediction, pred_type, target, data):
        ''' per-pixel classification loss '''
        if args.focal_gamma:
            seg_loss = focal_loss(score, target)
        else:
            seg_loss = ce_loss(score, target)
        terms = {'loss/loss': seg_loss, 'loss/cla': seg_loss}

        ''' area smoothness loss '''
        if args.l1_factor or args.l2_factor:
            l_loss = F.mse_loss if args.l2_factor else F.l1_loss
            l1_λ = args.l1_factor or args.l2_factor
            onehot_target = torch.zeros_like(score).scatter_(1, target.unsqueeze(1), 1)
            l1_loss = l_loss(score, onehot_target)
            terms['loss/loss'] += l1_loss * l1_λ
            terms['loss/area'] = l1_loss

        ''' layout edge constraint loss '''
        if args.edge_factor:
            edge_map = layout_gradient(prediction)
            edge_loss = F.binary_cross_entropy(edge_map, to_var(data['edge']))
            terms['loss/loss'] += edge_loss * args.edge_factor
            terms['loss/edge'] = edge_loss

        ''' room type ce loss '''
        # if args.type_λ:
        #     type_loss = F.cross_entropy(pred_type.squeeze(), to_var(data['type'].long()))
        #     terms['loss/loss'] += type_loss * args.type_λ
        #     terms['loss/room_type'] = type_loss

        return terms

    def _closure(model, data, volatile=False):
        source = to_var(data['image'], volatile=volatile)
        target = to_var(data['label'], volatile=volatile)
        score, pred_type = model(source)
        _, output = torch.max(score, 1)

        loss = objective(score, output, pred_type, target, data)
        accuracy = metric(output, target)

        if volatile:
            accuracy['score'] = score_metric(output, target)

        viz_results = {
            'input': source.data,
            'output': output.data.float(),
            'target': target.data.float()}
        tensorboard.image(
            viz_results,
            epoch=estimator.state['epoch'], prefix='val_' if volatile else 'train_')

        return loss, accuracy

    log = logging.getLogger(f'room.{args.name}')

    ce_loss = onegan.losses.CrossEntropyLoss2d()
    focal_loss = onegan.losses.FocalLoss2d(gamma=args.focal_gamma)

    sobel_y_conv2d = onegan.ops.VisionConv2d('sobel_vertical', padding=2, dilation=2, name='Sobel-Y')
    sobel_x_conv2d = onegan.ops.VisionConv2d('sobel_horizontal', padding=2, dilation=2, name='Sobel-X')

    scheduler = ReduceLROnPlateau(optimizer, patience=2, mode='min', factor=0.5, min_lr=1e-8, verbose=True)
    checkpoint = onegan.extension.Checkpoint(name=args.name, save_epochs=5)
    tensorboard = onegan.extension.TensorBoardLogger(name=args.name, max_num_images=30)
    metric = onegan.metrics.semantic_segmentation.Metric(num_class=args.num_class, only_scalar=True)
    score_metric = onegan.metrics.semantic_segmentation.max_bipartite_matching_score

    if args.pretrain_path:
        log.info('Load pre-trained weight as initialization')
        checkpoint.apply(args.pretrain_path, model, remove_module=False)

    log.info('Build training esimator')
    estimator = OneEstimator(
        model, optimizer,
        lr_scheduler=scheduler, logger=tensorboard, saver=checkpoint, name=args.name)

    return partial(estimator.run, update_fn=partial(_closure), inference_fn=partial(_closure, volatile=True))


def evaluation_estimator(model, args):

    def merge_viz(img, lbl, out):
        image = (img / 2 + .5)
        layout = colorizer.apply(lbl)
        output = colorizer.apply(out)
        if args.tri_visual:
            batch, c, h, w = image.size()
            pad = torch.ones(batch, c, h, 5)
            return torch.cat([image, pad, layout, pad, output], dim=3)
        return torch.cat([image * .6 + layout * .4, output], dim=3)

    def _closure(model, data, volatile=False):
        source = to_var(data['image'], volatile=volatile)
        target = to_var(data['label'], volatile=volatile)
        score, pred_type = model(source)
        _, output = torch.max(score, 1)

        set_gallery.image(merge_viz(data['image'], data['label'], output.data.cpu()), filenames=data['filename'])
        gallery.image(output.data.float() * (255 / 5), filenames=data['filename'])

        accuracy = metric(output, target)
        accuracy['score'] = score_metric(output, target)

        return accuracy

    log = logging.getLogger(f'room.{args.name}')
    colors = [[249, 69, 93], [255, 229, 170], [144, 206, 181], [81, 81, 119], [241, 247, 210]]

    colorizer = onegan.extension.Colorizer(colors=colors)
    gallery = onegan.extension.ImageSaver(name=args.name)
    set_gallery = onegan.extension.ImageSaver(savedir='./output/viz_results', name=args.name)
    checkpoint = onegan.extension.Checkpoint(name=args.name, save_epochs=5)
    metric = onegan.metrics.semantic_segmentation.Metric(num_class=args.num_class, only_scalar=True)
    score_metric = onegan.metrics.semantic_segmentation.max_bipartite_matching_score

    log.info('Build evaluation esimator')
    checkpoint.apply(args.pretrain_path, model, remove_module=False)
    estimator = OneEstimator(model, name=args.name)

    return partial(estimator.evaluate, inference_fn=partial(_closure, volatile=True))


def weights_estimator(model, args):

    def _closure(model, data, volatile=False):
        source = to_var(data['image'], volatile=volatile)
        score, pred_type = model(source)
        _, output = torch.max(score, 1)
        score = score_metric(output, data['label'])
        return {'score': score}

    def searching(data_loader):
        max_accuracy, best_path = 0, None

        for state_dict, path in searcher.get_weight():
            model.load_state_dict(state_dict)
            estimator.model = model
            history = estimator.evaluate(data_loader, inference_fn=partial(_closure, volatile=True))
            monitor_value = history['score_val']
            if monitor_value > max_accuracy:
                max_accuracy = monitor_value
                best_path = path
            print('->', path, monitor_value)

        print(f'Best weight: {best_path} (error: {1 - max_accuracy:.04f}%)')

    log = logging.getLogger(f'room.{args.name}')

    searcher = onegan.extension.WeightSearcher(args.pretrain_path)
    score_metric = onegan.metrics.semantic_segmentation.max_bipartite_matching_score

    log.info('Build weights esimator')
    estimator = OneEstimator(model, name=args.name)

    return searching
