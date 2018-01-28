import tqdm
import torch
import onegan.losses as L
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from easydict import EasyDict
from onegan.ops import VisionConv2d, VisionConv3d
from onegan.estimator import Estimator
from onegan.utils import to_var, save_batched_images, unique_experiment_name


lap_conv2d = VisionConv2d('laplacian', padding=2, dilation=2, name='Laplacian')
sobel_y_conv2d = VisionConv2d('sobel_vertical', padding=2, dilation=2, name='Sobel-Y')
sobel_x_conv2d = VisionConv2d('sobel_horizontal', padding=2, dilation=2, name='Sobel-X')
lap_conv3d = VisionConv3d('laplacian', channel=5, padding=2, dilation=2, name='Laplacian')


def layout_edge(output, σ=5.0):
    edge = lap_conv2d(output.unsqueeze(1).float()).squeeze()
    return torch.exp(-torch.abs(edge) / σ)


def layout_gradient(output, σ=5.0):
    grad_x = sobel_y_conv2d(output.unsqueeze(1).float()).squeeze()
    grad_y = sobel_x_conv2d(output.unsqueeze(1).float()).squeeze()
    return 1 - torch.exp(-torch.sqrt(grad_x ** 2 + grad_y ** 2) / σ)


def score_edge(output, σ=1.0):
    edge = lap_conv3d(output.float()).squeeze()
    return torch.exp(-torch.abs(edge) / σ)


class RoomTrainer(Estimator):

    def __init__(self, model, optimizer, metric, save_epochs, name):
        super().__init__(model, optimizer, metric, name, save_epochs=save_epochs)
        self.hyper_params = EasyDict({
            'l1_λ': 0,
            'edge_λ': 0,
            'type_λ': 0,
            'focal_gamma': 0,
        })
        self.ce_loss = L.CrossEntropyLoss2d()
        self.focal_loss = L.FocalLoss2d(gamma=self.hyper_params.focal_gamma)

    @property
    def lr_scheduler(self):
        if not hasattr(self, '_lr_scheduler'):
            self._lr_scheduler = ReduceLROnPlateau(self.optim, patience=2, mode='min', factor=0.5, min_lr=1e-8, verbose=True)
        return self._lr_scheduler

    def objective(self, score, prediction, pred_type, target, data):

        ''' per-pixel classification loss '''
        if self.hyper_params.focal_gamma:
            seg_loss = self.focal_loss(score, target)
        else:
            seg_loss = self.ce_loss(score, target)
        terms = {'loss/loss': seg_loss, 'loss/cla': seg_loss}

        ''' area l1 loss '''
        if self.hyper_params.l1_λ:
            l_loss = F.mse_loss if self.hyper_params.use_l2 else F.l1_loss
            onehot_target = torch.zeros_like(score).scatter_(1, target.unsqueeze(1), 1)
            l1_loss = l_loss(score, onehot_target)
            terms['loss/loss'] += l1_loss * self.hyper_params.l1_λ
            terms['loss/area'] = l1_loss

        ''' layout edge mse loss '''
        if self.hyper_params.edge_λ:
            # edge_map = layout_edge(prediction)
            edge_map = layout_gradient(prediction)
            # edge_map = F.log_softmax(pred_edge, dim=1).squeeze(1)
            # edge_map = score_edge(score, σ=1.0)
            edge_loss = F.binary_cross_entropy(edge_map, to_var(data['edge']))
            terms['loss/loss'] += edge_loss * self.hyper_params.edge_λ
            terms['loss/edge'] = edge_loss

        ''' room type ce loss '''
        if self.hyper_params.type_λ:
            type_loss = F.cross_entropy(pred_type.squeeze(), to_var(data['type'].long()))
            terms['loss/loss'] += type_loss * self.hyper_params.type_λ
            terms['loss/room_type'] = type_loss

        return terms

    def train(self, data_loader, epoch, history):
        progress = tqdm.tqdm(data_loader)
        progress.set_description(f'Epoch#{epoch + 1}')

        for data in progress:
            source, target = to_var(data['image']), to_var(data['label'])
            score, pred_type = self.model(source)
            _, pred = torch.max(score, 1)

            self.optim.zero_grad()
            loss_terms = self.objective(score, pred, pred_type, target, data)
            acc_terms = self.metric(pred, target)
            loss_terms['loss/loss'].backward()
            self.optim.step()

            progress.set_postfix(history.add({**loss_terms, **acc_terms}))

            self.logger.image({
                'input': source.data, 'pred': pred.data.float(), 'target': target.data.float()},
                epoch=epoch, prefix='train_')
        self.logger.scalar(history.metric(), epoch)

    def evaluate(self, data_loader, epoch, history):
        progress = tqdm.tqdm(data_loader, leave=False)
        progress.set_description('Evaluate')

        for data in progress:
            source, target = to_var(data['image'], volatile=True), to_var(data['label'], volatile=True)
            score, pred_type = self.model(source)
            _, pred = torch.max(score, 1)

            loss_terms = self.objective(score, pred, pred_type, target, data)
            acc_terms = self.metric(pred, target)

            progress.set_postfix(history.add({**loss_terms, **acc_terms}, log_suffix='_val'))

            self.logger.image({
                'input': source.data, 'pred': pred.data.float(), 'target': target.data.float()},
                epoch=epoch, prefix='val_')

        log = history.metric()
        print({k: v for k, v in log.items() if 'acc' in k})
        self.logger.scalar(log, epoch)
        self.lr_scheduler.step(log['loss/loss_val'])

    def static_evaluate(self, data_loader):
        import onegan

        self.model.eval()
        root = unique_experiment_name('./output/results', self.name)
        history = onegan.extension.History()
        for data in tqdm.tqdm(data_loader):
            score, pred_type = self.model(to_var(data['image'], volatile=True))
            _, pred = torch.max(score, 1)
            acc_terms = self.metric(pred.unsqueeze(0), to_var(data['label']).unsqueeze(0))
            history.add({**acc_terms})
            save_batched_images(img_tensors=pred.data.float() * (255 / 5), folder=root, filenames=data["path"])
        print(history.metric())

    def dynamic_evaluate(self, data_loader):
        self.model.eval()
        import os
        import scipy.misc
        import numpy as np
        import onegan
        from PIL import Image

        def from_np_to_var(t):
            return to_var(torch.from_numpy(np.array(t)), volatile=True)

        def merge_viz(img, cgt, pred):
            h, w, c = img.shape
            canvas = np.zeros((h, 2 * w, 3), dtype=float)
            canvas[:, :w] = img.astype(float) / 255 * 0.6 + cgt.astype(float) / 255 * 0.4
            canvas[:, w:] = np.repeat(pred[:, :, np.newaxis] / 5, 3, axis=2)
            return canvas

        assert data_loader.batch_size == 1

        root = os.path.join('./output/viz_results', self.name)
        for i in range(11):
            os.makedirs(os.path.join(root, f'type{i}'), exist_ok=True)

        history = onegan.extension.History()
        class_history = [onegan.extension.History() for _ in range(11)]

        progress = tqdm.tqdm(data_loader, desc='Dynamic Evaluation')
        for i, data in enumerate(progress):
            meta = data_loader.dataset.meta[i]
            gt_path = meta['layout_path']
            cgt_path = meta['layout_path'].replace('layout_seg_images', 'layout_seg_images_color')
            assert data['path'][0] in gt_path

            score, _ = self.model(to_var(data['image'], volatile=True))
            _, pred = torch.max(score, 1)
            output = pred.data.squeeze().cpu().numpy()

            gt = Image.open(gt_path)
            small_pred = Image.fromarray(output.astype('uint8'))

            viz_result = merge_viz(
                np.array(Image.open(meta['image_path']).convert('RGB').resize(small_pred.size, Image.NEAREST)),
                np.array(Image.open(cgt_path).convert('RGB').resize(small_pred.size, Image.NEAREST)),
                np.array(small_pred))
            scipy.misc.imsave(os.path.join(root, f'type{meta["type"]}', data['path'][0]), viz_result)

            pred = small_pred.resize(gt.size, Image.NEAREST)
            acc_terms = self.metric(from_np_to_var(pred).unsqueeze(0), (from_np_to_var(gt) - 1).unsqueeze(0))
            class_history[meta['type']].add({**acc_terms})
            progress.set_postfix(history.add({**acc_terms}, log_suffix='_val'))

        print(history.metric())
        [print(class_history[i].metric()) for i in range(11)]

        return history.metric()
