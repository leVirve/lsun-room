import onegan
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from kornia.filters import sobel
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid

from .model import ResPlanarSeg


class LayoutSeg(pl.LightningModule):

    def __init__(
        self, lr: float = 1e-4, backbone: str = 'resnet101',
        l1_factor: float = 0.2, l2_factor: float = 0.0, edge_factor: float = 0.2
    ):
        super().__init__()
        self.lr = lr
        self.model = ResPlanarSeg(pretrained=True, backbone=backbone)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.edge_factor = edge_factor
        self.save_hyperparameters()

    def forward(self, inputs):
        scores = self.model(inputs)
        _, outputs = torch.max(scores, 1)
        return scores, outputs

    def training_step(self, batch, batch_idx):
        inputs = batch['image']
        targets = batch['label']
        scores, outputs = self(inputs)
        loss_terms = self.criterion(scores, outputs, targets, batch)

        if self.global_step % 100 == 0:
            self.logger.experiment.add_image(
                'train_input', make_grid(inputs, nrow=4, normalize=True), self.global_step)
            self.logger.experiment.add_image(
                'train_prediction', make_grid(label_as_rgb_visual(outputs), nrow=4), self.global_step)
            self.logger.experiment.add_image(
                'train_target', make_grid(label_as_rgb_visual(targets), nrow=4), self.global_step)

        loss = loss_terms['loss/loss']
        loss_terms = {f'train_{k}': v for k, v in loss_terms.items()}
        self.log_dict(loss_terms, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        targets = batch['label']
        scores, outputs = self(inputs)

        metric_terms = self.metric(outputs, targets)
        self.log_dict(metric_terms, logger=True)

        loss_terms = self.criterion(scores, outputs, targets, batch)
        loss = loss_terms['loss/loss']
        loss_terms = {f'val_{k}': v for k, v in loss_terms.items()}
        self.log_dict(loss_terms, logger=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        inputs = batch['image']
        targets = batch['label']
        _, outputs = self(inputs)

        metric_terms = self.metric(outputs, targets)
        logger.info(f'batch#{batch_idx}: {metric_terms}')
        return metric_terms

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer),
            'monitor': 'val_loss/loss'
        }

    def criterion(self, score, prediction, target, data):
        def layout_gradient(output, σ=5.0):
            return 1 - torch.exp(-sobel(output.unsqueeze(1).float()) / σ)

        loss = 0
        terms = {}
        ''' per-pixel classification loss '''
        seg_loss = F.nll_loss(F.log_softmax(score, dim=1), target, ignore_index=255)
        loss += seg_loss
        terms['loss/cla'] = seg_loss

        ''' area smoothness loss '''
        if self.l1_factor or self.l2_factor:
            l_loss = F.mse_loss if self.l2_factor else F.l1_loss
            l1_λ = self.l1_factor or self.l2_factor
            # TODO ignore 255
            onehot_target = torch.zeros_like(score).scatter_(1, target.unsqueeze(1), 1)
            l1_loss = l_loss(score, onehot_target)
            loss += l1_loss * l1_λ
            terms['loss/area'] = l1_loss

        ''' layout edge constraint loss '''
        if self.edge_factor:
            edge_map = layout_gradient(prediction).squeeze(1)
            target_edge = data['edge'].to(device=edge_map.device)
            edge_loss = F.binary_cross_entropy(edge_map, target_edge)
            loss += edge_loss * self.edge_factor
            terms['loss/edge'] = edge_loss

        terms['loss/loss'] = loss
        return terms

    def metric(self, output, target):
        seg_metric = onegan.metrics.semantic_segmentation.Metric(num_class=5, only_scalar=True)
        score_metric = onegan.metrics.semantic_segmentation.max_bipartite_matching_score
        accuracies = seg_metric(output, target)
        score = score_metric(output, target)
        return {**accuracies, 'score': score}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items


def label_as_rgb_visual(x):
    """ Make segment tensor into colorful image
    Args:
        x (torch.Tensor): shape in (N, H, W) or (N, 1, H, W)
        colors (tuple or list): list of RGB colors, range from 0 to 1.
    Returns:
        canvas (torch.Tensor): colorized tensor in the shape of (N, C, H, W)
    """
    colors = [
        [0.9764706, 0.27058825, 0.3647059], [1., 0.8980392, 0.6666667],
        [0.5647059, 0.80784315, 0.70980394], [0.31764707, 0.31764707, 0.46666667],
        [0.94509804, 0.96862745, 0.8235294]]

    if x.dim() == 4:
        x = x.squeeze(1)
    assert x.dim() == 3

    n, h, w = x.size()
    palette = torch.tensor(colors).to(x.device)
    labels = torch.arange(x.max() + 1).to(x)

    canvas = torch.zeros(n, h, w, 3).to(x.device)
    for color, lbl_id in zip(palette, labels):
        if canvas[x == lbl_id].size(0):
            canvas[x == lbl_id] = color

    return canvas.permute(0, 3, 1, 2)
