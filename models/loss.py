import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


laplacian_kernel = Variable(torch.from_numpy(
    np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]])
    ).unsqueeze(0).unsqueeze(0).float()).cuda()


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)


class SegmentLoss():

    def __init__(self, l1_λ=0.1, edge_λ=0.1, weight=None):
        self.l1_λ = l1_λ
        self.edge_λ = edge_λ
        self.cross_entropy_criterion = CrossEntropyLoss2d(weight=weight).cuda()

    def __call__(self, score, pred, gt_layout, _, end_hook=None):
        loss_terms = self.pixelwise_loss(score, gt_layout)

        loss = loss_terms.get('classification')
        loss += self.l1_λ * loss_terms.get('seg_area', 0)
        loss_terms['loss'] = loss

        if end_hook:
            end_hook()

        return loss_terms

    def pixelwise_loss(self, pred, target) -> dict:

        ''' Cross-entropy loss '''
        xent_loss = self.cross_entropy_criterion(pred, target)

        if not self.l1_λ:
            return {'classification': xent_loss}

        ''' L1 loss '''
        onehot_target = (
            torch.FloatTensor(pred.size())
            .zero_().cuda()
            .scatter_(1, target.data.unsqueeze(1), 1))
        l1_loss = self.l1_criterion(pred, Variable(onehot_target))

        return {'classification': xent_loss, 'seg_area': l1_loss}

    def register_trainer(self, trainer):
        self.trainer = trainer


class LayoutLoss(SegmentLoss):

    def __init__(self, l1_λ=0.1, edge_λ=0.1, weight=None):
        self.l1_λ = l1_λ
        self.edge_λ = edge_λ
        self.cross_entropy_criterion = CrossEntropyLoss2d(weight=weight).cuda()
        self.l1_criterion = nn.L1Loss().cuda()
        self.edge_criterion = nn.MSELoss().cuda()

    def __call__(self, score, pred, gt_layout, item, end_hook=None):
        loss_terms = {}
        loss_terms.update(self.pixelwise_loss(score, gt_layout))
        loss_terms.update(self.edge_loss(pred, item, end_hook))

        loss = loss_terms.get('classification')
        loss += self.l1_λ * loss_terms.get('seg_area', 0)
        loss += self.edge_λ * loss_terms.get('edge', 0)
        loss_terms['loss'] = loss

        edge_map = None
        if '_edge_map' in loss_terms:
            edge_map = loss_terms.pop('_edge_map')
        if end_hook:
            end_hook(edge_map)

        return loss_terms

    def edge_loss(self, pred, item, end_hook) -> dict:
        if not self.edge_λ:
            return {}

        label = Variable(item['edge'], volatile=not self.trainer.model.training).cuda()
        edge = nn.functional.conv2d(
            pred.unsqueeze(1).float(), laplacian_kernel, padding=4, dilation=4)

        edge = edge.squeeze()
        edge[torch.abs(edge) < 1e-1] = 0
        edge_loss = self.edge_criterion(edge[edge != 0], label[edge != 0])

        if end_hook:
            end_hook(edge)

        return {'edge': edge_loss, '_edge_map': edge}
