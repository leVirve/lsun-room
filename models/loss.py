import cv2
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


class LayoutLoss():

    def __init__(self, l1_λ=0.1, edge_λ=0.1, weight=None):
        self.l1_λ = l1_λ
        self.edge_λ = edge_λ
        self.cross_entropy_criterion = nn.NLLLoss2d(weight=weight).cuda()
        self.l1_criterion = nn.L1Loss().cuda()
        self.edge_criterion = nn.MSELoss().cuda()

    def __call__(self, output, pred, target, edge_map) -> dict:
        loss_terms = {}
        loss_terms.update(self.pixelwise_loss(output, target))
        loss_terms.update(self.edge_loss(pred, edge_map))

        loss = loss_terms.get('classification')
        loss += self.l1_λ * loss_terms.get('seg_area', 0)
        loss += self.edge_λ * loss_terms.get('edge', 0)
        loss_terms['loss'] = loss

        return loss_terms

    def pixelwise_loss(self, pred, target) -> dict:

        ''' Cross-entropy loss '''
        log_pred = F.log_softmax(pred)
        xent_loss = self.cross_entropy_criterion(log_pred, target)

        if not self.l1_λ:
            return {'classification': xent_loss}

        ''' L1 loss '''
        onehot_target = (
            torch.FloatTensor(pred.size())
            .zero_().cuda()
            .scatter_(1, target.data.unsqueeze(1), 1))
        l1_loss = self.l1_criterion(pred, Variable(onehot_target))

        return {'classification': xent_loss, 'seg_area': l1_loss}

    def edge_loss(self, pred, label) -> dict:
        if not self.edge_λ:
            return {}
        edge = nn.functional.conv2d(
            pred.unsqueeze(1).float(), laplacian_kernel, padding=4, dilation=4)

        edge[torch.abs(edge) < 1e-1] = 0
        edge_loss = self.edge_criterion(edge, label)

        return {'edge': edge_loss}

    def _cv2_edge_loss(self, pred, edge_map) -> dict:
        if not self.edge_λ:
            return {}

        ''' Edge binary cross-entropy '''
        _, pred = torch.max(pred, 1)
        pred = pred.float().squeeze(1)

        imgs = pred.data.cpu().numpy()
        for i, img in enumerate(imgs):
            pred[i].data = torch.from_numpy(cv2.Laplacian(img, cv2.CV_32F))

        mask = pred != 0
        pred[mask] = 1
        edge_map = Variable(edge_map.float().cuda())
        edge_loss = self.edge_criterion(pred[mask], edge_map[mask].float())

        return {'edge': edge_loss}
