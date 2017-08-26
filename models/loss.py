import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LayoutLoss():

    def __init__(self, l1_λ=0.1, edge_λ=0.1, weight=None):
        self.l1_λ = l1_λ
        self.edge_λ = edge_λ
        self.cross_entropy_criterion = nn.NLLLoss2d(weight=weight).cuda()
        self.l1_criterion = nn.L1Loss().cuda()
        self.edge_criterion = nn.BCELoss().cuda()

    def __call__(self, pred, target, edge_map) -> (float, dict):
        pixelwise_loss, loss_term = self.pixelwise_loss(pred, target)

        if not self.edge_λ:
            return pixelwise_loss, loss_term

        edge_loss, edge_loss_term = self.edge_loss(pred, edge_map)

        loss_term.update(edge_loss_term)
        return pixelwise_loss + self.edge_λ * edge_loss, loss_term

    def pixelwise_loss(self, pred, target) -> (float, dict):
        log_pred = F.log_softmax(pred)
        xent_loss = self.cross_entropy_criterion(log_pred, target)

        if not self.l1_λ:
            return xent_loss, {'xent': xent_loss}

        onehot_target = (
            torch.FloatTensor(pred.size())
            .zero_().cuda()
            .scatter_(1, target.data.unsqueeze(1), 1))
        l1_loss = self.l1_criterion(pred, Variable(onehot_target))

        return xent_loss + self.l1_λ * l1_loss, {'xent': xent_loss, 'l1': l1_loss}

    def edge_loss(self, pred, edge_map) -> (float, dict):
        _, pred = torch.max(pred, 1)
        pred = pred.float().squeeze(1)

        imgs = pred.data.cpu().numpy()
        for i, img in enumerate(imgs):
            pred[i].data = torch.from_numpy(cv2.Laplacian(img, cv2.CV_32F))

        mask = pred != 0
        pred[mask] = 1
        edge_map = Variable(edge_map.float().cuda())
        edge_loss = self.edge_criterion(pred[mask], edge_map[mask].float())

        return edge_loss, {'edge': edge_loss}

    def set_summary_logger(self, tf_summary):
        self.tf_summary = tf_summary
