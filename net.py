import os
import cv2
import skimage
import skimage.io as sio
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
from logger import Logger
from utils import timeit
import math
import torch.utils.model_zoo as model_zoo
from PIL import Image
from lr_scheduler import *
from models import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class StageNet():

    def __init__(self, name,
                 pretrained=False, stage_2=False,
                 joint_class=False,
                 l1_weight=0.1, type_portion=False, edge_portion=False):
        self.name = name
        self.joint_class = joint_class
        self.model = nn.DataParallel(build_resnet101_FCN(
            nb_classes=37, stage_2=stage_2, joint_class=joint_class)).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, verbose=True, patience=2, mode='min', min_lr=1e-12)
        self.tf_summary = Logger('./logs', name=name)
        self.criterion = Joint_Loss(
            l1_portion=l1_weight, edge_portion=edge_portion, type_portion=type_portion)
        self.accuracy = Joint_Accuracy()
        self.stage_2 = stage_2

    def train(self, train_loader, validate_loader, epochs):
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.epoch = epoch
            hist = EpochHistory(length=len(train_loader),
                                joint_class=self.joint_class)
            progress = tqdm.tqdm(train_loader)

            for data in progress:
                self.optimizer.zero_grad()
                if not self.stage_2:
                    image, target = data[0], data[1]
                    edge_map, room_type = None, None
                    loss, loss_term, acc = self.forward(
                        image, target, edge_map, room_type)
                else:
                    image, target, edge_map, room_type = data[0], data[1], data[2], data[3]
                    loss, loss_term, acc = self.forward(
                        image, target, edge_map, room_type)

                loss.backward()
                self.optimizer.step()

                hist.add(loss, loss_term, acc)

                progress.set_description('Epoch#%i' % epoch)
                if self.joint_class:
                    seg_acc = acc[0]
                    progress.set_postfix(
                        loss='%.04f' % loss.data[0],
                        seg_acc='%.04f' % seg_acc.data[0],
                        type_acc='%.04f' % hist.type_accuracies.mean()
                    )
                else:
                    seg_acc = acc
                    progress.set_postfix(
                        loss='%.04f' % loss.data[0],
                        seg_acc='%.04f' % seg_acc.data[0]
                    )

            metrics = dict(**hist.metric(), **
                           self.evaluate(validate_loader, prefix='val_'))

            if self.joint_class:
                print('---> Epoch#{}:\n train_loss: {loss:.4f}, train_seg_acc={seg_accuracy:.4f}, train_type_acc={type_accuracy:.4f}\n'
                      ' val_loss: {val_loss:.4f}, val_seg_acc={val_seg_accuracy:.4f}, val_type_acc={val_type_accuracy}'
                      .format(self.epoch, **metrics))
            else:
                print('---> Epoch#{}:\n train_loss: {loss:.4f}, accuracy={seg_accuracy:.4f}\n'
                      ' val_loss: {val_loss:.4f}, val_accuracy={val_seg_accuracy:.4f}'
                      .format(self.epoch, **metrics))

            val_loss = metrics.get('val_loss')
            self.scheduler.step(val_loss)
            self.summary_scalar(metrics)
            self.save_model()

    @timeit
    def evaluate(self, data_loader, prefix=''):
        self.model.eval()
        hist = EpochHistory(length=len(data_loader),
                            joint_class=self.joint_class)
        for i, (data) in enumerate(data_loader):
            if not self.stage_2:
                loss, loss_term, acc, output = self.forward(
                    data[0], data[1], None, None, is_eval=True)
                hist.add(loss, loss_term, acc)
            else:
                loss, loss_term, acc, output = self.forward(
                    data[0], data[1], data[2], data[3], is_eval=True)
                hist.add(loss, loss_term, acc)
                if i == 0 & self.joint_class:
                    self.summary_image(output[0].data, data[1], prefix)
                elif i == 0 & self.joint_class == False:
                    self.summary_image(output.data, data[1], prefix)

        return hist.metric(prefix=prefix)

    def predict(self, data_loader, name):
        self.model.eval()
        layout_folder = 'output/layout/%s/' % name
        os.makedirs(layout_folder, exist_ok=True)
        loader = tqdm.tqdm(data_loader)
        for i, (data) in enumerate(loader):
            output, _ = self.model(Variable(data[0], volatile=True).cuda())
            _, output = torch.max(output, 1)
            fn = data_loader.dataset.filenames[i]
            out_ = output[0].cpu().data.numpy()
            out_ = out_[0]
            sio.imsave(layout_folder + '%s.png' % fn, out_)

    def predict_each(self, img):
        self.model.eval()
        if self.stage_2:
            pred, _ = self.model(Variable(img, volatile=True).cuda())
        else:
            pred = self.model(Variable(img, volatile=True).cuda())
        _, output = torch.max(pred, 1)
        res = output.squeeze().cpu().data.numpy()
        return res

    def forward(self, image, target, edge_map, room_type, is_eval=False):

        def to_var(t):
            return Variable(t, volatile=is_eval).cuda()

        if self.stage_2:
            image, target, room_type = to_var(image), to_var(
                target), to_var(room_type.long())
            output = self.model(image)
            loss, loss_term = self.criterion(
                output, target, edge_map, room_type)
            acc = self.accuracy(output, target, room_type, self.joint_class)
        else:
            image, target = to_var(image), to_var(target)
            output = self.model(image)
            loss, loss_term = self.criterion(output, target, None, None)
            acc = self.accuracy(output, target, None, self.joint_class)

        return (loss, loss_term, acc, output) if is_eval else (loss, loss_term, acc)

    def summary_scalar(self, metrics):
        for tag, value in metrics.items():
            self.tf_summary.scalar(tag, value, self.epoch - 1)

    def summary_image(self, output, target, prefix):
        def to_numpy(imgs):
            return imgs.squeeze().cpu().numpy()

        _, output = torch.max(output, 1)
        self.tf_summary.image(prefix + 'output', to_numpy(output), self.epoch)
        self.tf_summary.image(prefix + 'target', to_numpy(target), self.epoch)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self):
        folder = 'output/weight/%s' % self.name
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model.state_dict(), folder + '/%d.pth' % self.epoch)


class EpochHistory():

    def __init__(self, length, joint_class):
        self.count = 0
        self.len = length
        self.joint_class = joint_class
        self.loss_term = {'xent': None, 'l1': None, 'edge': None, 'type': None}
        self.losses = np.zeros(self.len)
        self.seg_accuracies = np.zeros(self.len)
        self.type_accuracies = np.zeros(self.len)

    def add(self, loss, loss_term, acc):
        self.losses[self.count] = loss.data[0]

        if not self.joint_class:
            self.seg_accuracies[self.count] = acc.data[0]
            self.type_accuracies[self.count] = None
        else:
            self.seg_accuracies[self.count] = acc[0].data[0]
            self.type_accuracies[self.count] = acc[1].data[0]

        for k, v in loss_term.items():
            if self.loss_term[k] is None:
                self.loss_term[k] = np.zeros(self.len)
            self.loss_term[k][self.count] = v.data[0]
        self.count += 1

    def metric(self, prefix=''):
        terms = {prefix + 'loss': self.losses.mean(),
                 prefix + 'seg_accuracy': self.seg_accuracies.mean(),
                 prefix + 'type_accuracy': self.type_accuracies.mean()}
        terms.update({
            prefix + k: v.mean() for k, v in self.loss_term.items()
            if v is not None})

        return terms


class Joint_Accuracy():

    def __call__(self, output, seg_target, type_target, joint_class):
        return (self.pixelwise_accuracy(output[0], seg_target), self.type_accuracy(output[1], type_target)) if joint_class else self.pixelwise_accuracy(output, seg_target)

    def pixelwise_accuracy(self, output, target):
        _, output = torch.max(output, 1)
        return (output == target).float().mean()

    def type_accuracy(self, output, target):
        _, output = torch.max(output, 1)
        return (output == target).float().mean()


class Joint_Loss():

    def __init__(self, l1_portion=0.1, edge_portion=0.1, type_portion=1., weights=None):
        self.l1_criterion = nn.L1Loss().cuda()
        self.crossentropy = nn.CrossEntropyLoss(weight=weights).cuda()
        self.edge_criterion = nn.BCELoss().cuda()
        self.type_criterion = nn.NLLLoss().cuda()
        self.l1_portion = l1_portion
        self.edge_portion = edge_portion
        self.type_portion = type_portion

    def __call__(self, pred, target, edge_map, room_type) -> (float, dict):
        loss_term = {}
        if (self.edge_portion is False) & (self.type_portion is False):
            pixelwise_loss, loss_term = self.pixelwise_loss(pred, target)
            return pixelwise_loss, loss_term
        elif (self.edge_portion is False) & (self.type_portion is not False):
            pixelwise_loss, loss_term = self.pixelwise_loss(pred[0], target)
            type_loss, type_loss_term = self.type_loss(pred[1], room_type)
            loss_term.update(type_loss_term)
            return pixelwise_loss + self.type_portion * type_loss, loss_term
        elif (self.type_portion is False) & (self.edge_portion is not False):
            pixelwise_loss, loss_term = self.pixelwise_loss(pred, target)
            edge_loss, edge_loss_term = self.edge_loss(pred, edge_map)
            loss_term.update(edge_loss_term)
            return pixelwise_loss + self.edge_portion * edge_loss, loss_term
        else:
            pixelwise_loss, loss_term = self.pixelwise_loss(pred[0], target)
            edge_loss, edge_loss_term = self.edge_loss(pred[0], edge_map)
            type_loss, type_loss_term = self.type_loss(pred[1], room_type)

            loss_term.update(edge_loss_term)
            loss_term.update(type_loss_term)
            return pixelwise_loss + self.edge_portion * edge_loss + self.type_portion * type_loss, loss_term

    def pixelwise_loss(self, pred, target):
        log_pred = F.log_softmax(pred)
        xent_loss = self.crossentropy(log_pred, target)

        if not self.l1_portion:
            return xent_loss, {'xent': xent_loss}

        onehot_target = (
            torch.FloatTensor(pred.size())
            .zero_().cuda()
            .scatter_(1, target.data.unsqueeze(1), 1))

        l1_loss = self.l1_criterion(pred, Variable(onehot_target))

        return xent_loss + self.l1_portion * l1_loss, {'xent': xent_loss, 'l1': l1_loss}

    def edge_loss(self, pred, edge_map):
        _, pred = torch.max(pred, 1)
        pred = pred.float().squeeze(1)

        imgs = pred.data.cpu().numpy()
        for i, img in enumerate(imgs):
            pred[i].data = torch.from_numpy(cv2.Laplacian(img, cv2.CV_32F))
        mask = pred != 0
        pred[mask] = 1
        edge = Variable(edge_map.float().cuda())
        edge_loss = self.edge_criterion(pred[mask], edge[mask].float())

        return edge_loss, {'edge': edge_loss}

    def type_loss(self, type_dis, room_type):
        log_pred = F.log_softmax(type_dis)
        type_loss = self.type_criterion(log_pred, room_type)

        return type_loss, {'type': type_loss}

    def set_summary_loagger(self, tf_summary):
        self.tf_summary = tf_summary
