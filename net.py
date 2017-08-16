import os
import cv2
import skimage
import skimage.io as sio
import tqdm
import torch
import torchvision
import numpy as np
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

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Stage_Net():

	def __init__(self, name, pretrained, stage_2):
		self.name = name
		self.model = build_resnet101_FCN(pretrained=pretrained, nb_classes=37, stage_2=stage_2).cuda()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True, patience=3, mode='min', min_lr=1e-8)
		self.tf_summary = Logger('./logs', name=name)
		self.criterion = Segmentation_Loss()
		self.accuracy = Pixelwise_Accuracy()

	def train(self, train_loader, validate_loader, epochs):
		
		for epoch in range(1, epochs+1):
			self.model.train()

			self.epoch = epoch
			hist = EpochHistory(length=len(train_loader))
			progress = tqdm.tqdm(train_loader)
			
			for image, target in progress:
				self.optimizer.zero_grad()	
				loss, loss_term, acc = self.forward(image, target)
				loss.backward()
				self.optimizer.step()

				hist.add(loss, loss_term, acc)
				progress.set_description('Epoch#%i' % epoch)
				progress.set_postfix(
					loss='%.04f' % loss.data[0],
					accuracy='%.04f' % acc.data[0])

			metrics = dict(**hist.metric(),**self.evaluate(validate_loader, prefix='val_'))
			print('---> Epoch#{} loss: {loss:.4f}, accuracy={accuracy:.4f}'
				' val_loss: {val_loss:.4f}, val_accuracy={val_accuracy:.4f}'
				.format(self.epoch, **metrics))	
			val_loss = metrics.get('val_loss')
			self.scheduler.step(val_loss)
			self.summary_scalar(metrics)	
			self.save_model()
	@timeit
	def evaluate(self, data_loader, prefix=''):
		self.model.eval()
		hist = EpochHistory(length=len(data_loader))
		for i, (image, target) in enumerate(data_loader):
			loss, loss_term, acc, output = self.forward(image, target, is_eval=True)
			hist.add(loss, loss_term, acc)

			if i == 0:
				self.summary_image(output.data, target, prefix)
		
		return hist.metric(prefix=prefix)

	def predict(self, data_loader, name):
		self.model.eval()
		layout_folder = 'output/layout/%s/' % name
		os.makedirs(layout_folder, exist_ok=True)
		for i, (image, _, ) in enumerate(data_loader):
			output = self.model(Variable(image, volatile=True).cuda())
			_, output = torch.max(output, 1)
			fn = data_loader.dataset.filenames[i]
#			print(i)
			out_ = output[0].cpu().data.numpy() 
			out_ = out_[0]
			sio.imsave(layout_folder + '%s.png' % fn, out_)

	def forward(self, image, target, is_eval=False):

		def to_var(t):
			return Variable(t, volatile=is_eval).cuda()

		image, target = to_var(image), to_var(target)
		output = self.model(image)
		loss, loss_term = self.criterion(output, target)
		acc = self.accuracy(output, target)
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

	def __init__(self, length):
		self.count = 0
		self.len = length
		self.loss_term = {'xent':None, 'l1':None}
		self.losses = np.zeros(self.len)
		self.accuracies = np.zeros(self.len)

	def add(self, loss, loss_term, acc):
		self.losses[self.count] = loss.data[0]
		self.accuracies[self.count] = acc.data[0]
#		print(self.loss_term.items())
		for k, v in loss_term.items():
#			print(self.loss_term[k])
			if self.loss_term[k] is None:
				self.loss_term[k] = np.zeros(self.len)
			self.loss_term[k][self.count] = v.data[0]
		print(loss_term)
		self.count += 1
	
	def metric(self, prefix=''):
		terms = {prefix + 'loss': self.losses.mean(),
				prefix + 'accuracy': self.accuracies.mean()}
		terms.update({
			prefix + k: v.mean() for k,v in self.loss_term.items()
			if v is not None})
		
		return terms


class Pixelwise_Accuracy():

	def __call__(self, output, target):
		return self.pixelwise_accuracy(output, target)

	def pixelwise_accuracy(self, output, target):
		_, output = torch.max(output, 1)
		return (output == target).float().mean()

class Segmentation_Loss():

	def __init__(self, l1_portion=0.1, weights=None):
		self.l1_criterion = nn.L1Loss().cuda()
		self.crossentropy = nn.CrossEntropyLoss(weight=weights).cuda()
		self.l1_portion = l1_portion
	def __call__(self, pred, target) -> (float,dict):
		pixelwise_loss, loss_term = self.pixelwise_loss(pred, target)

		return pixelwise_loss, loss_term

	def pixelwise_loss(self, pred, target):
		log_pred = F.log_softmax(pred)
		xent_loss = self.crossentropy(log_pred, target)

		if not self.l1_portion:
			return xent_loss, {'xent': xent_loss}

		onehot_target = (
			torch.FloatTensor(pred.size())
			.zero_().cuda()
			.scatter_(1, target.data.unsqueeze(1),1))

		l1_loss = self.l1_criterion(pred, Variable(onehot_target))

		return xent_loss + self.l1_portion*l1_loss, {'xent':xent_loss, 'l1':l1_loss}

	def set_summary_loagger(self, tf_summary):
		self.tf_summary = tf_summary


def build_resnet101_FCN(pretrained=False, nb_classes=None, stage_2=False):

	model = ResNet(Bottleneck, [3,4,23,3], num_classes=nb_classes, stage_2=stage_2)
	if pretrained:
		pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
		
		model_dict = model.state_dict()
		pretrained_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)

	return model

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
	expansion = 1 
	def __init__(self, in_planes, planes, stride=1, downsample=None):
		supper(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=37, stage_2=False):
		self.stage_2 = stage_2
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
			   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.dp1 = nn.Dropout(p=0.85)
		self.fc_conv = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False)
		self.bn2 = nn.BatchNorm2d(2048)
		self.dp2 = nn.Dropout(p=0.85)

		self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=False)
		self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)

		self.score_l4 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1,bias=False)
		self.upscore_l4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)

		self.score_l3 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1,  padding=1, bias=False)
		self.upscore_l3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
		self.upout = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
		nn.Conv2d(self.inplanes, planes * block.expansion,
		  kernel_size=1,stride=stride, bias=False),  
		nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		in_ = x
		x = self.conv1(in_)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x3 = self.layer3(x)
		x4 = self.layer4(x3)
		p4 = self.maxpool(x4)

		x = self.dp1(p4)
		x = self.fc_conv(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.dp2(x)
		x = self.classifier(x)

		upscore = self.upscore(x)
		score_l4 = self.score_l4(x4)
		upscore = upscore[:, :, 1:1+score_l4.size()[2], 1:1+score_l4.size()[3]].contiguous()
		upscore_l4 = self.upscore_l4(score_l4+upscore)

		score_l3 = self.score_l3(x3)
		upscore_l3 = self.upscore_l3(upscore_l4+score_l3)
		out = self.upout(upscore_l3)
		print(out)
		out = out[:, :, 27:27+in_.size()[2], 27:27+in_.size()[3]].contiguous()
		print(out)
		return out

class FCN(nn.Module):
	def __init__(self, num_classes=5):
		super(FCN, self).__init__()
		self.features = nn.Sequential(
		# conv1
			nn.Conv2d(3, 64, 3, padding=100),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

			# conv2
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

			# conv3
			nn.Conv2d(128, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8

			# conv4
			nn.Conv2d(256, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16

			# conv5
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
		)
		self.classifier = nn.Sequential(
			# fc6
			nn.Conv2d(512, 4096, 7),
			nn.ReLU(inplace=True),
			nn.Dropout2d(),

			# fc7
			nn.Conv2d(4096, 4096, 1),
			nn.ReLU(inplace=True),
			nn.Dropout2d(),

			# score_fr
			nn.Conv2d(4096, num_classes, 1),
			)
		self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False)
		self._initialize_weights()

#	@timeit
	def _initialize_weights(self):
		vgg16 = torchvision.models.vgg16(pretrained=True)

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				m.weight.data = weight_init.kaiming_normal(m.weight.data)
		for a, b in zip(vgg16.features, self.features):
			if (isinstance(a, nn.Conv2d) and isinstance(b, nn.Conv2d)):
				b.weight.data = a.weight.data
				b.bias.data = a.bias.data
		for i in [0, 3]:
			a, b = vgg16.classifier[i], self.classifier[i]
			b.weight.data = a.weight.data.view(b.weight.size())
			b.bias.data = a.bias.data.view(b.bias.size())

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		x = self.upscore(x)
		x = x[:, :, 44:44 + x.size()[2], 44:44 + x.size()[3]].contiguous()
		return x
