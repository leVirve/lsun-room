from Resnet_blocks import *
import math

def build_resnet101_FCN(pretrained=False, nb_classes=None, stage_2=False, joint_class=False):

	model = ResNet(Bottleneck, [3,4,23,3], num_classes=nb_classes, stage_2=stage_2, joint_class=joint_class)
	if pretrained:
		pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
		
		model_dict = model.state_dict()
		pretrained_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)

	return model

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=37, num_rtype=11, stage_2=False, joint_class=False):
		self.stage_2 = stage_2
		self.joint_class = joint_class
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
		self.avgpool = nn.AvgPool2d(7)
		self.type_fc = nn.Linear(512*block.expansion, num_rtype)

		self.dp1 = nn.Dropout(p=0.85)
		self.fc_conv = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False)
		self.bn2 = nn.BatchNorm2d(2048)
		self.dp2 = nn.Dropout(p=0.85)

		self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=False)
		self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)

		self.score_l4 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1,bias=False)
		self.upscore_l4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)

		self.score_l3 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1,  padding=1, bias=False)
		self.upscore_l3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)
		self.fc = nn.Conv2d(37, 5, kernel_size=1, stride=1, bias=False)

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
#		p2 = self.avgpool(x4)
		if self.joint_class == True:
			avg_p = self.avgpool(x4)
			type_x = avg_p.view(avg_p.size(0), -1)
			type_x = self.type_fc(type_x)

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
		out = upscore_l3[:, :, 27:27+in_.size()[2], 27:27+in_.size()[3]].contiguous()
		if self.stage_2 == True:
				out_stage2 = self.fc(out)
				if self.joint_class == True:
    					return out_stage2, type_x 
				else: 
						return out_stage2
		else:
    			return out


class FCN(nn.Module):
	def __init__(self, pretrained=True, num_classes=37, stage_2=False):
		super(FCN, self).__init__()
		self.pretrained = pretrained
		self.stage_2 = stage_2
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
		self.fc = nn.Conv2d(num_classes, 5, kernel_size=1, stride=1, bias=False)
		#self._initialize_weights()

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
		if self.stage_2 == True:
			x = self.fc(x)
		return x
