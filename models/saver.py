import os
import torch


class Saver():

    def __init__(self, network):
        self.net = network
        self.root = 'output/weight/%s' % self.net.name
        os.makedirs(self.root, exist_ok=True)

    def load_model(self, path):
        self.net.model = torch.load(path)

    def save(self):
        state_dict = self.net.model.state_dict()
        weight_path = os.path.join(self.root, '%d.pth' % self.net.epoch)
        torch.save(state_dict, weight_path)
