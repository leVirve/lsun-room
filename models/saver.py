import os
import torch


class Checkpoint():

    def register_trainer(self, trainer):
        self.trainer = trainer
        self.root = 'output/weight/%s' % self.trainer.name
        os.makedirs(self.root, exist_ok=True)

    def load_model(self, path):
        self.trainer.model = torch.load(path)

    def save(self):
        state_dict = self.trainer.model.state_dict()
        weight_path = os.path.join(self.root, '%d.pth' % self.trainer.epoch)
        torch.save(state_dict, weight_path)
        # torch.save({
        #     'model': self.trainer.model.state_dict(),
        #     'optimizer': self.trainer.optimizer.state_dict(),
        #     'epoch', 'arch'
        # }, weight_path)
