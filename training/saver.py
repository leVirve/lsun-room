import os
import torch


class Checkpoint():

    def register_trainer(self, trainer):
        self.trainer = trainer
        self.root = 'output/weight/%s' % self.trainer.logger.name
        os.makedirs(self.root, exist_ok=True)

    def load(self, path):
        ckpt = torch.load(path)
        assert ckpt['arch'] == self.trainer.model.__class__.__name__
        self.trainer.start_epoch = ckpt['epoch']
        self.trainer.model.load_state_dict(ckpt['model'])
        self.trainer.optimizer.load_state_dict(ckpt['optimizer'])

    def save(self):
        path = os.path.join(self.root, '%d.pth' % self.trainer.epoch)
        torch.save({
            'model': self.trainer.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
            'epoch': self.trainer.epoch + 1,
            'arch': self.trainer.model.__class__.__name__
        }, path)
