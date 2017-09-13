import os
import torchvision

from datasets.utils import load_image, read_line


class ImageFolderDataset(torchvision.datasets.ImageFolder):

    num_classes = 19
    image_folder = {
        'train': 'TrainVal_images/TrainVal_images/train_images',
        'val': 'TrainVal_images/TrainVal_images/val_images'
    }
    parsing_folder = {
        'train': 'TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations',
        'val': 'TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations',
    }
    image_ids = {
        'train': 'TrainVal_images/train_id.txt',
        'val': 'TrainVal_images/val_id.txt'
    }

    def __init__(self, root, phase,
                 input_transform=None, target_transform=None, **kwargs):
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.phase = phase

        self.ids = read_line(os.path.join(root, self.image_ids[phase]))
        self.filenames = [e + '.png' for e in self.ids]
        self.image_folder = os.path.join(root, self.image_folder[phase])
        self.parsing_folder = os.path.join(root, self.parsing_folder[phase])

    def __getitem__(self, index):
        return self.load(self.ids[index])

    def load(self, fname):
        image = load_image(os.path.join(self.image_folder, fname + '.jpg')).convert('RGB')
        label = load_image(os.path.join(self.parsing_folder, fname + '.png'))
        image = self.input_transform(image)
        label = self.target_transform(label)

        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.filenames)
