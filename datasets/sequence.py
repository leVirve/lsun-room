import pathlib

import cv2
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.datasets.folder import is_image_file


class VideoStream:

    def __init__(self, image_size, filepath=None, device=None):
        self.target_size = (image_size, image_size)
        self.origin_size = None
        self.stream = cv2.VideoCapture(filepath if filepath else device)

    def __iter__(self):
        while True:
            ret, frame = self.stream.read()
            if not self.origin_size:
                self.origin_size = (frame.shape[1], frame.shape[0])
            if not ret:
                break
            image = F.to_tensor(frame[..., ::-1].copy())
            image = F.resize(image, self.target_size, interpolation=Image.BILINEAR)

            yield F.normalize(image, mean=0.5, std=0.5)

    def __del__(self):
        self.stream.release()


class ImageFolder:

    def __init__(self, image_size, filepath):
        self.target_size = (image_size, image_size)
        self.origin_size = None
        self.filepaths = self.search_images(filepath)

    def search_images(self, path):
        path = pathlib.Path(path)
        if path.is_dir():
            return [e for e in path.glob('*') if is_image_file(str(e))]
        else:
            return [path]

    def __iter__(self):
        for image_path in self.filepaths:
            image = Image.open(image_path).convert('RGB')
            shape = image.size
            image = F.to_tensor(image)
            image = F.resize(image, self.target_size, interpolation=Image.BILINEAR)
            image = F.normalize(image, mean=0.5, std=0.5)
            yield image, shape, image_path
