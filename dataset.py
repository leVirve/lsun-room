import os

import numpy as np
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import Iterator as KerasIterator
from keras.preprocessing.image import ImageDataGenerator as KerasDataGenerator
from keras.preprocessing.image import load_img, img_to_array, flip_axis

from lsun_room import Phase, Dataset


class DataGenerator(KerasDataGenerator):

    def flow_from_directory(self, directory, **kwargs):
        return DatasetIterator(directory, self, **kwargs)


class DatasetIterator(KerasIterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), phase=Phase.TRAIN,
                 batch_size=32, shuffle=True, seed=None):
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)

        self.dataset = Dataset(root_dir=directory, phase=phase)
        self.filenames = [e.name for e in self.dataset.items]
        self.samples = len(self.filenames)
        print('==> Found %d images.' % (self.samples))

        super().__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        image_shape = self.target_size + (3,)
        label_shape = self.target_size + (1,)
        batch_img = np.zeros((current_batch_size,) + image_shape, dtype=K.floatx())
        batch_lbl = np.zeros((current_batch_size,) + label_shape, dtype=K.floatx())

        for i, j in enumerate(index_array):
            name = self.filenames[j]
            image_path = os.path.join(self.dataset.image, '%s.jpg' % name)
            label_path = os.path.join(self.dataset.layout_image, '%s.png' % name)

            # load image
            img = load_img(image_path)
            lbl = load_img(label_path, grayscale=True)

            # random horizental flip
            if np.random.random() < 0.5:
                img = flip_axis(img, 2)
                lbl = flip_axis(lbl, 2)

            # resize image
            img = resize(img, self.target_size)
            lbl = resize(lbl, self.target_size)

            # img to array
            img = img_to_array(img)
            lbl = img_to_array(lbl)

            img = rgb_to_bgr(img)
            img = remove_mean(img)
            lbl = np.clip(lbl, 1, 5) - 1

            batch_img[i], batch_lbl[i] = img, lbl

        return batch_img, batch_lbl


class Transformer():

    def __init__(self, target_size):
        self.target_size = target_size

    def transform(self)

def resize(img, shape):
    return img.resize(shape, Image.NEAREST)


def rgb_to_bgr(img):
    return img[:, :, ::-1]


def remove_mean(img):
    mean_val = [116.190, 97.203, 92.318]
    return img.astype('float') - mean_val
