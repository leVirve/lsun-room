import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator as KerasIterator
from keras.preprocessing.image import ImageDataGenerator as KerasDataGenerator

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
        self.files = self.dataset.items
        self.samples = len(self.files)
        print('Found %d images.' % (self.samples))

        super().__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        image_shape = self.target_size + (3,)
        label_shape = self.target_size + (1,)
        batch_img = np.zeros((current_batch_size,) + image_shape, dtype=K.floatx())
        batch_lbl = np.zeros((current_batch_size,) + label_shape, dtype=K.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            e = self.files[j]
            img, lbl = e.image, e.layout
            img = normalize(resize(img, image_shape))
            lbl = resize(np.clip(lbl, 1, 5) - 1, label_shape)

            batch_img[i], batch_lbl[i] = img, lbl

        return batch_img, batch_lbl


def resize(img, shape):
    return np.resize(img, shape)


def normalize(img):
    return img * 2 - 1
