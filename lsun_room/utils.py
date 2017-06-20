import os

import cv2
import numpy as np
import scipy.io as sio


def load_mat(path):
    return sio.loadmat(path)


def load_image(path):
    return cv2.imread(path)


def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.imwrite(path, img)


def mean_point(points):
    return np.mean(points, axis=0)
