import os
import cv2
import time
from functools import wraps
import numpy as np
import scipy.io as sio

def load_mat(path):
	mat = sio.loadmat(path)
	mat = {k:v for k,v in mat.items() if not k.startwith('____')}
	return list(mat.values())[0]

def load_image(path):
	return cv2.imread(path)

def save_image(path, img):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	return cv2.imwrite(path, img)

def mean_point(points):
	return np.mean(points, axis=0)


def timeit(f):
	@wraps(f)
	def wrap(*args, **kw):
		s = time.time()
		result = f(*args, **kw)
		e = time.time()
		print('--> %s(), cost %2.4f sec' % (f.__name__, e - s))
		return result

	return wrap
