import os

TRAIN = 'training'
VALIDATE = 'validation'
TEST = 'testing'

root_dir = '../data/'

image_dir = os.path.join(root_dir, 'images/')
layout_dir = os.path.join(root_dir, 'layout_seg/')
layout_image_dir = os.path.join(root_dir, 'layout_seg_images/')

image_pattern = image_dir + '%s.jpg'
layout_pattern = layout_dir + '%s.mat'
layout_image_pattern = layout_image_dir + '%s.png'

data_path = {
    TRAIN: os.path.join(root_dir, 'training.mat'),
    VALIDATE: os.path.join(root_dir, 'validation.mat'),
    TEST: os.path.join(root_dir, 'testing.mat'),
}
