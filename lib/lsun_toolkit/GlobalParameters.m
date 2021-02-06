ROOT_DIR = '../data/';

IMAGE_DIR = [ROOT_DIR 'images/'];
LAYOUT_DIR = [ROOT_DIR 'layout_seg/'];

IMAGE_PATTERN = [IMAGE_DIR '%s' '.jpg'];
LAYOUT_PATTERN = [LAYOUT_DIR '%s' '.mat'];

TRAIN_DATA_PATH = [ROOT_DIR 'training.mat'];
VALIDATION_DATA_PATH = [ROOT_DIR 'validation.mat'];
TEST_DATA_PATH = [ROOT_DIR 'testing.mat'];