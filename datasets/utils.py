from PIL import Image


def load_image(path):
    return Image.open(path)


def read_line(filepath):
    return [e.strip() for e in open(filepath)]
