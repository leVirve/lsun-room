import enum

import numpy as np


class Layout(enum.Enum):
    frontal = 1
    left = 2
    right = 3
    floor = 4
    ceiling = 5


def hex_to_rgb(x):
    return [255 & (x >> 8 * i) for i in (0, 1, 2)][::-1]


color_palette = np.array([
    hex_to_rgb(c) for c in (0x0, 0xF9455D, 0xFFE5AA, 0x90CEB5, 0x515177, 0xF1F7D2)])


class ColorLayout:
    frontal = color_palette[Layout.frontal.value]
    left = color_palette[Layout.left.value]
    right = color_palette[Layout.right.value]
    floor = color_palette[Layout.floor.value]
    ceiling = color_palette[Layout.ceiling.value]

    def to_layout(self, color):
        return np.argwhere(np.all(color_palette == color, axis=-1))[0][0]

    def color_mask(self, img, color):
        mask = np.all(img == color, axis=-1)
        return mask


def center_point(points, indices):
    indices = [idx - 1 for idx in indices]
    return np.mean(points[indices], axis=0).astype('int')


def type0(e):
    return [
        (Layout.left, center_point(e.points, [1, 2, 3, 4])),
        (Layout.right, center_point(e.points, [5, 6, 7, 8])),
        (Layout.frontal, center_point(e.points, [1, 3, 5, 7])),
        (Layout.floor, center_point(e.points, [3, 4, 5, 6])),
        (Layout.ceiling, center_point(e.points, [1, 2, 7, 8])),
    ]


def type1(e):
    ''' few ground truth labels pt2-pt3 error '''
    return [
        (Layout.left, center_point(e.points, [1, 2, 3])),
        (Layout.right, center_point(e.points, [4, 5, 6])),
        (Layout.frontal, center_point(e.points, [1, 2, 4, 5])),
        (Layout.floor, center_point(e.points, [1, 3, 4, 6])),
    ]


def type2(e):
    return [
        (Layout.left, center_point(e.points, [1, 2, 3])),
        (Layout.right, center_point(e.points, [4, 5, 6])),
        (Layout.frontal, center_point(e.points, [1, 3, 4, 6])),
        (Layout.ceiling, center_point(e.points, [1, 2, 4, 5])),
    ]


def type3(e):
    return [
        (Layout.left, center_point(e.points, [1, 2, 3])),
        (Layout.right, center_point(e.points, [1, 3, 4])),
        (Layout.ceiling, center_point(e.points, [1, 2, 4])),
    ]


def type4(e):
    return [
        (Layout.left, center_point(e.points, [1, 2, 3])),
        (Layout.right, center_point(e.points, [1, 3, 4])),
        (Layout.floor, center_point(e.points, [1, 2, 4])),
    ]


def type5(e):
    return [
        (Layout.left, center_point(e.points, [1, 2, 4, 5])),
        (Layout.right, center_point(e.points, [1, 3, 4, 6])),
        (Layout.floor, center_point(e.points, [4, 5, 6])),
        (Layout.ceiling, center_point(e.points, [1, 2, 3])),
    ]


def type6(e):
    return [
        (Layout.ceiling, center_point(e.points, [1, 2]) - [0, 1]),
        (Layout.frontal, center_point(e.points, [1, 2, 3, 4])),
        (Layout.floor, center_point(e.points, [3, 4]) + [0, 1]),
    ]


def type7(e):
    return [
        (Layout.left, center_point(e.points, [1, 2]) - [1, 0]),
        (Layout.frontal, center_point(e.points, [1, 2, 3, 4])),
        (Layout.right, center_point(e.points, [3, 4]) + [1, 0]),
    ]


def type8(e):
    return [
        (Layout.ceiling, center_point(e.points, [1, 2]) - [0, 1]),
        (Layout.frontal, center_point(e.points, [1, 2]) + [0, 1]),
    ]


def type9(e):
    return [
        (Layout.frontal, center_point(e.points, [1, 2]) - [0, 1]),
        (Layout.floor, center_point(e.points, [1, 2]) + [0, 1]),
    ]


def type10(e):
    return [
        (Layout.left, center_point(e.points, [1, 2]) - [1, 0]),
        (Layout.right, center_point(e.points, [1, 2]) + [1, 0]),
    ]


func_map = {
    0: type0,
    1: type1,
    2: type2,
    3: type3,
    4: type4,
    5: type5,
    6: type6,
    7: type7,
    8: type8,
    9: type9,
    10: type10,
}


def mapping_func(room_type):
    return func_map[room_type]
