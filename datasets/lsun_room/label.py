from datasets.lsun_room.utils import mean_point


class Layout:
    frontal = 1
    left = 2
    right = 3
    floor = 4
    ceiling = 5


def center_point(points, indices):
    indices = [idx - 1 for idx in indices]
    return mean_point(points[indices]).astype('int')


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
