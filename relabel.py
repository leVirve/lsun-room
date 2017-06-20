import sys
from multiprocessing import Pool

import lsun_room
from lsun_room.dataset import Dataset


def worker(item):
    item.layout_remap()
    item.save_layout(visualization=True if len(sys.argv) > 1 else False)


def main():
    dataset = Dataset(state=lsun_room.TRAIN)
    with Pool(8) as pool:
        pool.map(worker, dataset.items)


if __name__ == '__main__':
    main()
