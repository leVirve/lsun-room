import sys
from multiprocessing import Pool, cpu_count

from lsun_room import Phase
from lsun_room.dataset import Dataset


def main():

    def worker(item):
        item.layout_remap()
        item.save_layout(visualization=True if len(sys.argv) > 1 else False)

    dataset = Dataset(state=Phase.TRAIN)
    with Pool(cpu_count()) as pool:
        pool.map(worker, dataset.items)


if __name__ == '__main__':
    main()
