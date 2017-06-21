import sys
from multiprocessing import Pool, cpu_count

from lsun_room import Phase, Dataset


def worker(item):
    item.remap_layout()
    item.save_layout(visualization=True if len(sys.argv) > 1 else False)


def main():
    dataset = Dataset(root_dir='../data', state=Phase.TRAIN)
    with Pool(cpu_count()) as pool:
        pool.map(worker, dataset.items)


if __name__ == '__main__':
    main()
