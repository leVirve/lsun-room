import hack_path  # noqa
import sys
import time
from multiprocessing import Pool, cpu_count

from lsun_room import Phase, DataItems


def worker(item):
    item.remap_layout()
    item.save_layout(visualization=True if len(sys.argv) > 1 else False)


def main():

    for phase in [Phase.TRAIN, Phase.VALIDATE]:
        print('==> re-label for data in %s phase' % phase)
        s = time.time()
        dataset = DataItems(root_dir='../data', phase=phase)
        with Pool(cpu_count()) as pool:
            pool.map(worker, dataset.items)
        print('==> Done in %.4f sec.' % (time.time() - s))


if __name__ == '__main__':
    main()
