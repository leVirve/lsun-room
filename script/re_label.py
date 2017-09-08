import hack_path  # noqa
import time
from multiprocessing import Pool, cpu_count

import click

from datasets.lsun_room.item import DataItems


def worker(item):
    item.remap_layout()
    item.save_layout()


@click.command()
@click.option('--dataset_root', default='../data/lsun_room/')
def main(dataset_root):

    for phase in ['train', 'val']:
        print('==> re-label for data in %s phase' % phase)
        s = time.time()
        dataset = DataItems(root_dir=dataset_root, phase=phase)
        with Pool(cpu_count()) as pool:
            pool.map(worker, dataset.items)
        print('==> Done in %.4f sec.' % (time.time() - s))


if __name__ == '__main__':
    main()
