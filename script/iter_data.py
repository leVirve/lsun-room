import hack_path  # noqa
import time

from lsun_room import Phase, Dataset


def main():

    for phase in [Phase.TRAIN, Phase.VALIDATE]:
        print('==> iter for data in %s phase' % phase)
        s = time.time()
        dataset = Dataset(root_dir='../data', phase=phase)

        for i, e in enumerate(dataset.items):
            pass

        print('==> Done in %.4f sec.' % (time.time() - s))


if __name__ == '__main__':
    main()
