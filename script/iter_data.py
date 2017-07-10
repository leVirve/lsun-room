import hack_path  # noqa
import time

from lsun_room import Phase, DataItems
from lsun_room.edge import mapping_func


def main():

    for phase in [Phase.TRAIN, Phase.VALIDATE]:
        print('==> iter for data in %s phase' % phase)
        s = time.time()
        dataset = DataItems(root_dir='../data', phase=phase)

        for i, e in enumerate(dataset.items):
            if e.type == 10:
                func = mapping_func(e.type)
                out = func(e, image_size=(404, 404), width=2)

                import cv2
                cv2.imshow('edge', out)
                cv2.waitKey()

        print('==> Done in %.4f sec.' % (time.time() - s))


if __name__ == '__main__':
    main()
