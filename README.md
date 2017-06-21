# LSUN Room Layout Dataset Tool

## Prerequisite

- Python 3.5+

## Submodules

- `fcn_model/` from [weering/room_segmodel](https://bitbucket.org/weering/room_segmodel) @Bitbucket, mostly contributed by Michael.


## Tool

- Re-label

  - Output visualized layout image (range from 0-255)

  ```bash
  python re_label.py --visualized
  ```

  - Output layout image (range from 1-5)

  ```bash
  python re_label.py
  ```
