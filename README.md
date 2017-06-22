# LSUN Room Layout Dataset Tool

## Prerequisite

- Python 3.5+

## Submodules

- `fcn_models/` from [weering/room_segmodel](https://bitbucket.org/weering/room_segmodel) @Bitbucket, mostly contributed by Michael.

## Installation

- Clone this project
  ```bash
  git clone --recursive https://github.com/leVirve/lsun-room
  ```

- Updata `fcn_models`
  ```bash
  git submodule update --recursive
  ```

## Tools

- Training
  - Put your `LSUN Room Layout` dataset in folder `../data/` relative to this project.

  ```bash
  python train.py
  ```

- Re-label

  - Output visualized layout image (range from 0-255)

  ```bash
  python re_label.py --visualized
  ```

  - Output layout image (range from 1-5)

  ```bash
  python re_label.py
  ```
