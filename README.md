# LSUN Room Layout Estimation

## Prerequisite

- Python 3.5+
- OpenCV 3
- `scilit-image` and `click`

## Usage

- Dataset

  - Put `LSUN Room Layout Dataset` in folder `../data/lsun_room` relative to this project.
    - `images/`: RGB color image `*.jpg` of indoor room scene
    - `layout_seg/`: layout ground truth `*.mat` of indoor room scene
    - `layout_seg_images/`: generated layout ground truth `*.png` of indoor room scene

  - Put `SUN RGB-D Dataset` in folder `../data/sun_rgbd` relative to this project.
    - `images/`: RGB color image `*.jpg` in `train` and `test` respectly.
    - `labels/`: layout ground truth `*.png` in `train` and `test` respectly.

- Training

  ```bash
  python main.py

  Usage: main.py [OPTIONS]

  Options:
    --name TEXT
    --dataset_root TEXT
    --image_size <INTEGER INTEGER>
    --epochs INTEGER
    --batch_size INTEGER
    --workers INTEGER
    --l1_weight FLOAT
    --resume PATH
  ```

## Tools

- Re-label

  - Output visualized layout image (range from 0-255)

  ```bash
  python script/re_label.py --visualized
  ```

  - Output layout image (range from 1-5)

  ```bash
  python script/re_label.py
  ```
