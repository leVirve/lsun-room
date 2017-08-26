# LSUN Room Layout Estimation

## Prerequisite

- Python 3.5+

## Usage

- Dataset
  - Put `LSUN Room Layout Dataset` in folder `../data/` relative to this project.
    - `images/`: RGB color image `*.jpg` of indoor room scene
    - `layout_seg/`: layout ground truth `*.mat` of indoor room scene
    - `layout_seg_images/`: generated layout ground truth `*.png` of indoor room scene

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
