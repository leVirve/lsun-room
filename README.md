# LSUN Room Layout Dataset Tool

## Prerequisite

- Python 3.5+

## Tools

- Training
  - Put your `LSUN Room Layout` dataset in folder `../data/` relative to this project.

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

- Re-label

  - Output visualized layout image (range from 0-255)

  ```bash
  python script/re_label.py --visualized
  ```

  - Output layout image (range from 1-5)

  ```bash
  python script/re_label.py
  ```
