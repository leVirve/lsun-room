# General Indoor Layout Estimation from a Single Image

## LSUN Room Layout Estimation

![one_lsun_result_banner](./doc/banner.png)

## Prerequisite

- Python 3.6+
- [OneGAN](https://github.com/leVirve/OneGAN) >= `0.3.0`
- `scikit-image` and `click`, `tqdm`

## Usage

- Dataset

  - Put `LSUN Room Layout Dataset` in folder `../data/lsun_room` relative to this project.
    - `images/`: RGB color image `*.jpg` of indoor room scene
    - `layout_seg/`: layout ground truth `*.mat` of indoor room scene
    - `layout_seg_images/`: generated layout ground truth `*.png` of indoor room scene

  - Put `SUN RGB-D Dataset` in folder `../data/sun_rgbd` relative to this project.
    - `images/`: RGB color image `*.jpg` in `train` and `test` respectly.
    - `labels/`: layout ground truth `*.png` in `train` and `test` respectly.

- Toolkit

  - Put `LSUN Room Layout Dataset` toolkit in folder `../lsun_toolkit`
    - Integrated scripts (TBD)

- Training

  ```bash
  python main.py

  Usage: main.py [OPTIONS]

  Options:
    --name TEXT
    --dataset [lsun_room | others]
    --dataset_root TEXT
    --log_dir TEXT
    --image_size <INTEGER INTEGER>
    --epochs INTEGER
    --batch_size INTEGER
    --workers INTEGER
    --l1_weight FLOAT
    --resume PATH
  ```

- Demo

  ```bash
  python demo.py

  Usage: demo.py [OPTIONS]

  Options:
    --device INTEGER
    --video TEXT
    --weight TEXT
    --input_size <INTEGER INTEGER>.

  ```

- Evaluate with offical Matlab toolkit

  ```bash
  matlab -nojvm -nodisplay -nosplash -r "demo('$EXPERIMENT_OUTPUT_FODLER'); exit;"
  ```

## Tools

- Re-label

  - Output layout image (range from 1-5)

  ```bash
  python script/re_label.py
  ```
