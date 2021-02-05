# Indoor Layout Estimation from a Single Image

![one_lsun_result_banner](./doc/banner.png)

## Citation

```bibtex
@inproceedings{lin2018layoutestimation,
    Author = {Hung Jin Lin and Sheng-Wei Huang and Shang-Hong Lai and Chen-Kuo Chiang},
    Title = {Indoor Scene Layout Estimation from a Single Image},
    Booktitle = {2018 24th International Conference on Pattern Recognition (ICPR)},
    Year = {2018}
}
```

**The code is under evaluation and update TBD. Deprecated information below.**
Thanks [@shuuchen](https://github.com/shuuchen) for an all-in-one project, you may also refer to https://github.com/shuuchen/lsun-room-dsc!

## Prerequisite

- Python 3.6+
- [OneGAN](https://github.com/leVirve/OneGAN) == `0.3.2`, clone and checkout to that tag.

  ```bash
  git clone https://github.com/leVirve/OneGA
  git checkout 0.3.2
  ```

## Dataset

- Download dataset from http://lsun.cs.princeton.edu/2015.html#layout and put them in the following folders.
- Unfortunately, the website hosted LSUN Layout challenge is down, you can only find them from the [web archive](https://web.archive.org/web/20190118150204/http://lsun.cs.princeton.edu/2016/), e.g. [layout.zip](https://web.archive.org/web/20170221111502/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/layout.zip), [training.mat](https://web.archive.org/web/20180923231343/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/training.mat), [validation.mat](https://web.archive.org/web/20180923231343/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/validation.mat), [testing.mat](https://web.archive.org/web/20180923231343/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/testing.mat).
- However, the `image.zip` is too large and still unavailable to download from the web archive, fortunately, you can find them in https://github.com/liamw96/pytorch.room.layout#data. [@liamw96](https://github.com/liamw96) provides dataset image in the `lsun.zip` on Google drive!

## Usage

- Dataset

  - Edit the root of path at `config.yml` Line#8 `folder`.
  - Put `LSUN Room Layout Dataset` into folder `${folder}` relative to this project.
    - `images/`: RGB color image `*.jpg` of indoor room scene
    - `layout_seg/`: layout ground truth `*.mat` of indoor room scene
    - `layout_seg_images/`: generated layout ground truth `*.png` of indoor room scene
  - Prepare layout images for trianing/validation.

    ```bash
    python script/re_label.py
    ```

- Training

  - The trained model will be saved to folder `./exp/checkpoints/`
  - You can modify `config.yml` to play with hyperparameters for training.

  Example

  ```bash
  python main.py --phase train --arch resnet --edge_factor 0.2 --l2_factor 0.2 --name baseline
  ```

  Detials

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

- Toolkit

  - Put `LSUN Room Layout Dataset` toolkit in folder `../lsun_toolkit`
    - Integrated scripts (TBD)

- Evaluate with official Matlab toolkit

  ```bash
  matlab -nojvm -nodisplay -nosplash -r "demo('$EXPERIMENT_OUTPUT_FODLER'); exit;"
  ```
