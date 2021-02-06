# Indoor Scene Layout Estimation from a Single Image

![one_lsun_result_banner](./doc/banner.png)

[Hung Jin Lin](https://tw.linkedin.com/in/hung-jin-lin-5b66119b), [Sheng-Wei Huang](https://www.linkedin.com/in/sheng-wei-huang), [Shang-Hong Lai](https://tw.linkedin.com/in/shang-hong-lai-4913a0b), [Chen-Kuo Chiang](https://www.cs.ccu.edu.tw/~ckchiang/).

International Conference on Pattern Recognition (ICPR), 2018.

## Information

Confernece paper in [PDF](./doc/icpr2018_lin_layoutestimation.pdf).
Cannot provide original checkpoint in paper due to the agreement, so here's a re-trained checkpoint at [Google Drive](https://drive.google.com/file/d/1aUJoXM9SQMe0LC38pA8v8r43pPOAaQ-a/view?usp=sharing) for fast evaluation.

Updated Jan 2021.

## Prerequisite

- Python 3.6+
- PyTorch 1.0+
- [OneGAN](https://github.com/leVirve/OneGAN) == `0.3.2`, clone and checkout to that tag.

  ```bash
  git clone https://github.com/leVirve/OneGA
  git checkout 0.3.2
  ```

- `pip install -e requirements.txt`

## Dataset

- Download dataset from http://lsun.cs.princeton.edu/2015.html#layout.
  - Unfortunately, the website hosted LSUN Layout challenge is down.
- Find them from the [web archive](https://web.archive.org/web/20190118150204/http://lsun.cs.princeton.edu/2016/)
  - [layout.zip](https://web.archive.org/web/20170221111502/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/layout.zip)
  - [training.mat](https://web.archive.org/web/20180923231343/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/training.mat), [validation.mat](https://web.archive.org/web/20180923231343/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/validation.mat) and [testing.mat](https://web.archive.org/web/20180923231343/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/testing.mat).
- However, the `image.zip` is too large and still unavailable to download from the web archive.
  - Fortunately, you can find them in https://github.com/liamw96/pytorch.room.layout#data. [@liamw96](https://github.com/liamw96) provides dataset image in the `lsun.zip` on Google drive!

## Usage

Thanks [@shuuchen](https://github.com/shuuchen) for an all-in-one project, you may also refer to https://github.com/shuuchen/lsun-room-dsc!

- Dataset

  - Put `LSUN Room Layout Dataset` into folder `./data/lsun_room`.
    - `images/`: RGB color image `*.jpg` of indoor room scene
    - `layout_seg/`: layout ground truth `*.mat` of indoor room scene
  - Prepare layout images for trianing/validation.

    - `layout_seg_images/`: generated layout ground truth `*.png` of indoor room scene

    ```bash
    python -m script.re_label
    ```

- Training

  - The trained model will be saved to folder `./ckpts`

  Example

  ```bash
  python main.py --phase train --arch resnet --edge_factor 0.2 --l2_factor 0.2 --name baseline
  ```

- Testing

  ```bash
  python demo.py --weight {checkpoint_path} --video {test_video}
  ```

- Toolkit

  - Official `LSUN Room Layout Dataset` toolkit in folder [`./lib/lsun_toolkit`](lib/lsun_toolkit)

    ```bash
    # this is my modified scrpt for usage demonstration,
    # you may need to modify the official Matlab code to evaluate your results.
    matlab -nojvm -nodisplay -nosplash -r "demo('$EXPERIMENT_OUTPUT_FODLER'); exit;"
    ```

## Citation

```bibtex
@inproceedings{lin2018layoutestimation,
    Author = {Hung Jin Lin and Sheng-Wei Huang and Shang-Hong Lai and Chen-Kuo Chiang},
    Title = {Indoor Scene Layout Estimation from a Single Image},
    Booktitle = {2018 24th International Conference on Pattern Recognition (ICPR)},
    Year = {2018}
}
```

## More Results

![one_lsun_result_more](./doc/more.jpg)
