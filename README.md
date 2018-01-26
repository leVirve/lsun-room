# Room Layout Estimation with Semantic Transfer

*WARNING: this is a quick experimental implementation. Preserve the historical code base only.*
Contributed by my partner, [matlabdestroyer/lsun-room](https://github.com/matlabdestroyer/lsun-room).

## Stage1

    Train Model on SUNRGBD dataset for semantic segmentation
    Path: Add the dataset under this path -- ../data/stage1_data/
    Dataset Content: image and label folder, train-val split is recorded in thte two txt files located in  /stage1_utils

    Training command line: python stage1.py

## Stage2

    Train Model on LSUN RGBD dataset for semantic transfer
    Path: Add the dataset under the path of -- ../data/
    Dataset Content: images folder, layout_seg_images folder along with testing.mat, training.mat, validation.mat
        - Images: http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/image.zip
        - Training.mat: http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/training.mat
        - Validation.mat: http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/validation.mat
        - Testing.mat: http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/testing.mat
        - Layout_seg_images: Room Layout semantic representation is contributed by LeVrive

    Training command line: python stage2.py
