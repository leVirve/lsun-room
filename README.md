# Room Layout Estimation with Semantic Transfer 

Stage1
    Train Model on SUNRGBD dataset for semantic segmentation
    Path: Add the dataset under this path -- ../lsun-room/stage1_data/
    Dataset Content: image and label folder, train-val split is recorded in thte two txt files located in  /stage1_utils

    Training command line: python stage1.py

Stage2
    Train Model on LSUN RGBD dataset for semantic transfer \n
    Path: Add the dataset under the path of -- ../lsun-room/stage2_data/ \n
    Dataset Content: images folder, layout_seg_images folder along with testing.mat, training.mat, validation.mat \n
        - Images: http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/image.zip \n
        - Training.mat: http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/training.mat \n
        - Validation.mat: http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/validation.mat \n 
        - Testing.mat: http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/testing.mat \n
        - Layout_seg_images: Room Layout semantic representation is contributed by LeVrive \n 

    Training command line: python stage2.py
