# TEST SFM in system

## trans_data.py

This file can transfer gt.npy to 00.txt

### split_4x3

This function can laod gt.npy and save it in txt mode.

### relative2absolute

This function can load gt.npy(in txt mode, recode the relative pose of each 3 sequent frames) and change it to the absolute pose to the first frame.

The absolute poses is computed on each 2 sample, so there are only even frame(frame 0, 2, 4, 6 ...) are included int the absolute poses.

## gen_gt.py

### test_C_O_Cpre

This is the test demo to transfer the relative pose to the absolute pose.

### compute_pose_error

This is the function to evaluate the SFM-learner from origin sfm-learner.