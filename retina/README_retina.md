# Segmenting blood vessels in retina image

Steps:

1 Download the DRIVE data set from [here](https://www.isi.uu.nl/Research/Databases/DRIVE/).

2 Review and update the settings file (`settings_drive.py`), e.g. update the paths.

3 Execute `drive_train.py` to convert the DRIVE data set to HDF5 format and train the model.

4 Apply a trained model to the test set by running `drive_test.py` with the `--m` parameter pointing to the name and location of the trained model to use. For example:

  `drive_test --m="../savedmodels/UNet_DRIVE_ep10_np2000.model"`
