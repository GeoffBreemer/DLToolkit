# Segmenting blood vessels in retina image

Steps:

1 Download the data set from [here](https://www.isi.uu.nl/Research/Databases/DRIVE/)
2 Update the settings file `settings_drive.py`, e.g. by updating the paths
3 Execute `drive_train.py` to convert the DRIVE data set to HDF5 format and train the model
4 Apply the trained model to the test set by running `drive_test.py` using the paramater `--m` to point to the trained model. For example:

`drive_test --m="../savedmodels/UNet_DRIVE_ep10_np2000.model"`