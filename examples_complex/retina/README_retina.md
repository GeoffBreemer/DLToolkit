# Segmenting blood vessels in retinal images using Keras

# Keras version
The Keras version is available in:

  `../retina`

Steps:

1 Download the DRIVE data set from [here](https://www.isi.uu.nl/Research/Databases/DRIVE/).

2 Review and update the settings file (`settings_drive.py`), e.g. update the paths.

3 Execute `drive_train.py` to convert the DRIVE data set to HDF5 format and train the model.

4 Apply a trained model to the test set by running `drive_test.py` with the `--m` parameter pointing to the name and location of the trained model to use. For example:

  `drive_test --m="../../savedmodels/UNet_DRIVE_ep10_np2000.model"`

# TensorFlow-only version
The TensorFlow-only version is available in:

  `../retina_tf`

It uses most functions available in `drive_train.py`, `drive_test.py` and `drive_utils.py` to keep things simple, though it does make things a little messy. It also uses the same settings file (`settings_drive.py`). It does assume the HDF5 files have already been generated.

Execute `drive_train_tf.py` to train the model. To test it use:

  `drive_test --m="../../savedmodels/U-net_tf_DRIVE_ep10_np2000.model/tf_retina_model-epoch9_np2000bs1.ckpt"`
