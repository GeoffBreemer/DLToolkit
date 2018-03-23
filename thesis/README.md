# Thesis code

Work in progress.

The research project pursues two semantic segmentation models:

1. A U-Net named DECiSION
2. A 3D U-Net named VOLVuLuS

Code to train and test each model comes in three `.py` files:

- `_settings.py`: model specific constants
- `_training.py`: train the model using a training set
- `_test.py`: apply a trained model to a test set

In addition, `thesis_common.py` contains common code shared between the two models and `thesis_metric_loss.py` contains loss metrics used during training.
