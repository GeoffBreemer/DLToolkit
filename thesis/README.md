# Thesis Code
Work in progress.

## Getting Started
- Download/clone the repository to a local machine
- Add the full path to the repository to the `PYTHONPATH` environment variable (if running code from the terminal)
- Install prequisite packages (see below)

### Prerequisites
Install the Python packages listed below and any packages they may depend on:

- scikit-learn
- scikit-image
- OpenCV 3.3
- NumPy
- Keras 2.1.4
- Tensorflow 1.5
- HDF5
- graphviz
- matplotlib

All code is written in Python 3.6.3 using PyCharm Professional 2017.3.

### Running the Examples
To train a model first execute its training file, e.g. `DECiSION_training.py`. Training parameters, paths etc. are set in the settings file, e.g. `DECiSION_settings.py`. Upon completion use the trained model to predict on an unseen data set using its test file, e.g. `DECiSION_test.py`. Pass the full path to the trained model using the `-m` parameter. For example: `DECiSION_test.py -m="../savedmodels/UNet3D_paper_ep100.model`.

Note that all code is currently setup to test the training and test pipelines by training the models on a **very small** training data set and making predictions on the **same dataset**.

## Segmentation Models
The research project pursues two semantic segmentation models:

1. A U-Net named DECiSION [Ronneberger:2015aa]
2. A 3D U-Net named VOLVuLuS [Cicek:2016aa]

Code to train and test each model comes in three `.py` files:

- `_settings.py`: model specific constants
- `_training.py`: train the model using a training set
- `_test.py`: apply a trained model to a test set

In addition, `thesis_common.py` contains common code shared between the two models and `thesis_metric_loss.py` contains loss metrics used during training.

[Ronneberger:2015aa]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015.

[Cicek:2016aa]: Özgün Çiçek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3d u-net: learning dense volumetric segmentation from sparse annotation. In International Conference on Medical Image Computing and Computer- Assisted Intervention, pages 424–432. Springer, 2016.
