# Thesis Code
Work in progress.

## TO DO

Generic:

- [ ] implement baseline model
- [x] generators
- [x] image pre-processing
- [ ] data augmentation offline generation script
- [x] different opimizers ([algorithms](https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f)):
    - [x] Adam: good results
    - [x] AdaDelta: appears to take longer to converge
    - [x] SGD with Nesterov and Momentum 0.9: no good
    - [ ] Adam with `amsgrad=True`
- [ ] speed up convergence:
    - [x] kernel initialisers: bias constants, weights randomized
    - [x] batch normalization
- [ ] regularization:
    - [ ] less complex models
    - [ ] L1/L2 regularization
    - [x] early stopping
    - [x] small batch sizes: use 1 due to memry constraints, especially 3D UNet
    - [ ] ensembling (7.11):
        - [ ] bagging
        - [ ] model averaging
    - [x] dropout (~ bagging), possibly increase model size to offset capacity reduction, may be less effective due to small data set
- [ ] train final model with selected hyper parameters using all training data (incl. validation data) (7.7):
        - [ ] continue training with parameters obtained at early stopping point but now with all training data, OR:
        - [ ] train for the same number of epochs but now with all training data

- [ ] hyper parameter selection

2D U-Net:
- [ ] hyper parameter selection
- [ ] use validation set or CV

3D U-Net:
- [ ] reduce memory usage
- [ ] hyper parameter selection
- [ ] use validation set or CV


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

1. A variation of **U-Net** ([Ronneberger 2015](#references)) named DECiSION
2. A variation of **3D U-Net** ([Çiçek 2016](#references)) named VOLVuLuS

Code to train and test each model comes in three notebook files:

- `_settings`: model specific constants
- `_training`: train the model using a training set
- `_test`: apply a trained model to a test set

In addition, `thesis_common.py` contains common code shared between the two models and `thesis_metric_loss.py` contains loss metrics used during training.

## References
**[Cicek]**: Özgün Çiçek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3d u-net: learning dense volumetric segmentation from sparse annotation. In *International Conference on Medical Image Computing and Computer- Assisted Intervention*, pages 424–432. Springer, 2016.

**[Ronneberger]**: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, pages 234–241. Springer, 2015.
