# Thesis Code
Work in progress.

## TO DO

Generic:

- [ ] use Frangi filter groud truths and 3D slicer ground truths (DECiSION only)
- [ ] use signal strength threshold 14 instead of 20 (DECiSION only)
- [ ] multi-GU https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
- [x] determine which DECiSION model to use: 2, 3 or 4lyr -> cross-val? train/val split on three patients?
- [x] remove epoch from model name
- [x] use different thresholds during training https://stackoverflow.com/questions/42606207/keras-custom-decision-threshold-for-precision-and-recall
- [X] FCN: pre-train on 2 images, then on larger set?
- [x] code cleanup unet, 3dunet, fcn: replace model creation with shorter for loops
- [ ] kicking (increasing) the learning rate: https://blog.deepsense.ai/deep-learning-right-whale-recognition-kaggle/
- [ ] check for data augmentation: https://www.kaggle.com/lextoumbourou/humpback-whale-id-data-and-aug-exploration
- [x] Use evaluate for final metrics
- [x] check float16 usage, possibly change back to float32
- [x] implement baseline model
- [x] generators
- [x] image pre-processing
- [x] data augmentation offline generation script
- [x] different opimizers ([algorithms](https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f)):
    - [x] Adam: good results
    - [x] AdaDelta: appears to take longer to converge
    - [x] SGD with Nesterov and Momentum 0.9: no good
    - [x] Adam with `amsgrad=True`
- [x] speed up convergence:
    - [x] kernel initialisers: bias constants, weights randomized
    - [x] batch normalization
    - [x] try ReduceLROnPlateau
- [x] regularization:
    - [x] less complex models
    - [x] L1/L2 regularization
    - [x] early stopping
    - [x] small batch sizes: use 1 due to memry constraints, especially 3D UNet
    - [ ] ensembling (7.11):
        - [ ] bagging
        - [ ] model averaging
    - [x] dropout (~ bagging), possibly increase model size to offset capacity reduction, may be less effective due to small data set
- [ ] train final model with selected hyper parameters using all training data (incl. validation data) (7.7):
        - [ ] continue training with parameters obtained at early stopping point but now with all training data, OR:
        - [ ] train for the same number of epochs but now with all training data

- [X] hyper parameter selection

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
The research project pursues three semantic segmentation models:

1. **FCN-32s** by ([Long 2014](#references)) used as a baseline
2. A variation of **U-Net** ([Ronneberger 2015](#references)) named DECiSION
3. A variation of **3D U-Net** ([Çiçek 2016](#references)) named VOLVuLuS

Code to train and test each model comes in three notebook files:

- `_settings`: model specific constants
- `_training`: train the model using a training set
- `_test`: apply a trained model to a test set

In addition, `thesis_common.py` contains common code shared between the two models and `thesis_metric_loss.py` contains loss metrics used during training. `thesis_augment_data.ipynb` is used to augment the data set and `thesis_style.use` makes all plot have the same look and feel.

## References
**[Long]**: Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully Convolutional Networks for Semantic Segmentation. In *Corr*, abs/1411.4038, 2014.

**[Cicek]**: Özgün Çiçek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3d u-net: learning dense volumetric segmentation from sparse annotation. In *International Conference on Medical Image Computing and Computer- Assisted Intervention*, pages 424–432. Springer, 2016.

**[Ronneberger]**: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, pages 234–241. Springer, 2015.
