# dltoolkit
Collection of Deep Learning code based on the book ["Deep Learning for Computer Vision"](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/) by PyImageSearch.

# Dependencies
Required Python modules:

- scikit-learn
- NumPy
- Keras
- Tensorflow
- imutils (https://github.com/jrosebr1/imutils)
- Python 3.6.3

# Datasets
Some of the simple examples use data that came with the book. Most of the code though uses the usual suspects like MNIST, CIFAR-10 etc.

# Folders
The folders below contain all toolkit specific code:

- dltoolkit: toolkit source code containing the modules shown below:
  - io: classed to load data sets, convert to HDF5 format etc.
  - nn: various neural network architectures built using Keras
  - preprocess: various image preprocessing utilities (resize, crop etc.)
  - utils: various generic utilities
- settings: settings for more elaborate examples kept separate from the training source code
- simple_examples: a number of simple examples (e.g. MNIST using LeNet)

The root folder contains source code for more elaborate examples:

- kaggle_cats_and_dogs.py: Kaggle's Dogs vs Cats competition (https://www.kaggle.com/c/dogs-vs-cats)

The folders below are not included on GitHub due to their size:

- data: contains data sets that came with the book
- output: JSON files, plots etc. created by the examples
- savedmodels: saved Keras models
