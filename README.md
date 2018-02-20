# Project: DLToolkit
Collection of deep learning code being developed while working on my thesis. 

## Getting Started
- Download/clone the repository to a local machine
- Add the full path to the repository to the `PYTHONPATH` environment variable (if running code from the terminal)
- Install prequisite packages (see below)

### Prerequisites
Install the Python packages listed below and any packages they may depend on:

- scikit-learn
- scikit-image
- OpenCV3
- NumPy
- Keras 2.1.2
- Tensorflow
- HDF5
- graphviz
- matplotlib
- [imutils](https://github.com/jrosebr1/imutils)

All code is written in Python 3.6.3 using PyCharm Professional 2017.3.

### Running the Examples
To run a simple example use (instructions can be found in each `.py` file):

`python mnist_lenet.py -l=True`

Another example is training a VGG16 network with a custom fully connected layer to classify the `flowers17` data set:

`python flowers17.py -n=VGG16CustomNN -d=../data/flowers17 -l=False`

To use the trained model use:

`python flowers17.py -n=VGG16CustomNN -d=../data/flowers17 -l=True`

Alternatively, run each example from a IDE. Except for the `animals` and `flowers17` data sets, all data sets will first be downloaded if they are not yet available locally.

## Data sets
Data sets used:

- `animals`: this is a subset of Kaggle's Dogs vs Cats [competition](https://www.kaggle.com/c/dogs-vs-cats) data set, containing only images of cats, dogs and pandas.
- `flowers17`: available for download [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/), the only change made to the original data set is that each image was moved to a subfolder named after the class.
- `DRIVE`: this data set contains retinal images and it is used for the segmentation example. It can be downloaded from [here](https://www.isi.uu.nl/Research/Databases/DRIVE/).
- the usual suspects like MNIST, CIFAR-10 etc. will be downloaded using sklearn.datasets or keras.datasets when they are used for the first time.

## Folder Structure
The folders below contain toolkit specific code, settings for more elaborate examples and code for all simple examples:

- `dltoolkit`:
  - `io`: classes for loading data sets, converting to HDF5 format etc.
  - `nn`: various neural network architectures built using Keras, to date only a number of convolution neural networks (CNN) have been implemented.
    - `cnn`: convolutional neural network (CNN) architectures for classification tasks
    - `rnn` : recurrent neural network (RNN) architectures
    - `segment`:  neural network architectures for segmentation tasks
  - `preprocess`: various image preprocessing utilities (resize, crop etc.).
  - `utils`: various generic utilities.
- `settings`: settings for more elaborate examples, which are kept separate from the training source code.
- `simple_examples`: a number of simple examples (e.g. MNIST using LeNet).
- `complex_examples`: more involved examples, currently only:
  - `kaggle_cats_and_dogs.py` for Kaggle's Dogs vs Cats [competition](https://www.kaggle.com/c/dogs-vs-cats)
  - `/retina`: semantic segmentation using a U-Net, partially based on [this](https://github.com/orobix/retina-unet) repository. Doesn't produce winning results, but does quite well considering the minimal data augmentation and hardly any hyper parameter tuning was used. Comes in Keras and TensorFlow versions.

The folders below are not included on GitHub due to their (potential) size and/or because they contain output data created by the various examples:

- `data`: contains data sets.
- `output`: JSON files, plots etc. created by the examples.
- `savedmodels`: saved Keras models.

## Acknowledgments
Some of the code is based on the excellent book ["Deep Learning for Computer Vision"](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/) by PyImageSearch.
