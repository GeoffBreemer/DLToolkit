"""Settings for the Kaggle Cats and Dogs data set"""
DATA_PATH = "../data/kaggle_cats_and_dogs/train"       # Path to the original data set
MODEL_PATH = "../savedmodels/alexnet.model"            # Path to the saved model
OUTPUT_PATH = "output"                              # Path to where other output is saved
HISTORY_PATH = OUTPUT_PATH + "/history.json"        # Path to the training history JSON file
RGB_MEANS_PATH = OUTPUT_PATH + "/cats_dogs_alexnet_rgb_means.json"            # Path to the mean RGB values

TRAIN_SET_HDF5_PATH = "../data/kaggle_cats_and_dogs/hdf5/train.hdf5"   # Path to the training HDF5 dataset
VAL_SET_HDF5_PATH = "../data/kaggle_cats_and_dogs/hdf5/val.hdf5"       # Path to the validation HDF5 dataset
TEST_SET_HDF5_PATH = "../data/kaggle_cats_and_dogs/hdf5/test.hdf5"     # Path to the test HDF5 dataset

NUM_CLASSES = 2                                     # Two classes: dog or cat
NUM_VAL_IMAGES = NUM_CLASSES * 1250                 # Number of validation set images
NUM_TEST_IMAGES = NUM_CLASSES * 1250                # Number of test set images

RANDOM_STATE = 122177

IMG_DIM_WIDTH = 256                                 # Image width to resize all images to
IMG_DIM_HEIGHT = 256                                # Image height to resize all images to
IMG_CHANNELS = 3                                    # Expected number of channels

# Training parameters
NUM_EPOCHS = 15          # 70
BATCH_SIZE = 256
ADAM_LR = 1e-3
REG_RATE = 0.0002
