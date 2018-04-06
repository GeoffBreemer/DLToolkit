# Variables used to construct paths and filenames and convert data to HDF5 format
TRAINING_PATH = "../data/MSC8002/training3d"            # training images
TEST_PATH = "../data/MSC8002/test3d"                    # test images
MODEL_PATH = "../savedmodels/"                          # saved Keras models
OUTPUT_PATH = "../output/"                              # plots and other output

FLDR_GROUND_TRUTH = "groundtruths"                      # folder with the ground truths
FLDR_IMAGES = "images"                                  # folder with the images
HDF5_EXT = ".h5"
HDF5_KEY = "image"
IMG_EXTENSION = ".jpg"

# Image dimensions
IMG_HEIGHT = 240            # original image height (prior to any pre-processing)
IMG_WIDTH = 240             # original image width (prior to any pre-processing)
IMG_CHANNELS = 1            # number of channels for the images and ground truths (i.e. gray scale)
NUM_CLASSES = 2             # number of classes to segment

# Local testing:
# SLICE_START = 59 - 8
# SLICE_END = 59 + 8

# All slices:
# SLICE_START = 0
# SLICE_END = 247

# Useful slices only:
# SLICE_START = 0
# SLICE_END = 96

# AWS 2layer
SLICE_START = 59 - 16
SLICE_END = 59 + 16

IMG_CROP_HEIGHT = 40        # number of pixels to crop from BOTH the top and the bottom
IMG_CROP_WIDTH = 40         # number of pixels to crop from BOTH the left and the right

MASK_BINARY_THRESHOLD = 20  # Pixel intensities above this value are considered blood vessels
CLASS_WEIGHT_BACKGROUND = 1.
CLASS_WEIGHT_BLOODVESSEL = 10.
MASK_BACKGROUND = 0         # pixel intensity for background pixels (i.e. black)
MASK_BLOODVESSEL = 255      # pixel intensity for vessel pixels (i.e. white)

# Training hyper parameters
TRN_BATCH_SIZE = 1
TRN_LEARNING_RATE = 0.001
TRN_NUM_EPOCH = 200
TRN_TRAIN_VAL_SPLIT = 0#1/3       # Percentage of training data to use for the validation set
TRN_DROPOUT_RATE = 0.5          # Dropout rate used for all DropOut layers
TRN_MOMENTUM = 0.99
TRN_PRED_THRESHOLD = 0.5        # Pixel intensities that exceed the threshold are considered a positive detection
TRN_EARLY_PATIENCE = 3 #10         # Early stopping patience

# Miscellaneous
VERBOSE = True
RANDOM_STATE = 122177
