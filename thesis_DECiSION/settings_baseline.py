# Variables used to construct paths and filenames and convert data to HDF5 format
TRAINING_PATH = "../data/MSC8002/training"              # training images
TEST_PATH = "../data/MSC8002/test"                      # test images
MODEL_PATH = "../savedmodels/"                          # saved Keras models
OUTPUT_PATH = "../output/"                              # plots and other output

FLDR_GROUND_TRUTH = "groundtruths"                      # folder with the ground truths
FLDR_IMAGES = "images"                                  # folder with the images
HDF5_EXT = ".h5"
HDF5_KEY = "image"

# Image dimensions
IMG_HEIGHT = 320            # original image height (prior to any pre-processing)
IMG_WIDTH = 320             # original image width (prior to any pre-processing)
IMG_RESIZE_DIM = 227
IMG_RESIZE_DIM_GT = 224
IMG_CHANNELS = 1            # number of channels for the images and ground truths (gray scale)
MASK_BINARY_THRESHOLD = 14

MASK_BACKGROUND = 0
MASK_BLOODVESSEL = 255
ONEHOT_BACKGROUND = 0
ONEHOT_BLOODVESSEL = 1

# Training hyper parameters
TRN_BATCH_SIZE = 4
TRN_LEARNING_RATE = 0.001
TRN_NUM_EPOCH = 100              #10  #10 #30 #7
TRN_TRAIN_VAL_SPLIT = 0.1       # Percentage of training data to use for the validation set
TRN_DROPOUT_RATE = 0.0          # Dropout rate used for all DropOut layers
TRN_MOMENTUM = 0.99
TRN_PRED_THRESHOLD = 0.5        # Pixel intensities that exceed the threshold are considered a positive detection
NUM_CLASSES = 2

# Miscellaneous
VERBOSE = True
IS_DEVELOPMENT = True