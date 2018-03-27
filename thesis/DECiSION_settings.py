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
IMG_CHANNELS = 1            # number of channels for the images and ground truths (i.e. gray scale)
NUM_CLASSES = 2             # number of classes to segment

MASK_BINARY_THRESHOLD = 20      # Pixel intensities above this value are considered blood vessels
CLASS_WEIGHT_BACKGROUND = 1.
CLASS_WEIGHT_BLOODVESSEL = 10.
MASK_BACKGROUND = 0             # pixel intensity for background pixels (i.e. black)
MASK_BLOODVESSEL = 255          # pixel intensity for vessel pixels (i.e. white)
ONEHOT_BACKGROUND = 0
ONEHOT_BLOODVESSEL = 1

SLICE_START = 69
SLICE_END = 79

# Training hyper parameters
TRN_BATCH_SIZE = 1
TRN_LEARNING_RATE = 0.001
TRN_NUM_EPOCH = 100              #10  #10 #30 #7
TRN_TRAIN_VAL_SPLIT = 0.1       # Percentage of training data to use for the validation set
TRN_DROPOUT_RATE = 0.0          # Dropout rate used for all DropOut layers
TRN_MOMENTUM = 0.99
TRN_PRED_THRESHOLD = 0.5        # Pixel intensities that exceed the threshold are considered a positive detection
TRN_EARLY_PATIENCE = 4          # Early stopping patience

# Miscellaneous
VERBOSE = True
IS_DEVELOPMENT = True
RANDOM_STATE = 122177