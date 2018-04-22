# Variables used to construct paths and filenames and convert data to HDF5 format
TRAINING_PATH = "../data/MSC8002/training"              # training images
TEST_PATH = "../data/MSC8002/test"                      # test images
MODEL_PATH = "../savedmodels/"                          # saved Keras models
OUTPUT_PATH = "../output/"                              # plots and other output
SEGMAP_PATH = OUTPUT_PATH + "segmentation_maps/"

FLDR_GROUND_TRUTH = "groundtruths"                      # folder with the ground truths
FLDR_IMAGES = "images"                                  # folder with the images
HDF5_EXT = ".h5"
HDF5_KEY = "image"
IMG_EXTENSION = ".jpg"

# Image dimensions
IMG_HEIGHT = 256            # image height (after cropping)
IMG_WIDTH = 256             # image width (after cropping)
IMG_CHANNELS = 1            # number of channels for the images and ground truths (i.e. gray scale)
NUM_CLASSES = 2             # number of classes to segment
IMG_CROP_HEIGHT = 32        # number of pixels to crop from BOTH the top and the bottom
IMG_CROP_WIDTH = 32         # number of pixels to crop from BOTH the left and the right

MASK_BACKGROUND = 0             # pixel intensity for background pixels (i.e. black)
MASK_BLOODVESSEL = 255          # pixel intensity for vessel pixels (i.e. white)

# Local pipeline testing:
# SLICE_START = 59-2
# SLICE_END = 59+2

# All slices:
# SLICE_START = 0
# SLICE_END = 247

# # Useful slices only:
SLICE_START = 0                 # starting slice index
SLICE_END = 96                  # ending slice index (the slice itself is NOT included)

# Training hyper parameters
MASK_BINARY_THRESHOLD = 20      # pixel intensities above this value are considered blood vessels
CLASS_WEIGHT_BACKGROUND = 1.    # weight for the background class

TRN_LOSS = "ADAM"               # use Adam (ADAM) or another optimiser (SGD typically)
TRN_BATCH_SIZE = 1              # batch size
CLASS_WEIGHT_BLOODVESSEL = 10.  # weight for the blood vessel class

TRN_LEARNING_RATE = 0.001       # Momentum value (gradient descent only)
TRN_NUM_EPOCH = 500             # maximum number of epochs to train
TRN_TRAIN_VAL_SPLIT = 0.1       # percentage of training data to use for the validation set
TRN_DROPOUT_RATE = 0.5          # Dropout rate used for all Dropout layers
TRN_MOMENTUM = 0.99             # Momentum value (gradient descent only)
TRN_PRED_THRESHOLD = 0.5        # Pixel probabilities that exceed the threshold are considered a positive detection
TRN_EARLY_PATIENCE = 9          # Early Stopping patience
TRN_NUM_KFOLD_SPLITS = 3        # number of folds when using cross-validation
TRN_AMS_GRAD = True             # whether to enable AMSGrad (Adam optimiser only)
TRN_PLAT_PATIENCE = 3           # Reduce on plateau patience
TRN_PLAT_FACTOR = 0.2           # Reduce on plateau factor

# Miscellaneous
VERBOSE = True
RANDOM_STATE = 122177
