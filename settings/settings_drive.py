# Variables used to construct paths and filenames
TRAINING_PATH = "../data/DRIVE/training"            # training images
TEST_PATH = "../data/DRIVE/test"                    # test images
MODEL_PATH = "../savedmodels/"                      # model weights
OUTPUT_PATH = "../output/"                          # plots and other outputs

FOLDER_MANUAL_1 = "1st_manual"                      # ground truths
FOLDER_IMAGES = "images"                            # retina images
FOLDER_MASK = "mask"                                # masks
HDF5_EXT = ".hdf5"
HDF5_KEY = "image"

# Image dimensions
IMG_HEIGHT = 584            # original image height (prior to any pre-processing)
IMG_WIDTH = 565             # original image width (prior to any pre-processing)
IMG_CHANNELS_TIF = 3        # number of channels for the retina images (RGB)
IMG_CHANNELS_GIF = 1        # number of channels for the ground truths (gray scale)

# Patch dimensions
PATCH_DIM = 48              # patch dimension (squares, i.e. width == height == PATCH_DIM)
PATCH_CHANNELS = 1          # number of colour channels used for patches (gray scale)
PATCHES_NUM_RND = 1000    # total # of random patches to generate (i.e. for all images in the training set combined)

# Training parameters
NUM_EPOCH = 150 #10  #10 #30 #7
BATCH_SIZE = 1              # TODO increase after testing!!!
TRAIN_VAL_SPLIT = 0.1       # Percentage of training data to use for the validation set
DROPOUT_RATE = 0.0          # Dropout rate used for all DropOut layers
NUM_OUTPUT_CLASSES = 2      # number of classes the U-Net should identify
MOMENTUM = 0.99
PRED_THRESHOLD = 0.5        # Pixel intensities that exceed the threshold are considered a positive detection

# Other variables
VERBOSE = True              # set to True for debugging print statements to the console
DEVELOPMENT = True          # set to True to avoid converting data to HDF5 every run
