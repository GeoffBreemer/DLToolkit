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
PATCH_DIM = 48              # patch dimension (width == height)
PATCH_CHANNELS = 1          # number of channels (gray scale)
PATCHES_NUM_RND = 50000       # number of random patches to generate for each training image

# Training parameters
NUM_EPOCH = 1 #7
BATCH_SIZE = 32
TRAIN_VAL_SPLIT = 0.1
DROPOUT_RATE = 0.0          # 0.2

# Other variables
VERBOSE = True              # set to True for debugging print statements to the console
DEVELOPMENT = True          # set to True to avoid converting data to HDF5 every run
