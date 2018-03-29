"""Simple CIFAR-10 classification using MiniVGGNet or VGG16 with a custom FC layer and pre-trained on
ImageNet and using simple data augmentation for finetuning

Parameters:
    -l=True or -l=False
    -n=MiniVGGNN or -n=VGG16CustomNN
    -d=../data/flowers17

The approach is based on the excellent book "Deep Learning for Computer Vision" by PyImageSearch available on:
https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/
"""
from dltoolkit.nn.cnn import MiniVGGNN, VGG16CustomNN
from dltoolkit.preprocess import NormalisePreprocessor, ResizeWithAspectRatioPreprocessor, ImgToArrayPreprocessor
from dltoolkit.iomisc import MemoryDataLoader
from dltoolkit.utils import plot_training_history, str2bool, model_architecture_to_file, FLOWERS17_CLASS_NAMES,\
    model_performance, visualise_results

from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from imutils import paths
import argparse

# Constants
NUM_EPOCH_WARMUP = 10
NUM_EPOCH_TRAIN = 20
TEST_RATIO = 0.25
BATCH_SIZE = 32
RANDOM_STATE = 122177
SGD_LEARNING_RATE = 0.01            # SGD is used to training
SGD_LEARNING_RATE_VGG16 = 0.001
SGD_MOMENTUM = 0.9
SGD_LR_DECAY = 0.01 / NUM_EPOCH_TRAIN
RMS_LEARNING_RATE = 0.001           # RMSProp is used for VGG16CustomNN "warm-up"

MINIVGG_IMG_WIDTH = 64              # MiniVGG image resolution
MINIVGG_IMG_HEIGHT = 64
VGG16_CUSTOM_DENSE_SIZE = 256
VGG16_IMAGE_DIM = 224               # VGG16 expects 224x224 images
IMG_CHANNELS = 3
NUM_CLASSES = len(FLOWERS17_CLASS_NAMES)

MODEL_PATH = "../savedmodels/"
OUTPUT_PATH = "../output/"
DATASET_NAME = "flowers17"


# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the data set")
ap.add_argument("-l", "--load", type=str2bool, nargs='?',
                const=True, required=False, help="Set to True to load a previously trained model")
ap.add_argument("-n", "--net", type=str, nargs='?', default="",
                const=True, required=True, help="Set to the name of the neural net to use")
args = vars(ap.parse_args())

# Instantiate the selected network
if args["net"] == "MiniVGGNN":
    nnarch = MiniVGGNN(MINIVGG_IMG_WIDTH, MINIVGG_IMG_HEIGHT, IMG_CHANNELS, NUM_CLASSES)
elif args["net"] == "VGG16CustomNN":
    nnarch = VGG16CustomNN(num_classes=NUM_CLASSES, dense_size=VGG16_CUSTOM_DENSE_SIZE)

# String used for logging, naming files
MODEL_NAME = DATASET_NAME + "_" + nnarch.title

# Load and preprocess the data
print("--> Loading data...")
imagePaths = list(paths.list_images(args["dataset"]))
itoa_pre = ImgToArrayPreprocessor()
norm_pre = NormalisePreprocessor()

# Resize the image to the inut shape expected by the selected network
if isinstance(nnarch, VGG16CustomNN):
    res_pre = ResizeWithAspectRatioPreprocessor(VGG16_IMAGE_DIM, VGG16_IMAGE_DIM)
else:
    res_pre = ResizeWithAspectRatioPreprocessor(MINIVGG_IMG_WIDTH, MINIVGG_IMG_HEIGHT)

dl = MemoryDataLoader(preprocessors=[res_pre, itoa_pre, norm_pre])
(data, labels) = dl.load(imagePaths, verbose=250)

# Split the data set and one-hot encode the labels
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=TEST_RATIO, random_state=RANDOM_STATE)
lbl_bin = LabelBinarizer()
Y_train = lbl_bin.fit_transform(Y_train)
Y_test = lbl_bin.transform(Y_test)

# Fit the model or load a previously saved one
if args["load"]:
    print("--> Loading previously trained model...")
    nnarch.model = load_model(MODEL_PATH + MODEL_NAME + ".model")
else:
    # Initialise the NN
    nnarch.build_model()

    # Create the image generator
    img_gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    # Warm up the custom FC layer first when using the VGG16CustomNN network
    if isinstance(nnarch, VGG16CustomNN):
        # Train the network, data augmentation is only applied to the training data
        print("--> Warming up the custom VGG16 FC layers...")
        nnarch.model.compile(loss="categorical_crossentropy",
                             optimizer=RMSprop(lr=RMS_LEARNING_RATE),
                             metrics=["accuracy"])
        hist = nnarch.model.fit_generator(img_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                          epochs=NUM_EPOCH_WARMUP,
                                          validation_data=(X_test, Y_test),
                                          steps_per_epoch=len(X_train)//BATCH_SIZE,
                                          verbose=1)

        # Make predictions on the test set and print the results to the console
        print("Post warm up performance:")
        model_performance(nnarch, X_test, Y_test, FLOWERS17_CLASS_NAMES, BATCH_SIZE)

        # Unfreeze the last few VGG16 conv layers prior to continuing training
        nnarch.unfreeze_vgg_layers(starting_layer=15)

    # Setup the callback to save only the weights resulting in the lowest validation loss
    checkpoint = ModelCheckpoint(MODEL_PATH + MODEL_NAME + ".model",
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=2)

    # Train the network, data augmentation is only applied to the training data
    print("--> Training/finetuning the model...")
    if isinstance(nnarch, VGG16CustomNN):
        opt = SGD(lr=SGD_LEARNING_RATE_VGG16) # use a smaller rate than for MiniVGGNN
    else:
        opt = SGD(lr=SGD_LEARNING_RATE, momentum=SGD_MOMENTUM, decay=SGD_LR_DECAY, nesterov=True)

    nnarch.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    hist = nnarch.model.fit_generator(img_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                      epochs=NUM_EPOCH_TRAIN,
                                      validation_data=(X_test, Y_test),
                                      steps_per_epoch=len(X_train)//BATCH_SIZE,
                                      verbose=1,
                                      callbacks=[checkpoint])
    # note: the test data set should NOT be used for validation_data, but rather a true validation set should be used

    # Save the training and validation results
    plot_training_history(hist, NUM_EPOCH_TRAIN, show=False, save_path=OUTPUT_PATH + MODEL_NAME, time_stamp=True)

# Make predictions on the test set and print the results to the console
print("Post training performance:")
Y_pred = model_performance(nnarch, X_test, Y_test, FLOWERS17_CLASS_NAMES, BATCH_SIZE)

visualise_results(X_test, Y_test, Y_pred, FLOWERS17_CLASS_NAMES)
model_architecture_to_file(nnarch.model, OUTPUT_PATH + MODEL_NAME)
