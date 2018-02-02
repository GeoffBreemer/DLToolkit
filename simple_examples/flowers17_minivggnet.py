"""Simple CIFAR-10 classification using Keras, Stochastic Gradient Descent and MiniVGGNet
To load a saved model use:
    --load=true
"""
from dltoolkit.nn import MiniVGGNN, FLOWERS17_CLASSES
from dltoolkit.preprocess import NormalisePreprocessor, ResizePreprocessor, ImgToArrayPreprocessor
from dltoolkit.io import MemoryDataLoader
from dltoolkit.utils import plot_history, str2bool, save_model_architecture
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import argparse, datetime
import numpy as np
import cv2

# Constants
LEARNING_RATE = 0.01
NUM_EPOCH = 50
BATCH_SIZE = 64
MOMENTUM = 0.9
LR_DECAY = 0.01 / NUM_EPOCH
RANDOM_STATE = 122177

IMG_WIDTH = 64
IMG_HEIGHT = 64

MODEL_PATH = "../savedmodels/"
OUTPUT_PATH = "../output/"
MODEL_NAME = "flowers17_minivggnet"

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load", type=str2bool, nargs='?',
                const=True, required=False, help="Set to True to load a previously trained model")
ap.add_argument("-d", "--dataset", required=True, help="path to the data set")
args = vars(ap.parse_args())

# Load data
imagePaths = list(paths.list_images(args["dataset"]))
res_pre = ResizePreprocessor(IMG_WIDTH, IMG_HEIGHT)
itoa_pre = ImgToArrayPreprocessor()
norm_pre = NormalisePreprocessor()
dl = MemoryDataLoader(preprocessors=[res_pre, itoa_pre, norm_pre])
(data, labels) = dl.load(imagePaths, verbose=250)
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=RANDOM_STATE)

# One-hot encode the labels
Y_train = LabelBinarizer().fit_transform(Y_train)
Y_test= LabelBinarizer().fit_transform(Y_test)

# Fit the model or load the saved one
if args["load"]:
    print("Loading previously trained model")
    model = load_model(MODEL_PATH + MODEL_NAME + ".model")
else:
    print("Training the model")

    # Initialise the NN and optimiser
    opt = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=LR_DECAY, nesterov=True)
    model = MiniVGGNN.build_model(IMG_WIDTH, IMG_HEIGHT, len(FLOWERS17_CLASSES))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Setup the callback to save only the weights resulting in the lowest validation loss
    checkpoint = ModelCheckpoint(MODEL_PATH + MODEL_NAME + ".model",
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=2)

    # Train the network
    hist = model.fit(X_train, Y_train,
                     batch_size=BATCH_SIZE,
                     epochs=NUM_EPOCH,
                     validation_data=(X_test, Y_test),
                     verbose=2,
                     callbacks=[checkpoint])
    # note: the test data set should NOT be used for validation_data, but rather a true validation set should be used

    # Plot the training and validation results
    current_dt = datetime.datetime.now()
    plot_history(hist, NUM_EPOCH, show=False, save_path=OUTPUT_PATH + MODEL_NAME + "training_{}_{}.png".format(
        current_dt.strftime("%Y%m%d"),
        current_dt.strftime("%H%M%S")
    ))

# Make predictions on the test set and print the results to the console
Y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=-1), Y_pred.argmax(axis=1), target_names=FLOWERS17_CLASSES))

# Visualise a few random test images, increase the size for better visualisation
idxs = np.random.randint(0, len(X_test), size=(10,))
for (i, image) in enumerate(X_test[idxs]):
    print("Image {} is a {} predicted to be a {}".format(i+1,
                                                         FLOWERS17_CLASSES[Y_test[idxs[i]].argmax(axis=0)],
                                                         FLOWERS17_CLASSES[Y_pred[idxs[i]].argmax(axis=0)]))
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Save the model architecture to disk
save_model_architecture(model, OUTPUT_PATH + MODEL_NAME)
