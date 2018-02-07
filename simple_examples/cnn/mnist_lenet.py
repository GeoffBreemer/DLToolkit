"""MNIST classification using Keras, Stochastic Gradient Descent and LeNet
To load a saved model use:
    --load=true
"""
from dltoolkit.nn.cnn import LeNetNN
from dltoolkit.preprocess import NormalisePreprocessor
from dltoolkit.utils import str2bool, plot_history

from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

import numpy as np
import argparse
import cv2

# Constants
LEARNING_RATE = 0.01
RANDOM_STATE = 122177
TEST_PROP = 0.25
NUM_EPOCH = 5
BATCH_SIZE = 128
NUM_CLASSES = 10

MODEL_PATH = "../savedmodels/"
OUTPUT_PATH = "../output/"
DATASET_NAME = "mnist"

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load", type=str2bool, nargs='?',
                const=True, required=False, help="Set to True to load a previously trained model")
args = vars(ap.parse_args())

# Instantiate the network
nnarch = LeNetNN(NUM_CLASSES)

# String used for naming various things
MODEL_NAME = DATASET_NAME + "_" + nnarch.title

# Load dataset, assume channels_last, normalise
dataset = datasets.fetch_mldata("MNIST Original")
X = NormalisePreprocessor().preprocess(dataset.data)
Y = dataset.target

# Reshape from (# of records, 784) to (# of records, img width, img height, # of channels)
X = X.reshape(X.shape[0], nnarch._img_width, nnarch._img_height, nnarch._img_channels)

# Split the data set and one-hot encode the labels
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y.astype("int"), test_size=TEST_PROP, random_state=RANDOM_STATE)
Y_train = to_categorical(Y_train, nnarch._num_classes)
Y_test = to_categorical(Y_test, nnarch._num_classes)

# Fit the model or load the saved one
if args["load"]:
    print("Loading the previously trained {} model".format(nnarch.title))
    nnarch.model = load_model(MODEL_PATH + "mnist_lenet.model")
else:
    print("Training the {} model".format(nnarch.title))

    # Setup and train the model
    sgd = SGD(lr=LEARNING_RATE)
    nnarch.build_model()
    nnarch.model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    hist = nnarch.model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                     batch_size=BATCH_SIZE,
                     epochs=NUM_EPOCH,
                     verbose=1)
    # note: the test data set should NOT be used for validation_data, but rather a true validation set should be used

    # Save the trained model (architecture, weights, loss/optimizer, state)
    nnarch.model.save(MODEL_PATH + MODEL_NAME + ".model")

    # Plot results
    plot_history(hist, NUM_EPOCH, show=False, save_path=OUTPUT_PATH + MODEL_NAME, time_stamp=True)

# Predict on the test set and print the results
Y_pred = nnarch.model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=1),
                            Y_pred.argmax(axis=1),
                            target_names=[str(x) for x in range(NUM_CLASSES)]))

# Visualise a few random test images, increase the size for better visualisation
idxs = np.random.randint(0, len(X_test), size=(10,))
for (i, image) in enumerate(X_test[idxs]):
    print("Image {} is a {} predicted to be a {}".format(i+1, Y_test[idxs[i]].argmax( axis=0), Y_pred[idxs[i]].argmax(axis=0)))
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
