"""Simple CIFAR-10 classification using Keras, Stochastic Gradient Descent and MiniVGGNet
To load a saved model use:
    --load=true
"""
from dltoolkit.nn import MiniVGGNN, CIFAR10_CLASSES
from dltoolkit.preprocess import NormalisePreprocessor
from dltoolkit.utils import plot_history, str2bool, save_model_architecture
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import argparse, datetime
import numpy as np
import cv2

# Constants
LEARNING_RATE = 0.01
NUM_EPOCH = 40
BATCH_SIZE = 64
MOMENTUM = 0.9
LR_DECAY = 0.01 / NUM_EPOCH

NUM_CLASSES = 10
IMG_WIDTH = 32
IMG_HEIGHT = 32

MODEL_PATH = "../savedmodels/"
OUTPUT_PATH = "../output/"
MODEL_NAME = "cifar10_minivggnet"

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load", type=str2bool, nargs='?',
                const=True, required=False, help="Set to True to load a previously trained model")
args = vars(ap.parse_args())

# Load data
((X_train, Y_train), (X_test, Y_test)) = cifar10.load_data()

# Preprocess: scale to [0..1]
X_train = NormalisePreprocessor().preprocess(X_train)
X_test = NormalisePreprocessor().preprocess(X_test)

# Binarize labels
Y_train = to_categorical(Y_train, NUM_CLASSES)
Y_test = to_categorical(Y_test, NUM_CLASSES)

# Fit the model or load the saved one
if args["load"]:
    print("Loading previously trained model")
    model = load_model(MODEL_PATH + MODEL_NAME + ".model")
else:
    print("Training the model")

    # Initialise the NN and optimiser
    opt = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=LR_DECAY, nesterov=True)
    model = MiniVGGNN.build_model(IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Setup the callback to save only the weights resulting in the lowest validation loss
    checkpoint = ModelCheckpoint(MODEL_PATH + MODEL_NAME + ".model", monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    callbacks = [checkpoint]

    # Train the network
    hist = model.fit(X_train, Y_train,
                     batch_size=BATCH_SIZE,
                     epochs=NUM_EPOCH,
                     validation_data=(X_test, Y_test),
                     verbose=2,
                     callbacks=callbacks)
    # note: the test data set should NOT be used for validation_data, but rather a true validation set should be used

    # Plot the training results
    current_dt = datetime.datetime.now()
    plot_history(hist, NUM_EPOCH, show=False, save_path=OUTPUT_PATH + MODEL_NAME + "training_{}_{}.png".format(
        current_dt.strftime("%Y%m%d"),
        current_dt.strftime("%H%M%S")
    ))

# Make predictions on the test set and print the results to the console
Y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=-1), Y_pred.argmax(axis=1), target_names=CIFAR10_CLASSES))

# Visualise a few random test images, increase the size for better visualisation
idxs = np.random.randint(0, len(X_test), size=(10,))
for (i, image) in enumerate(X_test[idxs]):
    print("Image {} is a {} predicted to be a {}".format(i+1,
                                                         CIFAR10_CLASSES[Y_test[idxs[i]].argmax(axis=0)],
                                                         CIFAR10_CLASSES[Y_pred[idxs[i]].argmax(axis=0)]))
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Save the model architecture to disk
save_model_architecture(model, OUTPUT_PATH + MODEL_NAME)
