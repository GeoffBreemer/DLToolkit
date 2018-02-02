"""Simple CIFAR-10 classification using Keras, Stochastic Gradient Descent and ShallowNet
To load a saved model use:
    --load=true
"""
from dltoolkit.nn import ShallowNetNN, CIFAR10_CLASSES
from dltoolkit.utils import plot_history, str2bool
from dltoolkit.preprocess import NormalisePreprocessor
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import classification_report
import argparse

# Constants
LEARNING_RATE = 0.01
NUM_EPOCH = 40
BATCH_SIZE = 32
NUM_CLASSES = 10
MODEL_PATH = "../savedmodels/"
OUTPUT_PATH = "../output/"

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
# names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Fit the model or load the saved one
if args["load"]:
    print("Loading previously trained model")
    model = load_model(MODEL_PATH + "cifar10_shallownet.model")
else:
    print("Training the model")

    # Initialise the NN and optimiser
    opt = SGD(lr=LEARNING_RATE)
    model = ShallowNetNN.build_model(num_classes=NUM_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the network
    hist = model.fit(X_train, Y_train,
                     batch_size=BATCH_SIZE,
                     epochs=NUM_EPOCH,
                     validation_data=(X_test, Y_test),
                     verbose=1)
    # note: the test data set should NOT be used for validation_data, but rather a true validation set should be used

    # Save the model
    model.save(MODEL_PATH + "cifar10_shallownet.model")

    # Plot the training results
    plot_history(hist, NUM_EPOCH)

# Make predictions on the test set and print the results to the console
preds = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=-1), preds.argmax(axis=1), target_names=CIFAR10_CLASSES))
