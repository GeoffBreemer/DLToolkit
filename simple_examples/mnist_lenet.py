"""MNIST classification using Keras, Stochastic Gradient Descent and LeNet"""
from dltoolkit.nn import LeNetNN
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# Constants
LEARNING_RATE = 0.01
RANDOM_STATE = 122177
TEST_PROP = 0.25
NUM_EPOCH = 20
BATCH_SIZE = 128
IMG_DIM = 28
IMG_CHN = 1
NUM_CLASSES = 10
MODEL_PATH = "../savedmodels/"

# Parse arguments
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load", type=str2bool, nargs='?',
                const=True, required=False, help="Set to True to load a previously trained model")
args = vars(ap.parse_args())

# Load dataset, assume channels_last
dataset = datasets.fetch_mldata("MNIST Original")
X = dataset.data / 255.0
Y = dataset.target
X = X.reshape(X.shape[0], IMG_DIM, IMG_DIM, IMG_CHN)

# Split the data set and one-hot encode the labels
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y.astype("int"), test_size=TEST_PROP, random_state=RANDOM_STATE)
lb = LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
Y_test = lb.transform(Y_test)

# Fit the model or load the saved one
if args["load"] == True:       # use: --load=true
    print("Loading previously trained model")
    model = load_model(MODEL_PATH + "mnist_lenet.model")
else:
    print("Training the model")

    # Setup and train the model
    sgd = SGD(lr=LEARNING_RATE)
    model = LeNetNN.build_model(img_width=IMG_DIM, img_height=IMG_DIM, img_channels=IMG_CHN, num_classes=NUM_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1)
    # note: the test data set should NOT be used for validation_data, but rather a true validation set should be used

    # Save the trained model
    model.save(MODEL_PATH + "mnist_lenet.model")

    # Plot results
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCH), hist.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCH), hist.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCH), hist.history["acc"], label="train_acc")
    plt.plot(np.arange(0, NUM_EPOCH), hist.history["val_acc"], label="val_acc")
    plt.title("Loss/accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/accuracy")
    plt.legend()
    plt.show()

# Predict on the test set and print the results
Y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=1),
                            Y_pred.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# Visualise a few random test images
idxs = np.random.randint(0, len(X_test), size=(10,))
for (i, image) in enumerate(X_test[idxs]):
    print("Image {} is a {} predicted to be a {}".format(i+1, Y_test[idxs[i]].argmax( axis=0), Y_pred[idxs[i]].argmax(axis=0)))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
