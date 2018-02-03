"""Animal classification using ShallowNet"""
from dltoolkit.preprocess import ResizePreprocessor, ImgToArrayPreprocessor, NormalisePreprocessor
from dltoolkit.io import MemoryDataLoader
from dltoolkit.nn import ShallowNetNN
from dltoolkit.utils import plot_history, ANIMALS_CLASS_NAMES

from keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import argparse
import numpy as np
import cv2

# Constants
LEARNING_RATE = 0.005
NUM_EPOCH = 20
BATCH_SIZE = 32
RANDOM_STATE = 122177
NUM_CLASSES = 3
OUTPUT_PATH = "../output/"
DATASET_NAME = "animals"

# Check script arguments
ap = argparse.ArgumentParser(description="Apply ShallowNet to animal images.")
ap.add_argument("-d", "--dataset", required=True, help="path to the data set")
args = vars(ap.parse_args())

# Extract the full path to each image in the data set's location
imagePaths = list(paths.list_images(args["dataset"]))

# Load the images, normalising and resizing each to 32x32 pixels upon loading
resize_pre = ResizePreprocessor(32, 32)
itoarr_pre = ImgToArrayPreprocessor()
norm_pre = NormalisePreprocessor()

dl = MemoryDataLoader(preprocessors=[resize_pre, itoarr_pre, norm_pre])
(X, Y) = dl.load(imagePaths, verbose=500)

# Split into a training and test set
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_STATE)

# One-hot encode the labels
Y_train = LabelBinarizer().fit_transform(Y_train)
Y_test= LabelBinarizer().transform(Y_test)

# Initialise the NN and optimiser
opt = SGD(lr=LEARNING_RATE)
nnarch = ShallowNetNN(num_classes=NUM_CLASSES)
nnarch.build_model()
nnarch.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

MODEL_NAME = DATASET_NAME + "_" + nnarch.title

# Train the network
hist = nnarch.model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, validation_data=(X_test, Y_test), verbose=1)

# Make predictions on the test set and print the results to the console
preds = nnarch.model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=-1), preds.argmax(axis=1), target_names=ANIMALS_CLASS_NAMES))

# Plot the training results
plot_history(hist, NUM_EPOCH, show=False, save_path=OUTPUT_PATH + MODEL_NAME, time_stamp=True)

# Visualise a few random images (could be training and/or test images)
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))

# Load and preprocess as before
(X, Y) = dl.load(imagePaths=imagePaths[idxs])

# Make predictions
preds = nnarch.model.predict(X)

# Display results
for (i, imagePath) in enumerate(imagePaths[idxs]):
    image = cv2.imread(imagePath)
    print("Image {} is a {} predicted to be a {} with probability {}%".format(i + 1, Y[i], ANIMALS_CLASS_NAMES[preds[i].argmax(axis=0)], preds[i].max(axis=0)))

    if ANIMALS_CLASS_NAMES[preds[i].argmax(axis=0)] == Y[i]:
        colour = (0, 255, 0)
    else:
        colour = (0, 0, 255)

    cv2.putText(image, "Prediction: {} ({}%)".format(ANIMALS_CLASS_NAMES[preds[i].argmax(axis=0)], preds[i].max(axis=0)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=colour, thickness=2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
