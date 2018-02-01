"""Animal classification using ShallowNet"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dltoolkit.preprocess import ResizePreprocessor, ImageToArrayPreprocessor
from dltoolkit.io import MemoryDataLoader
from dltoolkit.nn import ShallowNetNN
from keras.optimizers import SGD
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Constants
NUM_EPOCH = 100

# Check script arguments
ap = argparse.ArgumentParser(description="Apply ShallowNet to animal images.")
ap.add_argument("-d", "--dataset", required=True, help="path to the data set")
args = vars(ap.parse_args())

# Extract the full path to each image in the data set's location
imagePaths = list(paths.list_images(args["dataset"]))

# Load the images, resizing each to 32x32 pixels upon loading
resize_pre = ResizePreprocessor(32, 32)
itoarr_pre = ImageToArrayPreprocessor()

dl = MemoryDataLoader(preprocessors=[resize_pre, itoarr_pre])
(X, Y) = dl.load(imagePaths, verbose=500)
print(X.shape)
# Normalise the images
X = X.astype("float") / 255.0

# Split into a training and test set
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, random_state=42)

# One-hot encode the labels
Y_train = LabelBinarizer().fit_transform(Y_train)
Y_test= LabelBinarizer().fit_transform(Y_test)

# Initialise the NN and optimiser
opt = SGD(lr=0.005)
model = ShallowNetNN.build_model(32, 32, 3, 3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
hist = model.fit(X_train, Y_train, batch_size=32, epochs=NUM_EPOCH, validation_data=(X_test, Y_test), verbose=1)

# Make predictions on the test set and print the results to the console
preds = model.predict(X_test, batch_size=32)
print(classification_report(Y_test.argmax(axis=-1), preds.argmax(axis=1), target_names=["cat", "dog", "panda"]))

# Plot the training results
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
