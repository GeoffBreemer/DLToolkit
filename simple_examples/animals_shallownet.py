"""Animal classification using ShallowNet"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dltoolkit.preprocess import ResizePreprocessor, ImgToArrayPreprocessor, NormalisePreprocessor
from dltoolkit.io import MemoryDataLoader
from dltoolkit.nn import ShallowNetNN
from keras.optimizers import SGD
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Constants
LEARNING_RATE = 0.005
NUM_EPOCH = 100
BATCH_SIZE = 32
RANDOM_STATE = 122177

# Check script arguments
ap = argparse.ArgumentParser(description="Apply ShallowNet to animal images.")
ap.add_argument("-d", "--dataset", required=True, help="path to the data set")
args = vars(ap.parse_args())

# Extract the full path to each image in the data set's location
imagePaths = list(paths.list_images(args["dataset"]))

# Load the images, resizing each to 32x32 pixels upon loading
resize_pre = ResizePreprocessor(32, 32)
itoarr_pre = ImgToArrayPreprocessor()
norm_pre = NormalisePreprocessor()

dl = MemoryDataLoader(preprocessors=[resize_pre, itoarr_pre, norm_pre])
(X, Y) = dl.load(imagePaths, verbose=500)

# Normalise the images
# X = X.astype("float") / 255.0

# Split into a training and test set
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_STATE)

# One-hot encode the labels
Y_train = LabelBinarizer().fit_transform(Y_train)
Y_test= LabelBinarizer().fit_transform(Y_test)

# Initialise the NN and optimiser
opt = SGD(lr=LEARNING_RATE)
model = ShallowNetNN.build_model(img_width=32, img_height=32, img_channels=3, num_classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, validation_data=(X_test, Y_test), verbose=1)

# Make predictions on the test set and print the results to the console
preds = model.predict(X_test, batch_size=BATCH_SIZE)
target_names = ["cats", "dogs", "panda"]
print(classification_report(Y_test.argmax(axis=-1), preds.argmax(axis=1), target_names=target_names))

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

# Visualise a few random images (could be training and/or test images)
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))

# Load and preprocess as before
(X, Y) = dl.load(imagePaths=imagePaths[idxs])
# X = X.astype("float") / 255.0

# Make predictions
preds = model.predict(X)

# Display results
for (i, imagePath) in enumerate(imagePaths[idxs]):
    image = cv2.imread(imagePath)
    print("Image {} is a {} predicted to be a {} with probability {}%".format(i+1, Y[i], target_names[preds[i].argmax(axis=0)], preds[i].max(axis=0)))

    if target_names[preds[i].argmax(axis=0)] == Y[i]:
        colour = (0, 255, 0)
    else:
        colour = (0, 0, 255)

    cv2.putText(image, "Prediction: {} ({}%)".format(target_names[preds[i].argmax(axis=0)], preds[i].max(axis=0)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=colour, thickness=2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
