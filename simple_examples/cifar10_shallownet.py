"""Simple CIFAR-10 classification using Keras, Stochastic Gradient Descent and ShallowNet"""
from sklearn.preprocessing import LabelBinarizer      # one-hot encode
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
from dltoolkit.nn import ShallowNetNN

import matplotlib.pyplot as plt
import numpy as np

# Constants
LEARNING_RATE = 0.01
NUM_EPOCH = 40
BATCH_SIZE = 32

# Load data
((X_train, Y_train), (X_test, Y_test)) = cifar10.load_data()

# Preprocess: scale to [0..1]
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

# Binarize labels
lab_bin = LabelBinarizer()
Y_train = lab_bin.fit_transform(Y_train)
Y_test = lab_bin.transform(Y_test)
names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Initialise the NN and optimiser
opt = SGD(lr=LEARNING_RATE)
model = ShallowNetNN.build_model(img_width=32, img_height=32, img_channels=3, num_classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, validation_data=(X_test, Y_test), verbose=1)

# Make predictions on the test set and print the results to the console
preds = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=-1), preds.argmax(axis=1), target_names=names))

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
