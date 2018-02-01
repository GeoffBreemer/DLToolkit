"""Simple CIFAR-10 classification using Keras, Stochastic Gradient Descent and a simple 2-layer FF NN"""
from sklearn.preprocessing import LabelBinarizer      # one-hot encode
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np

# Constants
LEARNING_RATE = 0.01
NUM_EPOCH = 100
BATCH_SIZE = 32

# Load data
((X_train, Y_train), (X_test, Y_test)) = cifar10.load_data()

# Preprocess: scale to [0..1]
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

# Flatten from (# of records, 32, 32, 3) to (# of records, 32*32*3)
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))
image_shape = X_train.shape[1]

# Binarize labels
lab_bin = LabelBinarizer()
Y_train = lab_bin.fit_transform(Y_train)
Y_test = lab_bin.transform(Y_test)
names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Build the NN: two fully-connected layers
model = Sequential()
model.add(Dense(1024, input_shape=(image_shape, ), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Train the NN using SGD and cross-entropy loss
sgd = SGD(LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=NUM_EPOCH, batch_size=BATCH_SIZE)
# note: the test data set should NOT be used for validation_data, but rather a true validation set should be used

# Make predictions
Y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=1), Y_pred.argmax(axis=1), target_names=names))

# Plot loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, NUM_EPOCH), hist.history["loss"], label="Training loss")
plt.plot(np.arange(0, NUM_EPOCH), hist.history["val_loss"], label="Validation loss")
plt.plot(np.arange(0, NUM_EPOCH), hist.history["acc"], label="Training accuracy")
plt.plot(np.arange(0, NUM_EPOCH), hist.history["val_acc"], label="Validation accuracy")
plt.title("Loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/accuracy")
plt.legend()
plt.show()