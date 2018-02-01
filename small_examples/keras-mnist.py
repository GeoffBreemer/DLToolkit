"""Simple Keras model for MNIST classification

Uses Stochastic Gradient Descent"""
from sklearn.preprocessing import LabelBinarizer      # one-hot encode
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Constants
NUM_EPOCH = 100
LEARNING_RATE = 0.01
BATCH_SIZE = 128

# (down)load MNIST
mnist = datasets.fetch_mldata("MNist Original")

# Preprocess and split
X = mnist.data.astype("float") / 255.0
(X_train, X_test, Y_train, Y_test) = train_test_split(X, mnist.target, test_size=0.25)

# Encode labels
lb = LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
Y_test = lb.transform(Y_test)

# Build NN
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# Train
sgd = SGD(lr=LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=NUM_EPOCH, batch_size=BATCH_SIZE)

# Predict
Y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=1), Y_pred.argmax(axis=1), target_names=[str(c) for c in lb.classes_]))

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
