"""Simple MNIST classification using Keras, Stochastic Gradient Descent and a simple 2-layer FF NN"""
from dltoolkit.utils import plot_history

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants
NUM_EPOCH = 100
LEARNING_RATE = 0.01
BATCH_SIZE = 128

OUTPUT_PATH = "../output/"
MODEL_NAME = "mnist" + "_" + "simple_ff"

# (Down)load the MNIST data set
mnist = datasets.fetch_mldata("MNist Original")

# Normalize images and split into a training and test set
X = mnist.data.astype("float") / 255.0
(X_train, X_test, Y_train, Y_test) = train_test_split(X, mnist.target, test_size=0.25)

# One-hot encode the integer labels
lbl_bin = LabelBinarizer()
Y_train = lbl_bin.fit_transform(Y_train)
Y_test = lbl_bin.transform(Y_test)

# Build the NN: two fully-connected layers
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# Train the NN using SGD and cross-entropy loss
sgd = SGD(lr=LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=NUM_EPOCH, batch_size=BATCH_SIZE)
# note: the test data set should NOT be used for validation_data, but rather a true validation set should be used

# Make predictions on the test set
Y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=1), Y_pred.argmax(axis=1), target_names=[str(c) for c in lbl_bin.classes_]))

# Plot loss and accuracy
plot_history(hist, NUM_EPOCH, show=False, save_path=OUTPUT_PATH + MODEL_NAME, time_stamp=True)
