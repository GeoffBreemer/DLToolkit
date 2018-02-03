"""LeNet NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras import backend as K
from .base_conv_nn import BaseConvNN


class LeNetNN(BaseConvNN):
    _title = "lenet"
    _img_width = 28
    _img_height = 28
    _img_channels = 1

    def __init__(self, num_classes):
        self._num_classes = num_classes

    def build_model(self):
        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (self._img_channels, self._img_height, self._img_height)
        else:
            input_shape = (self._img_height, self._img_width, self._img_channels)

        # Create the model
        self._model = Sequential()

        # First conv layer
        self._model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape, activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # Second conv layer
        self._model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # FC layer
        self._model.add(Flatten())
        self._model.add(Dense(500, activation="relu"))

        # Softmax classifier
        self._model.add(Dense(self._num_classes, activation="softmax"))

        return self._model
