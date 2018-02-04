"""ShallowNet (just a single Conv layer) NN architecture built using Keras"""
from .base_conv_nn import BaseConvNN

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras import backend as K


class ShallowNetNN(BaseConvNN):
    _title = "shallownet"
    _img_width = 32
    _img_height = 32
    _img_channels = 3

    def __init__(self, num_classes):
        self._num_classes = num_classes

    def build_model(self):
        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (self._img_channels, self._img_height, self._img_width)
        else:
            input_shape = (self._img_height, self._img_width, self._img_channels)

        # Create the model
        self._model = Sequential()

        self._model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu"))

        self._model.add(Flatten())
        self._model.add(Dense(self._num_classes, activation="softmax"))

        return self._model
