"""Mini (shallower version) of the VGG16 NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import backend as K
from .base_nn import BaseNN

# Architecture parameters
MINIVGGNET_DROPOUT_PERC1 = 0.25
MINIVGGNET_DROPOUT_PERC2 = 0.5


class MiniVGGNN(BaseNN):
    _title = "minivggnet"

    def __init__(self, img_width, img_height, img_channels, num_classes):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._num_classes = num_classes

    def build_model(self):
        # Set the input shape
        input_shape = (self._img_height, self._img_width, self._img_channels)
        channel_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (self._img_channels, self._img_height, self._img_width)
            channel_dim = 1

        # Create the model
        self._model = Sequential()

        self._model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu"))
        self._model.add(BatchNormalization(axis=channel_dim))

        self._model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        self._model.add(BatchNormalization(axis=channel_dim))

        self._model.add(MaxPooling2D(pool_size=(2,2)))
        self._model.add(Dropout(MINIVGGNET_DROPOUT_PERC1))

        self._model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        self._model.add(BatchNormalization(axis=channel_dim))

        self._model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        self._model.add(BatchNormalization(axis=channel_dim))

        self._model.add(MaxPooling2D(pool_size=(2,2)))
        self._model.add(Dropout(MINIVGGNET_DROPOUT_PERC1))

        self._model.add(Flatten())
        self._model.add(Dense(512, activation="relu"))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(MINIVGGNET_DROPOUT_PERC2))

        self._model.add(Dense(self._num_classes, activation="softmax"))

        return self._model
