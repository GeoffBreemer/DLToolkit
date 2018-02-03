"""AlexNet NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras import backend as K
from .base_nn import BaseNN

# AlexNet architecture parameters
DROPOUT_PERC1 = 0.25
DROPOUT_PERC2 = 0.5
L2_REG_DEFAULT = 0.0002


class AlexNetNN(BaseNN):
    _title = "alexnet"
    _img_width = 227
    _img_height = 227
    _img_channels = 3

    def __init__(self, num_classes):
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

        self._model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="same", input_shape=input_shape,
                               kernel_regularizer=l2(L2_REG_DEFAULT)))
        self._model.add(Activation("relu"))
        self._model.add(BatchNormalization(axis=channel_dim))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self._model.add(Dropout(DROPOUT_PERC1))

        self._model.add(Conv2D(256, (5, 5), padding="same", kernel_regularizer=l2(L2_REG_DEFAULT)))
        self._model.add(Activation("relu"))
        self._model.add(BatchNormalization(axis=channel_dim))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self._model.add(Dropout(DROPOUT_PERC1))

        self._model.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(L2_REG_DEFAULT)))
        self._model.add(Activation("relu"))
        self._model.add(BatchNormalization(axis=channel_dim))

        self._model.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(L2_REG_DEFAULT)))
        self._model.add(Activation("relu"))
        self._model.add(BatchNormalization(axis=channel_dim))

        self._model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(L2_REG_DEFAULT)))
        self._model.add(Activation("relu"))
        self._model.add(BatchNormalization(axis=channel_dim))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self._model.add(Dropout(DROPOUT_PERC1))

        self._model.add(Flatten())
        self._model.add(Dense(4096, kernel_regularizer=l2(L2_REG_DEFAULT)))
        self._model.add(Activation("relu"))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(DROPOUT_PERC2))

        self._model.add(Dense(4096, kernel_regularizer=l2(L2_REG_DEFAULT)))
        self._model.add(Activation("relu"))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(DROPOUT_PERC2))

        self._model.add(Dense(self._num_classes, kernel_regularizer=l2(L2_REG_DEFAULT)))
        self._model.add(Activation("softmax"))

        return self._model
