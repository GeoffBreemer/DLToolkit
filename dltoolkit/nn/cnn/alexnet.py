"""AlexNet NN architecture built using Keras"""
from .base_conv_nn import BaseConvNN

from keras.models import Sequential, Model
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Permute
from keras.layers import Activation, Dense, Flatten, Dropout, Input
from keras.regularizers import l2
from keras import backend as K

# AlexNet architecture parameters
DROPOUT_PERC1 = 0.25
DROPOUT_PERC2 = 0.5
L2_REG_DEFAULT = 0.0002


class AlexNetNN(BaseConvNN):
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


    def build_model_conv(self, img_num_channels):
        # Set the input shape
        input_shape = (self._img_height, self._img_width, img_num_channels)
        channel_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (img_num_channels, self._img_height, self._img_width)
            channel_dim = 1

        input = Input(shape=input_shape, name='image_input')

        output = Conv2D(96,
                        (11, 11),
                        strides=(4, 4),
                        padding="same",
                        input_shape=input_shape,
                        kernel_regularizer=l2(L2_REG_DEFAULT),
                        activation="relu")(input)
        output = BatchNormalization(axis=channel_dim)(output)
        output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(output)
        output = Dropout(DROPOUT_PERC1)(output)

        output = Conv2D(256,
                        (5, 5),
                        strides=(1, 1),
                        padding="same",
                        kernel_regularizer=l2(L2_REG_DEFAULT),
                        activation="relu")(output)
        output = BatchNormalization(axis=channel_dim)(output)
        output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(output)
        output = Dropout(DROPOUT_PERC1)(output)

        output = Conv2D(384,
                        (3, 3),
                        strides=(1, 1),
                        padding="same",
                        kernel_regularizer=l2(L2_REG_DEFAULT),
                        activation="relu")(output)
        output = BatchNormalization(axis=channel_dim)(output)

        output = Conv2D(384,
                        (3, 3),
                        strides=(1, 1),
                        padding="same",
                        kernel_regularizer=l2(L2_REG_DEFAULT),
                        activation="relu")(output)
        output = BatchNormalization(axis=channel_dim)(output)

        output = Conv2D(256,
                        (3, 3),
                        strides=(1, 1),
                        padding="same",
                        kernel_regularizer=l2(L2_REG_DEFAULT),
                        activation="relu")(output)
        output = BatchNormalization(axis=channel_dim)(output)
        output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(output)
        output = Dropout(DROPOUT_PERC1)(output)

        # Add convolutional layers
        output = Conv2D(4096, (6, 6), strides=(1, 1), activation='relu', padding='same', name="fc6conv")(output)
        # output = Dropout(self._dropout_rate)(output)
        output = Conv2D(4096, (1, 1), strides=(1, 1), activation='relu', padding='same', name="fc7conv")(output)
        # output = Dropout(self._dropout_rate)(output)

        # Add a 1x1 convolution to produce scores
        output = Conv2D(self._num_classes,
                        (1, 1),
                        kernel_initializer='he_normal',
                        activation='relu',
                        padding='same',
                        name="score")(output)

        # Upsample to produce pixel-wise predictions, formula for the output size:
        # output_size = (input_size - 1) * stride + kernel_size - 2 * padding

        output = Conv2DTranspose(filters=self._num_classes,
                                 kernel_size=(64, 64),
                                 strides=(32, 32),
                                 padding="valid",
                                 name="upsample",
                                 activation=None,
                                 use_bias=False)(output)

        # Determine the output shape after upsampling
        output_shape = Model(input, output).output_shape
        outputHeight = output_shape[1]
        outputWidth = output_shape[2]

        output = Reshape((outputHeight * outputWidth, self._num_classes))(output)
        # output = Permute((2, 1))(output)
        output = Activation("softmax")(output)

        self._model = Model(inputs=[input], outputs=[output])

        return self._model, (outputHeight, outputWidth)
