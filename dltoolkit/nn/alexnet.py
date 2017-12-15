"""AlexNet NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras import backend as K

# AlexNet parameters
ALEX_IMG_WIDTH = 227
ALEX_IMG_HEIGHT = 227
ALEX_IMG_CHANNELS = 3

DROPOUT_PERC1 = 0.25
DROPOUT_PERC2 = 0.5
L2_REG_DEFAULT = 0.0002

class AlexNetNN:
    @staticmethod
    def build_model(img_width, img_height, img_channels, num_classes, reg=L2_REG_DEFAULT):
        # Set the input shape
        inputShape = (img_height, img_width, img_channels)
        channel_dim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (img_channels, img_height, img_width)
            channel_dim = 1

        # Create the model
        model = Sequential()

        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="same", input_shape=inputShape,
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(DROPOUT_PERC1))

        model.add(Conv2D(256, (5, 5), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(DROPOUT_PERC1))

        model.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(DROPOUT_PERC1))

        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_PERC2))

        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_PERC2))

        model.add(Dense(num_classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        return model
