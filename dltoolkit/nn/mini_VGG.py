"""MiniVGG NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras import backend as K

DROPOUT_PERC1 = 0.25
DROPOUT_PERC2 = 0.5

class MiniVGGNN:
    @staticmethod
    def build_model(img_width, img_height, img_channels, num_classes):
        # Set the input shape
        inputShape = (img_height, img_width, img_channels)
        channel_dim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (img_channels, img_height, img_width)
            channel_dim = 1

        # Create the model
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(DROPOUT_PERC1))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(DROPOUT_PERC1))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_PERC2))

        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        return model