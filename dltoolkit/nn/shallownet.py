"""ShallowNet NN architecture built using Keras: single FC layer"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Dense, Flatten
from keras import backend as K


class ShallowNetNN:
    @staticmethod
    def build_model(img_width, img_height, img_channels, num_classes):
        model = Sequential()

        # Set the input shape
        inputShape = (img_height, img_width, img_channels)
        channel_dim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (img_channels, img_height, img_width)
            channel_dim = 1

        # Create the model
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        return model
