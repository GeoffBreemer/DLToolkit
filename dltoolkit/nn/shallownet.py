"""ShallowNet (just a single FC layer) NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Dense, Flatten
from keras import backend as K


class ShallowNetNN:
    @staticmethod
    def build_model(img_width, img_height, img_channels, num_classes):
        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (img_channels, img_height, img_width)
        else:
            input_shape = (img_height, img_width, img_channels)

        # Create the model
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        return model
