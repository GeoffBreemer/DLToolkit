"""ShallowNet (just a single FC layer) NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense, Flatten
from keras import backend as K

# ShallowNet architecture parameters
SHALLOWNET_IMG_WIDTH = 32
SHALLOWNET_IMG_HEIGHT = 32
SHALLOWNET_IMG_CHANNELS = 3

class ShallowNetNN:
    @staticmethod
    def build_model(num_classes):
        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (SHALLOWNET_IMG_CHANNELS, SHALLOWNET_IMG_HEIGHT, SHALLOWNET_IMG_WIDTH)
        else:
            input_shape = (SHALLOWNET_IMG_HEIGHT, SHALLOWNET_IMG_WIDTH, SHALLOWNET_IMG_CHANNELS)

        # Create the model
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu"))

        model.add(Flatten())
        model.add(Dense(num_classes, activation="softmax"))

        return model
