"""LeNet NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.layers import Dense, Flatten
from keras import backend as K

# LeNet architecture parameters
LENET_IMG_WIDTH = 28
LENET_IMG_HEIGHT = 28
LENET_IMG_CHANNELS = 1
LENET_NUM_CLASSES = 10


class LeNetNN:
    @staticmethod
    def build_model():
        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (LENET_IMG_CHANNELS, LENET_IMG_HEIGHT, LENET_IMG_WIDTH)
        else:
            input_shape = (LENET_IMG_HEIGHT, LENET_IMG_WIDTH, LENET_IMG_CHANNELS)

        # Create the model
        model = Sequential()

        # First conv layer
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second conv layer
        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FC layer
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))

        # Softmax classifier
        model.add(Dense(LENET_NUM_CLASSES, activation="softmax"))

        return model
