"""LeNet NN architecture built using Keras"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten
from keras import backend as K


class LeNetNN:
    @staticmethod
    def build_model(img_width, img_height, img_channels, num_classes):
        """Build a LeNet network using Keras"""
        model = Sequential()

        # Set the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (img_channels, img_height, img_width)
        else:
            inputShape = (img_height, img_width, img_channels)

        # Create the model
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        return model
