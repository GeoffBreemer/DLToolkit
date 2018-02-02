"""NN architecture based on NVIDIA's paper: https://arxiv.org/abs/1604.07316"""
from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.core import SpatialDropout2D
from keras.layers import Dense, Dropout, Flatten, Lambda
from .base_nn import BaseNN


class NVIDIA_NN(BaseNN):
    @staticmethod
    def normalize(image):
        '''Normalize the image to be between -0.5 and 0.5'''
        return image / 255.0 - 0.5

    @staticmethod
    def resize(image):
        '''Resize the image to 66x200 as documented in the NVIDIA paper'''
        import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
        return tf.image.resize_images(image, (66, 200))

    @staticmethod
    def build_model(img_width, img_height, img_channels, num_classes):
        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (img_channels, img_height, img_width)
        else:
            input_shape = (img_height, img_width, img_channels)

        # Create the model
        model = Sequential()

        # Create the model pipeline, including image preprocessing (avoids having to change drive.py)
        model = Sequential([

            # Resize and normalize the image
            Lambda(NVIDIA_NN.resize),
            Lambda(NVIDIA_NN.normalize),

            # Conv1
            Conv2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init="he_normal"),
            SpatialDropout2D(0.2),

            # Conv2
            Conv2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init="he_normal"),
            SpatialDropout2D(0.2),

            # Conv3
            Conv2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init="he_normal"),
            SpatialDropout2D(0.2),

            # Conv4
            Conv2D(64, 3, 3, border_mode='valid', activation='elu', init="he_normal"),
            SpatialDropout2D(0.2),

            # Conv5
            Conv2D(64, 3, 3, border_mode='valid', activation='elu', init="he_normal"),
            SpatialDropout2D(0.2),

            # FC1
            Flatten(),
            Dense(100, activation='elu', init="he_normal"),
            Dropout(0.5),

            # FC2
            Dense(50, activation='elu', init="he_normal"),

            # FC3
            Dense(10, activation='elu', init="he_normal"),
            Dropout(0.5),

            # Final layer
            Dense(1)
        ])

        return model
