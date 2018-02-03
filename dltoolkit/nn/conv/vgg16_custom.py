"""Wrapper for the VGG16 architecture supporting swapping out the original FC layers with custom ones"""
from keras.applications import VGG16
from keras import backend as K
from keras.layers import Dense, Flatten, Dropout, Input
from keras.models import Model
from .base_conv_nn import BaseConvNN


class VGG16CustomNN(BaseConvNN):
    _title = "vgg16"
    _img_width = 224
    _img_height = 224
    _img_channels = 3

    def __init__(self, num_classes, dense_size=256):
        """
        Initialise the class
        :param num_classes: number of classes to classify
        :param dense_size: number of neurons in the FC layer
        """
        self._num_classes = num_classes
        self._dense_size = dense_size

    def build_model(self):
        # Set the input shape
        input_shape = (self._img_height, self._img_width, self._img_channels)
        if K.image_data_format() == "channels_first":
            input_shape = (self._img_channels, self._img_height, self._img_width)

        # Load the VGG16 model without the FC layers
        self._vgg16_model = VGG16(weights="imagenet",
                                  include_top=False,
                                  input_tensor=Input(shape=input_shape))

        # Freeze all remaining VGG16 layers to prevent changing the weights during 'warm-up'
        for l in self._vgg16_model.layers:
            l.trainable = False

        # Add the custom FC, Dropout and softmax layers
        self._custom_fc = self._vgg16_model.output
        self._custom_fc = Flatten(name="flatten")(self._custom_fc)
        self._custom_fc = Dense(self._dense_size, activation="relu")(self._custom_fc)
        self._custom_fc = Dropout(0.5)(self._custom_fc)
        self._custom_fc = Dense(self._num_classes, activation="softmax")(self._custom_fc)

        # Build the model
        self._model = Model(inputs=self._vgg16_model.input, outputs=self._custom_fc)

        return self._model

    def unfreeze_vgg_layers(self, starting_layer):
        """Unfreeze the final VGG16 conv layers, starting with layer # starting_layer"""
        for l in self._vgg16_model.layers[starting_layer:]:
            l.trainable=True

    def layer_info(self):
        """List the VGG16 layers, their type and whether they are trainable"""
        print("VGG16 layers:")
        for (i, layer) in enumerate(self._vgg16_model.layers):
            print("Layer {}\t{}\t{}".format(i, layer.__class__.__name__, layer.trainable))

    def __str__(self):
        return super().__str__() + ", fc={}".format(self._dense_size)
