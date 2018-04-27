"""Implementation of FCN32 pre-trained on ImageNet using Keras:
Based on https://arxiv.org/pdf/1605.06211.pdf
"""
from keras.layers import Conv2D, Activation, Conv2DTranspose, Cropping2D, BatchNormalization, Reshape
from keras.models import Model
from keras.applications import VGG16
from keras.regularizers import l2
from dltoolkit.nn.base_nn import BaseNN


class FCN32_ImageNet_NN(BaseNN):
    _title = "FCN32-ImageNet"

    def __init__(self, img_height, img_width, img_channels, num_classes, dropout_rate=0.0):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._num_classes = num_classes

        self._dropout_rate = dropout_rate
        self.input_shape = (self._img_height, self._img_width, self._img_channels)
        self._vgg_body = None

    def freeze_vgg_layers(self, freeze):
        """(Un)freeze all VGG layers"""
        for layer in self._vgg_body.layers:
            layer.trainable = freeze

    def build_model(self, crop=16, use_bn=False):
        """Build the FCN-32s architecture as defined by Long et al (http://arxiv.org/abs/1411.4038)
        and use weights pre-trained on ImageNet
        """
        self._title+= "_BN" if use_bn else ""

        l2_reg = 5e-4

        # Load VGG with ImageNet trained weights but without the FC layers
        self._vgg_body = VGG16(include_top=False,
                               weights="imagenet",
                               input_shape=self.input_shape,
                               pooling=None)

        # Add the FCN-32s specific layers as a replacement for the VGG fully connected layers
        x = Conv2D(4096, (7, 7), activation="relu", padding="same",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(self._vgg_body.output)
        x = BatchNormalization()(x) if use_bn else x

        x = Conv2D(4096, (1, 1), activation="relu", padding="same",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x) if use_bn else x

        x = Conv2D(self._img_channels * self._num_classes, (1, 1), kernel_initializer="he_normal",
                   kernel_regularizer=l2(l2_reg))(x)
        x = Conv2DTranspose(self._img_channels * self._num_classes, kernel_size=(64, 64), strides=(32, 32),
                            kernel_initializer="he_normal", use_bias=False)(x)
        x = Cropping2D(((crop, crop), (crop, crop)))(x)

        x = Reshape((self._img_height, self._img_width, self._img_channels, self._num_classes))(x)
        x = (Activation("softmax"))(x)

        self._model = Model(inputs=self._vgg_body.input, outputs=[x])

        return self._model
