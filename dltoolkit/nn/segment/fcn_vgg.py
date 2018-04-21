"""Implementation of FCN32 pre-trained on ImageNet using Keras - NOT TESTED
Based on https://arxiv.org/pdf/1605.06211.pdf
"""
from keras.layers import Input, Conv2D, Activation, Reshape, Conv2DTranspose, Cropping2D
from keras.models import Model
from keras.applications import VGG16
from dltoolkit.nn.base_nn import BaseNN


class FCN32_VGG16_NN(BaseNN):
    _title = "FCN32-VGG16-ImageNet"

    def __init__(self, img_height, img_width, img_channels, num_classes, dropout_rate=0.0):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._num_classes = num_classes

        self._dropout_rate = dropout_rate
        self.input_shape = (self._img_height, self._img_width, self._img_channels)
        self._vgg16 = None

    def set_vgg16_trainable(self, trainable, starting_layer):
        for layer in self._vgg16.layers[starting_layer:]:
            layer.trainable = trainable

    def build_model(self, crop=16):
        """Build the FCN architecture using VGG16 pre-trained on ImageNet"""
        _title = "FCN_VGG16"

        # Create the full VGG16 model and parameters trained on ImageNet
        self._vgg16 = VGG16(include_top=False,
                      weights="imagenet")

        print("vgg16: {}".format(self._vgg16))

        input = Input(shape=self.input_shape, name='image_input')
        output = self._vgg16(input)

        output = (Conv2D(4096, (7, 7), activation='relu', padding='same', name="fc6conv"))(output)
        # output = Dropout(self._dropout_rate)(output)
        output = (Conv2D(4096, (1, 1), activation='relu', padding='same', name="fc7conv"))(output)
        # output = Dropout(self._dropout_rate)(output)

        # Add a 1x1 convolution to produce scores
        output = (Conv2D(self._num_classes, (1, 1),
                         kernel_initializer='he_normal',
                         activation='relu',
                         # padding='same',
                         name="score"))(output)

        # Upsample to produce pixel-wise predictions, formula for the output size:
        # output_size = (input_size - 1) * stride + kernel_size - 2 * padding

        output = Conv2DTranspose(filters=self._num_classes,
                                 kernel_size=(64, 64),
                                 strides=(32, 32),
                                 # padding="valid",
                                 name="upsample",
                                 activation=None,
                                 use_bias=False)(output)

        output = Cropping2D(((crop, crop), (crop, crop)))(output)

        # output_shape = Model(inputs=[input], outputs=[output]).output_shape
        # output = Reshape((output_shape[1] * output_shape[2], self._num_classes), name="reshape")(output)
        output = (Activation("softmax", name="softmax"))(output)
        self._model = Model(inputs=[input], outputs=[output])

        return self._model#, output_shape
