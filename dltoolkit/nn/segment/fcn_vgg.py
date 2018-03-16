"""Implementation of FCN32 pre-trained on ImageNet using Keras - NOT TESTED
Based on https://arxiv.org/pdf/1605.06211.pdf

# https://github.com/keras-team/keras/issues/4040
# DEZE: https://github.com/keras-team/keras/issues/3465
# https://github.com/divamgupta/image-segmentation-keras/tree/master/Models                 nice and simple
# https://github.com/aurora95/Keras-FCN
# https://github.com/keras-team/keras/issues/3540
# https://github.com/keras-team/keras/issues/3824
# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
"""
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Cropping2D,\
    Dropout, Activation, Reshape, Conv2DTranspose, merge, Flatten, Dense, Permute
from keras.models import Model
from keras.initializers import RandomNormal
from keras import backend as K
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

    def build_model(self):
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
                         padding='same',
                         name="score"))(output)

        # Upsample to produce pixel-wise predictions, formula for the output size:
        # output_size = (input_size - 1) * stride + kernel_size - 2 * padding

        output = Conv2DTranspose(filters=self._num_classes,
                                 kernel_size=(64, 64),
                                 strides=(32, 32),
                                 padding="valid",
                                 name="upsample",
                                 activation=None,
                                 use_bias=False)(output)

        output_shape = Model(inputs=[input], outputs=[output]).output_shape
        output = Reshape((output_shape[1] * output_shape[2], self._num_classes), name="reshape")(output)
        output = (Activation("softmax", name="softmax"))(output)
        self._model = Model(inputs=[input], outputs=[output])

        return self._model, output_shape

    # def build_model_pretraing_nowork(self):
    #     """Build the FCN architecture using VGG16 pre-trained on ImageNet"""
    #     _title = "FCN_VGG16"
    #
    #     # Create the full VGG16 model and parameters trained on ImageNet
    #     vgg16 = VGG16(include_top=True,
    #                   # weights="imagenet",
    #                   weights=None,
    #                   input_tensor=None,
    #                   pooling=None,
    #                   input_shape=self.input_shape)
    #
    #     # Replace the fully connected layers after the last MaxPooling2D layer with convolutional layers
    #     top_model = vgg16.layers[-5].output
    #
    #     top_model = (Conv2D(4096, (7, 7), activation='relu', padding='same', name="fc6"))(top_model)
    #     # top_model = Dropout(self._dropout_rate)(top_model)
    #     top_model = (Conv2D(4096, (1, 1), activation='relu', padding='same', name="fc7"))(top_model)
    #     # top_model = Dropout(self._dropout_rate)(top_model)
    #
    #     # Add a 1x1 convolution to produce scores
    #     top_model = (Conv2D(self._num_classes, (1, 1),
    #                         kernel_initializer='he_normal',
    #                         activation='relu',
    #                         padding='same',
    #                         name="score_fr"))(top_model)
    #
    #     # Upsample to produce pixel-wise predictions
    #     top_model = Conv2DTranspose(self._num_classes,
    #                                 kernel_size=(64, 64),
    #                                 strides=(32, 32),
    #                                 padding="valid",
    #                                 name="score2",
    #                                 activation=None,
    #                                 use_bias=False)(top_model)
    #
    #     output_shape = Model(inputs=[vgg16.input], outputs=[top_model]).output_shape
    #
    #     top_model = Reshape((output_shape[1] * output_shape[2], self._num_classes))(top_model)
    #     top_model = (Activation("softmax"))(top_model)
    #     self._model = Model(inputs=[vgg16.input], outputs=[top_model])
    #
    #     return self._model, output_shape
    #
    # def build_model_manual(self):
    #     """Build the FCN architecture based on VGG16 manually without using keras.applications.VGG16"""
    #     _title = "FCN32s_VGG16_untrained"
    #
    #     # Set the input shape
    #     input_shape = (self._img_height, self._img_width, self._img_channels)
    #     inputs = Input(input_shape)
    #
    #     # Create the VGG16 model
    #     x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    #     x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    #     x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #     f1 = x
    #
    #     # Block 2
    #     x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    #     x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    #     x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #     f2 = x
    #
    #     # Block 3
    #     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    #     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    #     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    #     x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #     f3 = x
    #
    #     # Block 4
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    #     x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #     f4 = x
    #
    #     # Block 5
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    #     x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #     f5 = x
    #
    #     x = Flatten(name='flatten')(x)
    #     x = Dense(4096, activation='relu', name='fc1')(x)
    #     x = Dense(4096, activation='relu', name='fc2')(x)
    #     x = Dense(1000, activation='softmax', name='predictions')(x)
    #
    #     # Load the VGG16 weights
    #     vgg = Model(inputs=[inputs], outputs=[x])
    #     # vgg.load_weights(VGG_Weights_path)
    #
    #     # Replace the fully connected layers with convolutional layers
    #     o = f5
    #     o = (Conv2D(4096, (7, 7), activation='relu', padding='same'))(o)
    #     # o = Dropout(self._dropout_rate)(o)
    #     o = (Conv2D(4096, (1, 1), activation='relu', padding='same'))(o)
    #     # o = Dropout(self._dropout_rate)(o)
    #
    #     o = (Conv2D(self._num_classes, (1, 1), kernel_initializer='he_normal'))(o)
    #     o = Conv2DTranspose(self._num_classes, kernel_size=(64, 64), strides=(32, 32), use_bias=False)(o)
    #     o_shape = Model(inputs=[inputs], outputs=[o]).output_shape
    #
    #     print("output shape: {}".format(o_shape))
    #
    #     outputHeight = o_shape[1]
    #     outputWidth = o_shape[2]
    #
    #     o = (Reshape((-1, outputHeight * outputWidth)))(o)
    #     o = (Permute((2, 1)))(o)
    #     o = (Activation('softmax'))(o)
    #     self._model = Model(inputs=[inputs], outputs=[o])
    #
    #     return self._model
