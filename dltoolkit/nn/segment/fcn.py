"""Implementation of a FCN-32s using Keras
"""
from keras.layers import Input, Conv2D, MaxPooling2D, Cropping2D, Activation, Conv2DTranspose,\
    BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from dltoolkit.nn.base_nn import BaseNN


class FCN32_NN(BaseNN):
    _title = "FCN-32s"

    def __init__(self, img_height, img_width, img_channels, num_classes, dropout_rate=0.0):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._num_classes = num_classes

        self._dropout_rate = dropout_rate
        self.input_shape = (self._img_height, self._img_width, self._img_channels)

    def build_model(self, crop=16, use_bn=False):
        """Build the FCN-32s architecture as defined by Long et al:
        http://arxiv.org/abs/1411.4038
        """
        self._title+= "_BN" if use_bn else ""

        l2_reg = 5e-4

        # Set the input shape
        input_shape = (self._img_height, self._img_width, self._img_channels)
        inputs = Input(input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(inputs)
        # x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        # x = BatchNormalization()(x) if use_bn else x
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

        x = Conv2D(4096, (7, 7), activation="relu", padding="same",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x) if use_bn else x
        x = Conv2D(4096, (1, 1), activation="relu", padding="same",
                   kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x) if use_bn else x

        x = Conv2D(self._num_classes, (1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=l2(l2_reg))(x)
        x = Conv2DTranspose(self._num_classes, kernel_size=(64, 64), strides=(32, 32),
                            kernel_initializer="he_normal", use_bias=False)(x)
        x = Cropping2D(((crop, crop), (crop, crop)))(x)

        x = (Activation("softmax"))(x)

        self._model = Model(inputs=[inputs], outputs=[x])

        return self._model
