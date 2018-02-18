from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Cropping2D, Dropout, Activation, Reshape, Permute
from keras.models import Model
from keras.initializers import RandomNormal
from keras import backend as K
from dltoolkit.nn.base_nn import BaseNN
from math import sqrt


class UNet_NN(BaseNN):
    _title = "UNet"

    def __init__(self, img_height, img_width, img_channels, num_classes, dropout_rate=0.0):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._dropout_rate = dropout_rate
        self._num_classes = num_classes

    def get_unet(self):
        inputs = Input(shape=(self._img_height, self._img_width, self._img_channels))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        #
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        #
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(conv3)

        up1 = UpSampling2D(size=(2, 2))(conv3)
        up1 = concatenate([conv2, up1], axis=3)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(up1)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(conv4)
        #
        up2 = UpSampling2D(size=(2, 2))(conv4)
        up2 = concatenate([conv1, up2], axis=3)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(up2)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(conv5)
        #
        conv6 = Conv2D(self._num_classes, (1, 1), activation='relu', padding='same', kernel_initializer="he_normal")(conv5)
        conv6 = Reshape((self._img_height * self._img_width, self._num_classes))(conv6)
        # conv6 = Permute((2, 1))(conv6)
        ############
        # conv7 = Activation('sigmoid')(conv6)
        conv7 = Activation('softmax')(conv6)
        # conv7 = Reshape((self._img_height, self._img_width, self._num_classes))(conv7)

        self._model = Model(input=inputs, output=conv7)

        return self._model


    def build_model(self):
        """Build the U-Net architecture used for the DRIVE data set"""
        _title = "UNet_DRIVE"

        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (self._img_channels, self._img_height, self._img_height)
        else:
            input_shape = (self._img_height, self._img_width, self._img_channels)
            print("CHANNELS LAST")

        inputs = Input(input_shape)

        # Contracting path
        conv_contr1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(self._img_width * self._img_height * self._img_channels))))(inputs)
        # conv_contr1 = Dropout(self._dropout_rate)(conv_contr1)
        conv_contr1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 32))))(conv_contr1)
        pool_contr1 = MaxPooling2D(pool_size=(2, 2))(conv_contr1)

        conv_contr2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 32))))(pool_contr1)
        # conv_contr2 = Dropout(self._dropout_rate)(conv_contr2)
        conv_contr2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 64))))(conv_contr2)
        pool_contr2 = MaxPooling2D(pool_size=(2, 2))(conv_contr2)

        # "Bottom" layer
        conv_bottom = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 64))))(pool_contr2)
        # conv_bottom = Dropout(self._dropout_rate)(conv_bottom)
        conv_bottom = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 128))))(conv_bottom)

        # Crop outputs of each contracting path "layer" for use in their corresponding expansive path "layer"
        # crop_up1 = Cropping2D(cropping=((88, 88), (88, 88)))(conv_contr1)
        # crop_up2 = Cropping2D(cropping=((40, 40), (40, 40)))(conv_contr2)
        # crop_up1 = Cropping2D(cropping=((0, 0), (0, 0)))(conv_contr1)
        # crop_up2 = Cropping2D(cropping=((0, 0), (0, 0)))(conv_contr2)
        crop_up1 = conv_contr1  # no cropping required
        crop_up2 = conv_contr2  # no cropping required

        # Expansive path
        scale_up2 = UpSampling2D(size=(2, 2))(conv_bottom)
        conv_scale_up2 = Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 128))))(scale_up2)
        merge_up2 =  concatenate([conv_scale_up2, crop_up2], axis=3)
        conv_up2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 64))))(merge_up2)
        # conv_up2 = Dropout(self._dropout_rate)(conv_up2)
        conv_up2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 64))))(conv_up2)

        scale_up1 = UpSampling2D(size=(2, 2))(conv_up2)
        conv_scale_up1 = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 64))))(scale_up1)
        merge_up1 =  concatenate([conv_scale_up1, crop_up1], axis=3)
        conv_up1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 32))))(merge_up1)
        # conv_up1 = Dropout(self._dropout_rate)(conv_up1)
        conv_up1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 32))))(conv_up1)

        # Final 1x1 conv layer
        conv_final = Conv2D(self._num_classes , (1, 1), activation='relu', padding='same',
                             kernel_initializer=RandomNormal(stddev=sqrt(2/(3 * 3 * 32))))(conv_up1)
        conv_final = Reshape((self._num_classes, self._img_height * self._img_width))(conv_final)  # TODO org
        conv_final = Permute((2, 1))(conv_final)                                                    # TODO org
        # conv_final = Reshape((self._img_height * self._img_width, self._num_classes))(conv_final) # TODO meh
        # conv_final = Reshape((-1, self._num_classes))(conv_final)  # TODO meh
        conv_final = Activation('softmax')(conv_final)

        self._model = Model(inputs=[inputs], outputs=[conv_final])

        return self._model

    def build_model_paper(self):
        """
        Build the U-Net architecture as defined by Ronneberger et al:
        https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

        Uses an input shape of 572x572. Instantiate the model using:

            UNet_NN(img_height=572,
                    img_width=572,
                    img_channels=1).build_model()
        """
        _title = "UNet_paper"

        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (self._img_channels, self._img_height, self._img_height)
        else:
            input_shape = (self._img_height, self._img_width, self._img_channels)

        inputs = Input(input_shape)

        # Contracting path, from the paper:
        # The contracting path follows the typical architecture of a convolutional network. It consists of the
        # repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear
        # unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we
        # double the number of feature channels.
        conv_contr1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')(inputs)
        conv_contr1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')(conv_contr1)
        pool_contr1 = MaxPooling2D(pool_size=(2, 2))(conv_contr1)

        conv_contr2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid')(pool_contr1)
        conv_contr2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid')(conv_contr2)
        pool_contr2 = MaxPooling2D(pool_size=(2, 2))(conv_contr2)

        conv_contr3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='valid')(pool_contr2)
        conv_contr3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='valid')(conv_contr3)
        pool_contr3 = MaxPooling2D(pool_size=(2, 2))(conv_contr3)

        conv_contr4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='valid')(pool_contr3)
        conv_contr4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='valid')(conv_contr4)
        pool_contr4 = MaxPooling2D(pool_size=(2, 2))(conv_contr4)

        # "Bottom" layer
        conv_bottom = Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='valid')(pool_contr4)
        conv_bottom = Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='valid')(conv_bottom)

        # Crop outputs of each contracting path "layer" for use in their corresponding expansive path "layer"
        crop_up1 = Cropping2D(cropping=((88, 88), (88, 88)))(conv_contr1)
        crop_up2 = Cropping2D(cropping=((40, 40), (40, 40)))(conv_contr2)
        crop_up3 = Cropping2D(cropping=((16, 16), (16, 16)))(conv_contr3)
        crop_up4 = Cropping2D(cropping=((4, 4), (4, 4)))(conv_contr4)

        # Expansive path, from the paper:
        # Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution
        # (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly
        # cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU
        scale_up4 = UpSampling2D(size=(2, 2))(conv_bottom)
        conv_scale_up4 = Conv2D(filters=512, kernel_size=(2, 2), activation='relu', padding='same')(scale_up4)
        merge_up4 =  concatenate([conv_scale_up4, crop_up4], axis=3)
        conv_up4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='valid')(merge_up4)
        conv_up4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='valid')(conv_up4)

        scale_up3 = UpSampling2D(size=(2, 2))(conv_up4)
        conv_scale_up3 = Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same')(scale_up3)
        merge_up3 =  concatenate([conv_scale_up3, crop_up3], axis=3)
        conv_up3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='valid')(merge_up3)
        conv_up3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='valid')(conv_up3)

        scale_up2 = UpSampling2D(size=(2, 2))(conv_up3)
        conv_scale_up2 = Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same')(scale_up2)
        merge_up2 =  concatenate([conv_scale_up2, crop_up2], axis=3)
        conv_up2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid')(merge_up2)
        conv_up2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid')(conv_up2)

        scale_up1 = UpSampling2D(size=(2, 2))(conv_up2)
        conv_scale_up1 = Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same')(scale_up1)
        merge_up1 =  concatenate([conv_scale_up1, crop_up1], axis=3)
        conv_up1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')(merge_up1)
        conv_up1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')(conv_up1)

        # Final 1x1 conv layer
        conv_final = Conv2D(filters=2, kernel_size=(1, 1), activation='sigmoid')(conv_up1)

        self._model = Model(input=inputs, output=conv_final)

        return self._model
