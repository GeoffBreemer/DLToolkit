from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Cropping2D
from keras.layers import Convolution2D, merge
from keras.models import Model
from keras import backend as K
from dltoolkit.nn.base_nn import BaseNN


class UNet_NN(BaseNN):
    def __init__(self, img_height, img_width, img_channels):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels

    def build_model(self):
        """
        Build the U-Net architecture as defined by Ronneberger et al:
        https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

        Uses input shape of 572x572
        """

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

        # Expansive path, from the paper:
        # Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution
        # (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly
        # cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU
        scale_up4 = UpSampling2D(size=(2, 2))(conv_bottom)
        conv_scale_up4 = Conv2D(filters=512, kernel_size=(2, 2), activation='relu', padding='same')(scale_up4)
        crop_up4 = Cropping2D(cropping=((4, 4), (4, 4)))(conv_contr4)
        merge_up4 =  concatenate([conv_scale_up4, crop_up4], axis=3)
        conv_up4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='valid')(merge_up4)
        conv_up4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='valid')(conv_up4)

        scale_up3 = UpSampling2D(size=(2, 2))(conv_up4)
        conv_scale_up3 = Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same')(scale_up3)
        crop_up3 = Cropping2D(cropping=((16, 16), (16, 16)))(conv_contr3)
        merge_up3 =  concatenate([conv_scale_up3, crop_up3], axis=3)
        conv_up3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='valid')(merge_up3)
        conv_up3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='valid')(conv_up3)

        scale_up2 = UpSampling2D(size=(2, 2))(conv_up3)
        conv_scale_up2 = Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same')(scale_up2)
        crop_up2 = Cropping2D(cropping=((40, 40), (40, 40)))(conv_contr2)
        merge_up2 =  concatenate([conv_scale_up2, crop_up2], axis=3)
        conv_up2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid')(merge_up2)
        conv_up2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid')(conv_up2)

        scale_up1 = UpSampling2D(size=(2, 2))(conv_up2)
        conv_scale_up1 = Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same')(scale_up1)
        crop_up1 = Cropping2D(cropping=((88, 88), (88, 88)))(conv_contr1)
        merge_up1 =  concatenate([conv_scale_up1, crop_up1], axis=3)
        conv_up1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')(merge_up1)
        conv_up1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')(conv_up1)

        # Final 1x1 conv layer
        conv_final = Conv2D(filters=2, kernel_size=(1, 1), activation='sigmoid')(conv_up1)

        self._model = Model(input=inputs, output=conv_final)

        return self._model
