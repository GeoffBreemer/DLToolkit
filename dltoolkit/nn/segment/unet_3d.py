"""Implementation of a 3D U-Net using Keras - NOT TESTED"""
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Cropping2D,\
    Dropout, Activation, Reshape, Conv2DTranspose
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D
from keras.models import Model
from keras.initializers import RandomNormal
from keras import backend as K
from dltoolkit.nn.base_nn import BaseNN


class UNet_3D_NN(BaseNN):
    _title = "UNet3D"

    def __init__(self, img_height=132, img_width=132, num_slices=116, img_channels=3, num_classes=3, dropout_rate=0.0):
        """Input: 132 x 132 x 116 with 3 color channels
         Output: 44 x 44 x 28 with 3 output classes
         """
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._num_slices = num_slices
        self._dropout_rate = dropout_rate
        self._num_classes = num_classes

    def build_model(self):
        """
        Build the 3D U-Net architecture as defined by Cicek et al:
        https://arxiv.org/abs/1606.06650
        """
        _title = "UNet3D_paper"

        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (self._num_slices, self._img_channels, self._img_height, self._img_height)
        else:
            input_shape = (self._img_height, self._img_width, self._num_slices, self._img_channels)

        inputs = Input(input_shape)
        print(inputs)

        # Contracting path, from the paper:
        conv_contr1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr1_1")(inputs)
        conv_contr1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr1_2")(conv_contr1)
        pool_contr1 = MaxPooling3D(pool_size=(2, 2, 2), name="contr1_mp")(conv_contr1)

        conv_contr2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr2_1")(pool_contr1)
        conv_contr2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr2_2")(conv_contr2)
        pool_contr2 = MaxPooling3D(pool_size=(2, 2, 2), name="contr2_mp")(conv_contr2)

        conv_contr3 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr3_1")(pool_contr2)
        conv_contr3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr3_2")(conv_contr3)
        pool_contr3 = MaxPooling3D(pool_size=(2, 2, 2), name="contr3_mp")(conv_contr3)

        # "Bottom" layer
        conv_bottom = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', name="bottom1")(pool_contr3)
        conv_bottom = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', name="bottom2")(conv_bottom)


        # Crop outputs of each contracting path "layer" for use in their corresponding expansive path "layer"
        crop_up1 = Cropping3D(cropping=((12, 12), (12, 12), (12, 12)))(conv_contr1)
        crop_up2 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv_contr2)
        # crop_up3 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv_contr3)

        # crop_up1 = conv_contr1
        # crop_up2 = conv_contr2
        crop_up3 = conv_contr3

        # Expansive path:
        scale_up3 = UpSampling3D(size=(2, 2, 2))(conv_bottom)
        conv_scale_up3 = Conv3D(filters=256, kernel_size=(2, 2, 2), activation='relu', padding='same')(scale_up3)
        merge_up3 =  concatenate([conv_scale_up3, crop_up3], axis=4)
        conv_up3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='valid')(merge_up3)
        conv_up3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='valid')(conv_up3)

        scale_up2 = UpSampling3D(size=(2, 2, 2))(conv_up3)
        conv_scale_up2 = Conv3D(filters=128, kernel_size=(2, 2, 2), activation='relu', padding='same')(scale_up2)
        merge_up2 =  concatenate([conv_scale_up2, crop_up2], axis=4)
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='valid')(merge_up2)
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='valid')(conv_up2)

        scale_up1 = UpSampling3D(size=(2, 2, 2))(conv_up2)
        conv_scale_up1 = Conv3D(filters=64, kernel_size=(2, 2, 2), activation='relu', padding='same')(scale_up1)
        merge_up1 =  concatenate([conv_scale_up1, crop_up1], axis=4)
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='valid')(merge_up1)
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='valid')(conv_up1)

        # Final 1x1 conv layer
        conv_final = Conv3D(filters=self._num_classes, kernel_size=(1, 1, 1), activation='sigmoid')(conv_up1)

        self._model = Model(inputs=inputs, outputs=conv_final)

        return self._model
