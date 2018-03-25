"""Implementation of a 3D U-Net using Keras"""
from keras.layers import concatenate, Activation, Input, Conv3D, MaxPooling3D, UpSampling3D,\
    Cropping3D, BatchNormalization
from keras.models import Model
from keras import backend as K
from dltoolkit.nn.base_nn import BaseNN


class UNet_3D_NN(BaseNN):
    _title = "UNet3D"

    def __init__(self, img_height, img_width, num_slices, img_channels, num_classes):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._num_slices = num_slices
        self._num_classes = num_classes

    def build_model_no_BN(self):
        """
        Build the 3D U-Net architecture as defined by Cicek et al:
        https://arxiv.org/abs/1606.06650
        """
        self._title = "UNet3D_brain_no_BN"

        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (self._num_slices, self._img_channels, self._img_height, self._img_height)
        else:
            input_shape = (self._img_height, self._img_width, self._num_slices, self._img_channels)

        inputs = Input(input_shape)

        # Contracting path, from the paper:
        conv_contr1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr1_1")(inputs)
        conv_contr1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr1_2")(conv_contr1)
        pool_contr1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr1_mp")(conv_contr1)

        print("conv_contr1: {}".format(conv_contr1.shape))

        conv_contr2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr2_1")(pool_contr1)
        conv_contr2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr2_2")(conv_contr2)
        pool_contr2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr2_mp")(conv_contr2)

        print("conv_contr2: {}".format(conv_contr2.shape))

        conv_contr3 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr3_1")(pool_contr2)
        conv_contr3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr3_2")(conv_contr3)
        pool_contr3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  name="contr3_mp")(conv_contr3)

        print("conv_contr3: {}".format(conv_contr3.shape))

        # "Bottom" layer
        conv_bottom = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', name="bottom1")(pool_contr3)
        print("conv_bottom I: {}".format(conv_bottom.shape))
        conv_bottom = Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', padding='same', name="bottom2")(conv_bottom)
        print("conv_bottom II: {}".format(conv_bottom.shape))


        # Crop outputs of each contracting path "layer" for use in their corresponding expansive path "layer"
        # For brain MRA:
        # crop_up1 = Cropping3D(cropping=((12, 12), (12, 12), (12, 12)))(conv_contr1)
        # crop_up2 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv_contr2)
        # crop_up3 = Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv_contr3)

        # crop_up1 = Cropping3D(cropping=((12, 12), (12, 12), (12, 12)))(conv_contr1)
        # crop_up2 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv_contr2)
        # crop_up3 = Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv_contr3)

        crop_up1 = conv_contr1
        crop_up2 = conv_contr2
        crop_up3 = conv_contr3

        # Expansive path:
        scale_up3 = UpSampling3D(size=(2, 2, 2))(conv_bottom)
        print("scale_up3: {}".format(scale_up3.shape))
        merge_up3 =  concatenate([scale_up3, crop_up3], axis=4)
        print("merge_up3: {}".format(merge_up3.shape))
        conv_up3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge_up3)
        print("conv_up3 I: {}".format(conv_up3.shape))
        conv_up3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_up3)
        print("conv_up3 II: {}".format(conv_up3.shape))

        scale_up2 = UpSampling3D(size=(2, 2, 2))(conv_up3)
        print("scale_up2: {}".format(scale_up2.shape))
        merge_up2 =  concatenate([scale_up2, crop_up2], axis=4)
        print("merge_up2: {}".format(merge_up2.shape))
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge_up2)
        print("conv_up2 I: {}".format(conv_up2.shape))
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_up2)
        print("conv_up2 II: {}".format(conv_up2.shape))

        scale_up1 = UpSampling3D(size=(2, 2, 2))(conv_up2)
        print("scale_up1: {}".format(scale_up1.shape))
        merge_up1 =  concatenate([scale_up1, crop_up1], axis=4)
        print("merge_up1: {}".format(merge_up1.shape))
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge_up1)
        print("conv_up1 I: {}".format(conv_up1.shape))
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_up1)
        print("conv_up1 II: {}".format(conv_up1.shape))

        # Final 1x1 conv layer
        conv_final = Conv3D(filters=self._num_classes, kernel_size=(1, 1, 1))(conv_up1)
        print("final: {}".format(conv_final.shape))
        conv_final = Activation('softmax')(conv_final)

        self._model = Model(inputs=inputs, outputs=conv_final)

        return self._model

    def build_model(self):
        """
        Build the 3D U-Net architecture as defined by Cicek et al:
        https://arxiv.org/abs/1606.06650
        """
        self._title = "UNet3D_paper"

        # Set the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (self._num_slices, self._img_channels, self._img_height, self._img_height)
        else:
            input_shape = (self._img_height, self._img_width, self._num_slices, self._img_channels)

        inputs = Input(input_shape)
        print("\n", inputs)

        # Contracting path, from the paper:
        conv_contr1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr1_1")(inputs)
        conv_contr1 = BatchNormalization()(conv_contr1)
        conv_contr1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr1_2")(conv_contr1)
        conv_contr1 = BatchNormalization()(conv_contr1)
        pool_contr1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr1_mp")(conv_contr1)

        print("conv_contr1: {}".format(conv_contr1.shape))

        conv_contr2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr2_1")(pool_contr1)
        conv_contr2 = BatchNormalization()(conv_contr2)
        conv_contr2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr2_2")(conv_contr2)
        conv_contr2 = BatchNormalization()(conv_contr2)
        pool_contr2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr2_mp")(conv_contr2)

        print("conv_contr2: {}".format(conv_contr2.shape))

        conv_contr3 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr3_1")(pool_contr2)
        conv_contr3 = BatchNormalization()(conv_contr3)
        conv_contr3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', name="contr3_2")(conv_contr3)
        conv_contr3 = BatchNormalization()(conv_contr3)
        pool_contr3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  name="contr3_mp")(conv_contr3)

        print("conv_contr3: {}".format(conv_contr3.shape))

        # "Bottom" layer
        conv_bottom = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', name="bottom1")(pool_contr3)
        conv_bottom = BatchNormalization()(conv_bottom)
        print("conv_bottom I: {}".format(conv_bottom.shape))
        conv_bottom = Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', padding='same', name="bottom2")(conv_bottom)
        conv_bottom = BatchNormalization()(conv_bottom)
        print("conv_bottom II: {}".format(conv_bottom.shape))


        # Crop outputs of each contracting path "layer" for use in their corresponding expansive path "layer"
        # For brain MRA:
        # crop_up1 = Cropping3D(cropping=((12, 12), (12, 12), (12, 12)))(conv_contr1)
        # crop_up2 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv_contr2)
        # crop_up3 = Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv_contr3)

        # crop_up1 = Cropping3D(cropping=((12, 12), (12, 12), (12, 12)))(conv_contr1)
        # crop_up2 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv_contr2)
        # crop_up3 = Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv_contr3)

        crop_up1 = conv_contr1
        crop_up2 = conv_contr2
        crop_up3 = conv_contr3

        # Expansive path:
        scale_up3 = UpSampling3D(size=(2, 2, 2))(conv_bottom)
        print("scale_up3: {}".format(scale_up3.shape))
        merge_up3 =  concatenate([scale_up3, crop_up3], axis=4)
        print("merge_up3: {}".format(merge_up3.shape))
        conv_up3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge_up3)
        conv_up3 = BatchNormalization()(conv_up3)
        print("conv_up3 I: {}".format(conv_up3.shape))
        conv_up3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_up3)
        conv_up3 = BatchNormalization()(conv_up3)
        print("conv_up3 II: {}".format(conv_up3.shape))

        scale_up2 = UpSampling3D(size=(2, 2, 2))(conv_up3)
        print("scale_up2: {}".format(scale_up2.shape))
        merge_up2 =  concatenate([scale_up2, crop_up2], axis=4)
        print("merge_up2: {}".format(merge_up2.shape))
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge_up2)
        conv_up2 = BatchNormalization()(conv_up2)
        print("conv_up2 I: {}".format(conv_up2.shape))
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_up2)
        conv_up2 = BatchNormalization()(conv_up2)
        print("conv_up2 II: {}".format(conv_up2.shape))

        scale_up1 = UpSampling3D(size=(2, 2, 2))(conv_up2)
        print("scale_up1: {}".format(scale_up1.shape))
        merge_up1 =  concatenate([scale_up1, crop_up1], axis=4)
        print("merge_up1: {}".format(merge_up1.shape))
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge_up1)
        conv_up1 = BatchNormalization()(conv_up1)
        print("conv_up1 I: {}".format(conv_up1.shape))
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_up1)
        conv_up1 = BatchNormalization()(conv_up1)
        print("conv_up1 II: {}".format(conv_up1.shape))

        # Final 1x1 conv layer
        conv_final = Conv3D(filters=self._num_classes, kernel_size=(1, 1, 1))(conv_up1)
        print("final: {}".format(conv_final.shape))
        conv_final = Activation('softmax')(conv_final)

        self._model = Model(inputs=inputs, outputs=conv_final)

        return self._model
