"""Implementation of a 3D U-Net using Keras"""
from keras.layers import concatenate, Activation, Input, Conv3D, MaxPooling3D, UpSampling3D,\
    BatchNormalization, Deconvolution3D, Concatenate
from keras.models import Model
from dltoolkit.nn.base_nn import BaseNN


class UNet_3D_NN(BaseNN):
    _title = "UNet3D"

    def __init__(self, img_height, img_width, num_slices, img_channels, num_classes):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._num_slices = num_slices
        self._num_classes = num_classes
        self._input = Input((self._img_height, self._img_width, self._num_slices, self._img_channels))

    def build_model_alt(self, num_layers, n_base_filters, deconvolution, use_bn=False):
        """
        Create a 3D Unet model with a variable number of layers and initial number of filters
        :param num_layers: number of layers (i.e. number of skip connections + 1)
        :param n_base_filters: number of filters to use in the first conv layer
        :param deconvolution: True for Deconvolution3D, False for UpSampling3D
        :param use_bn: True to use BatchNormalisation, False otherwise
        :return: Keras model
        """
        POOL_SIZE = (2, 2, 2)
        POOL_STRIDE = (2, 2, 2)
        CONV_KERNEL = (3, 3, 3)
        CONV_STRIDE = (1, 1, 1)
        DECONV_KERNEL = (2, 2, 2)
        DECONV_STRIDE = (2, 2, 2)
        UPSAMPLE_SIZE = (2, 2, 2)
        FEATURE_AXIS = -1

        self._title = "UNet3D_{}layer_{}flt_deconv{}".format(num_layers, n_base_filters, int(deconvolution))
        self._title += "_BN" if use_bn else ""

        inputs = self._input
        current_layer = inputs
        layers = list()

        # Contracting path
        for layer_ix in range(num_layers):
            # Two conv layers, note the difference in the number of filters
            contr_conv1 = Conv3D(filters=n_base_filters*(2**layer_ix),
                                 kernel_size=CONV_KERNEL,
                                 strides=CONV_STRIDE,
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer="he_normal")(current_layer)
            if use_bn: contr_conv1 = BatchNormalization(axis=FEATURE_AXIS)(contr_conv1)

            contr_conv2 = Conv3D(filters=n_base_filters*(2**layer_ix)*2,
                                 kernel_size=CONV_KERNEL,
                                 strides=CONV_STRIDE,
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer="he_normal")(contr_conv1)
            if use_bn: contr_conv2 = BatchNormalization(axis=FEATURE_AXIS)(contr_conv2)

            # Do not include maxpooling in the final bottom layer
            if layer_ix < num_layers - 1:
                current_layer = MaxPooling3D(pool_size=POOL_SIZE,
                                             strides=POOL_STRIDE,
                                             padding="same")(contr_conv2)
                layers.append([contr_conv1, contr_conv2, current_layer])
            else:
                current_layer = contr_conv2
                layers.append([contr_conv1, contr_conv2])

        # Expanding path
        for layer_ix in range(num_layers-2, -1, -1):
            if deconvolution:
                exp_deconv = Deconvolution3D(filters=current_layer._keras_shape[-1],
                                             kernel_size=DECONV_KERNEL,
                                             strides=DECONV_STRIDE)(current_layer)
            else:
                exp_deconv = UpSampling3D(size=UPSAMPLE_SIZE)(current_layer)

            concat_layer = Concatenate(axis=FEATURE_AXIS)([exp_deconv, layers[layer_ix][1]])
            current_layer = Conv3D(filters=layers[layer_ix][1]._keras_shape[FEATURE_AXIS],
                                   kernel_size=CONV_KERNEL,
                                   strides=CONV_STRIDE,
                                   padding="same",
                                   activation="relu",
                                   kernel_initializer="he_normal")(concat_layer)
            if use_bn: current_layer = BatchNormalization(axis=FEATURE_AXIS)(current_layer)

            current_layer = Conv3D(filters=layers[layer_ix][1]._keras_shape[FEATURE_AXIS],
                                   kernel_size=CONV_KERNEL,
                                   strides=CONV_STRIDE,
                                   padding="same",
                                   activation="relu",
                                   kernel_initializer="he_normal")(current_layer)
            if use_bn: current_layer = BatchNormalization(axis=FEATURE_AXIS)(current_layer)

        act = Conv3D(self._num_classes, (1, 1, 1), activation="softmax", padding="same",
                                   kernel_initializer="he_normal")(current_layer)

        self._model = Model(inputs=[inputs], outputs=[act])

        return self._model

    def build_model(self, use_bn=False):
        """
        Build the 4 layer 3D U-Net architecture as defined by Cicek et al:
        https://arxiv.org/abs/1606.06650
        """
        self._title = "UNet3D_brain_4layer"
        self._title += "_BN" if use_bn else ""

        # Set the input shape
        input_shape = (self._img_height, self._img_width, self._num_slices, self._img_channels)
        inputs = Input(input_shape)

        # Contracting path, from the paper:
        conv_contr1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr1_1")(inputs)
        conv_contr1 = BatchNormalization()(conv_contr1) if use_bn else conv_contr1
        conv_contr1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr1_2")(conv_contr1)
        conv_contr1 = BatchNormalization()(conv_contr1) if use_bn else conv_contr1
        pool_contr1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr1_mp")(conv_contr1)

        conv_contr2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr2_1")(pool_contr1)
        conv_contr2 = BatchNormalization()(conv_contr2) if use_bn else conv_contr2
        conv_contr2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr2_2")(conv_contr2)
        conv_contr2 = BatchNormalization()(conv_contr2) if use_bn else conv_contr2
        pool_contr2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr2_mp")(conv_contr2)

        conv_contr3 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr3_1")(pool_contr2)
        conv_contr3 = BatchNormalization()(conv_contr3) if use_bn else conv_contr3
        conv_contr3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr3_2")(conv_contr3)
        conv_contr3 = BatchNormalization()(conv_contr3) if use_bn else conv_contr3
        pool_contr3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  name="contr3_mp")(conv_contr3)

        # "Bottom" layer
        conv_bottom = Conv3D(filters=256, kernel_size=(3, 3, 3), activation="relu", padding="same", name="bottom1")(pool_contr3)
        conv_bottom = BatchNormalization()(conv_bottom) if use_bn else conv_bottom
        conv_bottom = Conv3D(filters=512, kernel_size=(3, 3, 3), activation="relu", padding="same", name="bottom2")(conv_bottom)
        conv_bottom = BatchNormalization()(conv_bottom) if use_bn else conv_bottom

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
        merge_up3 = concatenate([scale_up3, crop_up3], axis=4)
        conv_up3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation="relu", padding="same")(merge_up3)
        conv_up3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation="relu", padding="same")(conv_up3)

        scale_up2 = UpSampling3D(size=(2, 2, 2))(conv_up3)
        merge_up2 = concatenate([scale_up2, crop_up2], axis=4)
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same")(merge_up2)
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same")(conv_up2)

        scale_up1 = UpSampling3D(size=(2, 2, 2))(conv_up2)
        merge_up1 = concatenate([scale_up1, crop_up1], axis=4)
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same")(merge_up1)
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same")(conv_up1)

        # Final 1x1 conv layer
        conv_final = Conv3D(filters=self._num_classes, kernel_size=(1, 1, 1))(conv_up1)
        conv_final = Activation("softmax")(conv_final)

        self._model = Model(inputs=inputs, outputs=conv_final)

        return self._model

    def build_model_3lyr(self, use_bn=False):
        """
        Build a 3 layer version of Cicek et al's 3D U-Net:
        """
        self._title = "UNet3D_brain_3layer"
        self._title += "_BN" if use_bn else ""

        input_shape = (self._img_height, self._img_width, self._num_slices, self._img_channels)
        inputs = Input(input_shape)

        # Contracting path, from the paper:
        # Layer 1
        conv_contr1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr1_1")(inputs)
        conv_contr1 = BatchNormalization()(conv_contr1) if use_bn else conv_contr1

        conv_contr1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr1_2")(conv_contr1)
        conv_contr1 = BatchNormalization()(conv_contr1) if use_bn else conv_contr1

        pool_contr1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr1_mp")(conv_contr1)

        # Layer 2
        conv_contr2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr2_1")(pool_contr1)
        conv_contr2 = BatchNormalization()(conv_contr2) if use_bn else conv_contr2

        conv_contr2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr2_2")(conv_contr2)
        conv_contr2 = BatchNormalization()(conv_contr2) if use_bn else conv_contr2

        pool_contr2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr2_mp")(conv_contr2)

        # "Bottom" layer 3
        conv_bottom = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", name="bottom1")(pool_contr2)
        conv_bottom = BatchNormalization()(conv_bottom) if use_bn else conv_bottom

        conv_bottom = Conv3D(filters=256, kernel_size=(3, 3, 3), activation="relu", padding="same", name="bottom2")(conv_bottom)
        conv_bottom = BatchNormalization()(conv_bottom) if use_bn else conv_bottom

        crop_up1 = conv_contr1
        crop_up2 = conv_contr2

        # Expansive path:
        scale_up2 = UpSampling3D(size=(2, 2, 2))(conv_bottom)
        print("scale_up2: ", scale_up2.shape)
        print("crop_up2: ", crop_up2.shape)

        merge_up2 =  concatenate([scale_up2, crop_up2], axis=4)
        print("merge_up2: ", merge_up2.shape)

        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same")(merge_up2)
        conv_up2 = BatchNormalization(axis=-1)(conv_up2) if use_bn else conv_up2
        print("conv_up2: ", conv_up2.shape)
        conv_up2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same")(conv_up2)
        conv_up2 = BatchNormalization(axis=-1)(conv_up2) if use_bn else conv_up2
        print("conv_up2: ", conv_up2.shape)

        scale_up1 = UpSampling3D(size=(2, 2, 2))(conv_up2)
        print("scale_up1: ", scale_up1.shape)
        print("crop_up1: ", crop_up1.shape)

        merge_up1 =  concatenate([scale_up1, crop_up1], axis=4)
        print("merge_up1: ", merge_up1.shape)

        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same")(merge_up1)
        conv_up1 = BatchNormalization(axis=-1)(conv_up1) if use_bn else conv_up1
        print("conv_up1: ", conv_up1.shape)
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same")(conv_up1)
        conv_up1 = BatchNormalization(axis=-1)(conv_up1) if use_bn else conv_up1
        print("conv_up1: ", conv_up1.shape)

        # Final 1x1 conv layer
        conv_final = Conv3D(filters=self._num_classes, kernel_size=(1, 1, 1))(conv_up1)
        print("conv_final: ", conv_final.shape)

        conv_final = Activation("softmax")(conv_final)
        print("act: ", conv_final.shape)

        self._model = Model(inputs=inputs, outputs=conv_final)

        return self._model

    def build_model_2lyr(self, use_bn=False):
        """
        Build a 2 layer version of Cicek et al's 3D U-Net:
        """
        self._title = "UNet3D_brain_2layer"
        self._title += "_BN" if use_bn else ""

        input_shape = (self._img_height, self._img_width, self._num_slices, self._img_channels)
        inputs = Input(input_shape)

        # Contracting path
        # Layer 1
        conv_contr1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr1_1")(inputs)
        print("conv_contr1 conv1: ", conv_contr1.shape)
        conv_contr1 = BatchNormalization(axis=-1)(conv_contr1) if use_bn else conv_contr1

        conv_contr1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same", name="contr1_2")(conv_contr1)
        print("conv_contr1 conv2: ", conv_contr1.shape)
        conv_contr1 = BatchNormalization(axis=-1)(conv_contr1) if use_bn else conv_contr1

        pool_contr1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="contr1_mp")(conv_contr1)
        print("pool_contr1: ", pool_contr1 .shape)

        # "Bottom" layer 2
        conv_bottom = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same", name="bottom1")(pool_contr1)
        print("conv_bottom conv1: ", conv_bottom.shape)
        conv_bottom = BatchNormalization(axis=-1)(conv_bottom) if use_bn else conv_bottom

        conv_bottom = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", name="bottom2")(conv_bottom)
        print("conv_bottom conv2: ", conv_bottom.shape)
        conv_bottom = BatchNormalization(axis=-1)(conv_bottom) if use_bn else conv_bottom

        crop_up1 = conv_contr1

        # Expansive path:
        scale_up1 = UpSampling3D(size=(2, 2, 2))(conv_bottom)
        print("scale_up1: ", scale_up1.shape)
        print("crop_up1: ", crop_up1.shape)
        merge_up1 =  concatenate([scale_up1, crop_up1], axis=4)
        print("merge_up1: ", merge_up1.shape)
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same")(merge_up1)
        print("conv_up1 conv1: ", conv_up1.shape)
        conv_up1 = BatchNormalization(axis=-1)(conv_up1) if use_bn else conv_up1
        conv_up1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same")(conv_up1)
        print("conv_up1 conv1: ", conv_up1.shape)
        conv_up1 = BatchNormalization(axis=-1)(conv_up1) if use_bn else conv_up1

        # Final 1x1 conv layer
        conv_final = Conv3D(filters=self._num_classes, kernel_size=(1, 1, 1))(conv_up1)
        print("conv_final conv1: ", conv_final.shape)
        conv_final = Activation("softmax")(conv_final)
        print("conv_final act: ", conv_final.shape)

        self._model = Model(inputs=inputs, outputs=conv_final)

        return self._model
