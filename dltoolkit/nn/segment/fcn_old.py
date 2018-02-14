"""FCN architecture based on Long et al., 2015 built using Keras

Based on: A Fully Convolutional Neural Network for Cardiac Segmentation in Short-Axis MRI
Available at: https://arxiv.org/abs/1604.00494
"""
from dltoolkit.nn.base_nn import BaseNN

from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Dropout, Lambda,\
    ZeroPadding2D, Conv2DTranspose, Input, Cropping2D, average
from keras.losses import kullback_leibler_divergence

# Constants
DROP_OUT_PERC = 0.5

# Custom loss functions:
#
# https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras
# https://keras.io/losses/

# From GitHub
# def dice_coef(y_true, y_pred, smooth=0.0):
#     """Average dice coefficient per batch"""
#     axes = (1, 2, 3)
#     intersection = K.sum(y_true * y_pred, axis=axes)
#     summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
#
#     return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


# From GitHub
# def dice_loss():
#     def dice(y_true, y_pred):
#         return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)
#
#     return dice


# def dice_coef(y_true, y_pred, smooth=0.0):
#     # y_pred = y_pred > 0.5
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#
# def dice_loss():
#     def dice(y_true, y_pred):
#         # return -dice_coef(y_true, y_pred, smooth=1e-2)
#         return 1.0 - dice_coef(y_true, y_pred, smooth=1e5)
#     return dice


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def jaccard_coef(y_true, y_pred, smooth=0.0):
    """Average jaccard coefficient per batch"""
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection

    return K.mean( (intersection + smooth) / (union + smooth), axis=0)


def crop(tensors):
    """List of 2 tensors, the second tensor having larger spatial dimensions"""
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h // 2, crop_h // 2 + rem_h)
    crop_w_dims = (crop_w // 2, crop_w // 2 + rem_w)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])

    return cropped


def mvn(tensor):
    """Performs per-channel spatial mean-variance normalization"""
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) // (std + epsilon)

    return mvn


class FCN_NN(BaseNN):
    # Input shape dimensions
    _img_width = -1
    _img_height = -1
    _img_channels = -1

    # Number of classes to classify
    _num_classes = -1

    def __init__(self, img_width, img_height, img_channels, num_classes):
        self._img_height = img_height
        self._img_width = img_width
        self._img_channels = img_channels
        self._num_classes = num_classes
        self.optimizer = None
        self.loss = None

    def build_model(self):
        # Set the input shape
        input_shape = (self._img_height, self._img_width, self._img_channels)
        if K.image_data_format() == "channels_first":
            input_shape = (self._img_channels, self._img_height, self._img_width)

        if self._num_classes == 2:
            self._num_classes = 1
            self.loss = dice_coef_loss
            activation = 'sigmoid'
        else:
            self.loss = 'categorical_crossentropy'
            activation = 'softmax'

        # Common Conv2D parameters
        conv2d_args = dict(
            kernel_size=3,
            strides=1,
            activation='relu',
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
        )

        data = Input(shape=input_shape, dtype='float', name='data')
        mvn0 = Lambda(mvn, name='mvn0')(data)
        pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

        conv1 = Conv2D(filters=64, name='conv1', **conv2d_args)(pad)
        mvn1 = Lambda(mvn, name='mvn1')(conv1)

        conv2 = Conv2D(filters=64, name='conv2', **conv2d_args)(mvn1)
        mvn2 = Lambda(mvn, name='mvn2')(conv2)

        conv3 = Conv2D(filters=64, name='conv3', **conv2d_args)(mvn2)
        mvn3 = Lambda(mvn, name='mvn3')(conv3)
        pool1 = MaxPooling2D(pool_size=3, strides=2,
                             padding='valid', name='pool1')(mvn3)

        conv4 = Conv2D(filters=128, name='conv4', **conv2d_args)(pool1)
        mvn4 = Lambda(mvn, name='mvn4')(conv4)

        conv5 = Conv2D(filters=128, name='conv5', **conv2d_args)(mvn4)
        mvn5 = Lambda(mvn, name='mvn5')(conv5)

        conv6 = Conv2D(filters=128, name='conv6', **conv2d_args)(mvn5)
        mvn6 = Lambda(mvn, name='mvn6')(conv6)

        conv7 = Conv2D(filters=128, name='conv7', **conv2d_args)(mvn6)
        mvn7 = Lambda(mvn, name='mvn7')(conv7)
        pool2 = MaxPooling2D(pool_size=3, strides=2,
                             padding='valid', name='pool2')(mvn7)

        conv8 = Conv2D(filters=256, name='conv8', **conv2d_args)(pool2)
        mvn8 = Lambda(mvn, name='mvn8')(conv8)

        conv9 = Conv2D(filters=256, name='conv9', **conv2d_args)(mvn8)
        mvn9 = Lambda(mvn, name='mvn9')(conv9)

        conv10 = Conv2D(filters=256, name='conv10', **conv2d_args)(mvn9)
        mvn10 = Lambda(mvn, name='mvn10')(conv10)

        conv11 = Conv2D(filters=256, name='conv11', **conv2d_args)(mvn10)
        mvn11 = Lambda(mvn, name='mvn11')(conv11)
        pool3 = MaxPooling2D(pool_size=3, strides=2,
                             padding='valid', name='pool3')(mvn11)
        drop1 = Dropout(rate=DROP_OUT_PERC, name='drop1')(pool3)

        conv12 = Conv2D(filters=512, name='conv12', **conv2d_args)(drop1)
        mvn12 = Lambda(mvn, name='mvn12')(conv12)

        conv13 = Conv2D(filters=512, name='conv13', **conv2d_args)(mvn12)
        mvn13 = Lambda(mvn, name='mvn13')(conv13)

        conv14 = Conv2D(filters=512, name='conv14', **conv2d_args)(mvn13)
        mvn14 = Lambda(mvn, name='mvn14')(conv14)

        conv15 = Conv2D(filters=512, name='conv15', **conv2d_args)(mvn14)
        mvn15 = Lambda(mvn, name='mvn15')(conv15)
        drop2 = Dropout(rate=DROP_OUT_PERC, name='drop2')(mvn15)

        score_conv15 = Conv2D(filters=self._num_classes, kernel_size=1,
                              strides=1, activation=None, padding='valid',
                              kernel_initializer='glorot_uniform', use_bias=True,
                              name='score_conv15')(drop2)
        upsample1 = Conv2DTranspose(filters=self._num_classes, kernel_size=3,
                                    strides=2, activation=None, padding='valid',
                                    kernel_initializer='glorot_uniform', use_bias=False,
                                    name='upsample1')(score_conv15)
        score_conv11 = Conv2D(filters=self._num_classes, kernel_size=1,
                              strides=1, activation=None, padding='valid',
                              kernel_initializer='glorot_uniform', use_bias=True,
                              name='score_conv11')(mvn11)
        crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
        fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

        upsample2 = Conv2DTranspose(filters=self._num_classes, kernel_size=3,
                                    strides=2, activation=None, padding='valid',
                                    kernel_initializer='glorot_uniform', use_bias=False,
                                    name='upsample2')(fuse_scores1)
        score_conv7 = Conv2D(filters=self._num_classes, kernel_size=1,
                             strides=1, activation=None, padding='valid',
                             kernel_initializer='glorot_uniform', use_bias=True,
                             name='score_conv7')(mvn7)
        crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
        fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

        upsample3 = Conv2DTranspose(filters=self._num_classes, kernel_size=3,
                                    strides=2, activation=None, padding='valid',
                                    kernel_initializer='glorot_uniform', use_bias=False,
                                    name='upsample3')(fuse_scores2)
        crop3 = Lambda(crop, name='crop3')([data, upsample3])
        predictions = Conv2D(filters=self._num_classes, kernel_size=1,
                             strides=1, activation=activation, padding='valid',
                             kernel_initializer='glorot_uniform', use_bias=True,
                             name='predictions')(crop3)

        self._model = Model(inputs=data, outputs=predictions)
        self.optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

        return self._model

    def __str__(self):
        return self._title + " architecture, input shape: {} x {} x {}, {} classes".format(self._img_width,
                                                                                           self._img_height,
                                                                                           self._img_channels,
                                                                                           self._num_classes)


if __name__ == "__main__":
    import numpy as np
    a = np.random.random((420, 100))
    b = np.random.random((420, 100))
    res = dice_coef(a, b)
    print(res)
