"""Keras loss and metric functions for use with model.compile(metrics=[..., "..."], loss="...")
Helpful resources;

Keras discussions:
https://github.com/keras-team/keras/issues/3653
https://github.com/keras-team/keras/issues/6261
https://github.com/keras-team/keras/issues/5335

Focal loss:
https://arxiv.org/abs/1708.02002
"""

import keras.backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred):
    """Dice loss coefficient metric"""
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """Dice loss"""
    return -dice_coef(y_true, y_pred)


def focal_loss(target, output, gamma=2):
    """Focal loss"""
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)

    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output), axis=-1)


def weighted_pixelwise_crossentropy_loss(class_weights):
    """Weighted loss cross entropy loss, call with a weight array, e.g. [1, 10]"""
    def loss(y_true, y_pred):
        epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)

        # Clip very small and very large predictions
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Return the *weighted* cross entropy
        return -tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))

    return loss


##############################################################
# NOT TESTED NOT TESTED NOT TESTED NOT TESTED NOT TESTED
##############################################################
def weighted_dice_coefficient(y_true, y_pred, axis=(-4, -3, -2), smooth=0.00001):
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))

def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)
