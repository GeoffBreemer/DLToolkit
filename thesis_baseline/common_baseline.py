import settings_baseline as settings
from dltoolkit.io import HDF5Reader
from dltoolkit.utils.image import normalise, standardise

from keras import backend as K

import tensorflow as tf

import cv2
import numpy as np
import time


def convert_img_to_pred_4D(ground_truths, num_model_channels, verbose=False):
    # from (-1, height, width, 1) to (-1, height, width, num_classes)
    # last axis: 0 = background, 1 = blood vessel
    start_time = time.time()

    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    # ground_truths = np.reshape(ground_truths, (ground_truths.shape[0], img_height, img_width))
    new_masks = np.empty((ground_truths.shape[0], img_height, img_width, num_model_channels))
    print("new_masks: {}".format(new_masks.shape))

    for image in range(ground_truths.shape[0]):
        if image != 0 and verbose and image % 1000 == 0:
            print("Processed {}/{}".format(image, ground_truths.shape[0]))

        for pix in range(img_height):
            for pix_w in range(img_width):
                if ground_truths[image, pix, pix_w] == 0:      # TODO: update for num_model_channels > 2
                    new_masks[image, pix, pix_w, 0] = 1.0
                    new_masks[image, pix, pix_w, 1] = 0.0
                else:
                    new_masks[image, pix, pix_w, 0] = 0.0
                    new_masks[image, pix, pix_w, 1] = 1.0

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return new_masks


def convert_img_to_pred_3D(ground_truths, num_model_channels, verbose=False):
    # from (-1, height, width, 1) to (-1, height * width, num_classes)
    # last axis: 0 = background, 1 = blood vessel
    start_time = time.time()

    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    print("gt from: {}".format(ground_truths.shape))
    ground_truths = np.reshape(ground_truths, (ground_truths.shape[0], img_height * img_width))
    print("  gt to: {} ".format(ground_truths.shape))

    new_masks = np.empty((ground_truths.shape[0], img_height * img_width, num_model_channels))

    for image in range(ground_truths.shape[0]):
        if verbose and image % 1000 == 0:
            print("{}/{}".format(image, ground_truths.shape[0]))

        for pix in range(img_height*img_width):
            if ground_truths[image, pix] == 0:      # TODO: update for num_model_channels > 2
                new_masks[image, pix, 0] = 1.0
                new_masks[image, pix, 1] = 0.0
            else:
                new_masks[image, pix, 0] = 0.0
                new_masks[image, pix, 1] = 1.0

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return new_masks


def convert_pred_to_img_4D(pred, patch_dim, threshold=0.5, verbose=False):
    # from (-1, height, width, num_classes) to (-1, height, width, 1)
    start_time = time.time()

    pred_images = np.empty((pred.shape[0], pred.shape[1], pred.shape[2]), dtype=np.uint8)
    # pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1] * pred.shape[2]))

    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            for pix_w in range(pred.shape[2]):
                if pred[i, pix, pix_w, 0] > threshold:        # TODO for multiple classes > 2 use argmax
                    # print("from {} to {}".format(pred[i, pix, 1], 1))
                    pred_images[i, pix, pix_w] = 0
                else:
                    # print("from {} to {}".format(pred[i, pix, 1], 0))
                    pred_images[i, pix, pix_w] = 255

    pred_images = np.reshape(pred_images, (pred.shape[0], patch_dim, patch_dim, 1))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


def convert_pred_to_img_3D(pred, patch_dim, threshold=0.5, verbose=False):
    # from (-1, height * width, num_classes) to (-1, height, width, 1)
    start_time = time.time()

    pred_images = np.empty((pred.shape[0], pred.shape[1]), dtype=np.uint8)
    # pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1] * pred.shape[2]))

    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            if pred[i, pix, 0] > threshold:        # TODO for multiple classes > 2 use argmax
                # print("from {} to {}".format(pred[i, pix, 1], 1))
                pred_images[i, pix] = 0
            else:
                # print("from {} to {}".format(pred[i, pix, 1], 0))
                pred_images[i, pix] = 255

    pred_images = np.reshape(pred_images, (pred.shape[0], patch_dim, patch_dim, 1))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


# OpenCV expects pixel intensities:
# 0.0 - 1.0 if dtype is uint8
# 0 - 255 if dtype is float32

def perform_image_preprocessing(image_path, key, is_training=True):
    """Perform image pre-processing, resulting pixel values are between 0.0 and 1.0"""
    print("Loading image HDF5: {}".format(image_path))
    imgs = HDF5Reader().load_hdf5(image_path, key)#.astype("uint8")
    print("With dtype = {}".format(imgs.dtype))

    # Standardise
    imgs = standardise(imgs)
    print("dtype after std preprocessing = {}\n".format(imgs.dtype))

    return imgs


def perform_groundtruth_preprocessing(ground_truth_path, key, is_training=True):
    """Perform ground truth image pre-processing, resulting pixel values are between 0 and 255"""
    print("Loading ground truth HDF5: {}".format(ground_truth_path))
    imgs = HDF5Reader().load_hdf5(ground_truth_path, key).astype("uint8")
    print("With dtype = {}\n".format(imgs.dtype))

    return imgs


# def weighted_pixelwise_crossentropy(class_weights):
#     def loss(y_true, y_pred):
#         epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
#         return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))
#
#     return loss


def dice_coef2(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def class_weighted_pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.8, 0.2]
    return -tf.reduce_sum(target * weights * tf.log(output))

