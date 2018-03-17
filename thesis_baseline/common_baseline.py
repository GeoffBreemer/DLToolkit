import settings_baseline as settings
from dltoolkit.io import HDF5Reader
from dltoolkit.utils.image import normalise, standardise

from keras import backend as K
from keras.utils import to_categorical

import tensorflow as tf

import numpy as np
import time


# Note - OpenCV expects pixel intensities:
#
# 0.0 - 1.0 if dtype is uint8
# 0 - 255 if dtype is float32
#
# otherwise it will not show images properly.


def convert_img_to_pred_4D(ground_truths, num_classes, verbose=False):
    """
    Convert an array of grayscale images with shape (-1, height, width, 1) to an array of the same length with
    shape (-1, height, width, num_classes).
    :param ground_truths: array of grayscale images, pixel values are integers 0 (background) or 255 (blood vessels)
    :param num_classes: the number of classes, only accepts values of 2
    :param verbose: True if additional information is to be printed to the console during training
    :return: one-hot encoded version of the image
    """
    start_time = time.time()

    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    new_masks = np.empty((ground_truths.shape[0], img_height, img_width, num_classes), dtype=np.uint8 )
    print("new_masks type = {}".format(new_masks.dtype))

    for image in range(ground_truths.shape[0]):
        if image != 0 and verbose and image % 1000 == 0:
            print("Processed {}/{}".format(image, ground_truths.shape[0]))

        for pix_h in range(img_height):
            for pix_w in range(img_width):
                if ground_truths[image, pix_h, pix_w] == settings.MASK_BACKGROUND:
                    new_masks[image, pix_h, pix_w, settings.ONEHOT_BACKGROUND] = 1
                    new_masks[image, pix_h, pix_w, settings.ONEHOT_BLOODVESSEL] = 0
                else:
                    new_masks[image, pix_h, pix_w, settings.ONEHOT_BACKGROUND] = 0
                    new_masks[image, pix_h, pix_w, settings.ONEHOT_BLOODVESSEL] = 1

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return new_masks


def convert_img_to_pred_3D(ground_truths, num_classes, verbose=False):
    # from (-1, height, width, 1) to (-1, height * width, num_classes)
    # last axis: 0 = background, 1 = blood vessel
    start_time = time.time()

    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    print("gt from: {}".format(ground_truths.shape))
    ground_truths = np.reshape(ground_truths, (ground_truths.shape[0], img_height * img_width))
    print("  gt to: {} ".format(ground_truths.shape))

    new_masks = np.empty((ground_truths.shape[0], img_height * img_width, num_classes), dtype=np.uint8)

    for image in range(ground_truths.shape[0]):
        if verbose and image % 1000 == 0:
            print("{}/{}".format(image, ground_truths.shape[0]))

        for pix in range(img_height*img_width):
            if ground_truths[image, pix] == settings.MASK_BACKGROUND:      # TODO: update for num_model_channels > 2
                new_masks[image, pix, settings.ONEHOT_BACKGROUND] = 1
                new_masks[image, pix, settings.ONEHOT_BLOODVESSEL] = 0
            else:
                new_masks[image, pix, settings.ONEHOT_BACKGROUND] = 0
                new_masks[image, pix, settings.ONEHOT_BLOODVESSEL] = 1

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
                if pred[i, pix, pix_w, settings.ONEHOT_BLOODVESSEL] > threshold:        # TODO for multiple classes > 2 use argmax
                    # print("from {} to {}".format(pred[i, pix, 1], 1))
                    pred_images[i, pix, pix_w] = settings.MASK_BLOODVESSEL
                else:
                    # print("from {} to {}".format(pred[i, pix, 1], 0))
                    pred_images[i, pix, pix_w] = settings.MASK_BACKGROUND

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
            if pred[i, pix, settings.ONEHOT_BLOODVESSEL] > threshold:        # TODO for multiple classes > 2 use argmax
                # print("from {} to {}".format(pred[i, pix, 1], 1))
                pred_images[i, pix] = settings.MASK_BLOODVESSEL
            else:
                # print("from {} to {}".format(pred[i, pix, 1], 0))
                pred_images[i, pix] = settings.MASK_BACKGROUND

    pred_images = np.reshape(pred_images, (pred.shape[0], patch_dim, patch_dim, 1))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


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


def group_images(imgs, num_per_row):
    """Attempts to put an array of images into a single image, provided the number of images can be divided by
    the number of images desired per row
    """
    all_rows = []

    for i in range(int(imgs.shape[0] / num_per_row)):
        # Add the first image to the current row
        row = imgs[i * num_per_row]

        # Concatenate the remaining images to the current row
        for k in range(i * num_per_row + 1, i * num_per_row + num_per_row):
            row = np.concatenate((row, imgs[k]), axis=1)

        all_rows.append(row)

    # Take the first row and concatenate the remaining ones
    final_image = all_rows[0]
    for i in range(1, len(all_rows)):
        final_image = np.concatenate((final_image,all_rows[i]),axis=0)

    return final_image


# Metric functions for use with model.compile(metrics=[..., "..."])


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Loss functions for use with model.compile(loss="...")


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
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))

    return loss