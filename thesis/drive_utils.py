"""Common functions for drive_train.py and drive_test.py"""
from dltoolkit.io import HDF5Reader
from dltoolkit.utils.image import rgb_to_gray, normalise, clahe_equalization, adjust_gamma

import numpy as np
from PIL import Image

def crop_image(imgs, img_height, img_width):
    """Cut off the top and bottom pixel rows so that image height and width are the same"""
    new_top = int((img_height-img_width)/2)
    new_bottom = img_height-round((img_height-img_width)/2)

    return imgs[:, new_top:new_bottom, :, :]


def perform_image_preprocessing(image_path, key):
    """Perform image pre-processing, resulting pixel values are between 0 and 1"""
    imgs = HDF5Reader().load_hdf5(image_path, key).astype("uint8")

    # Convert RGB to gray scale
    imgs = rgb_to_gray(imgs)

    # Normalise
    imgs = normalise(imgs)

    # Apply CLAHE equalization
    imgs = clahe_equalization(imgs)

    # Apply gamma adjustment
    imgs = adjust_gamma(imgs)

    # Cut off top and bottom pixel rows to convert images to squares
    imgs = crop_image(imgs, imgs.shape[1], imgs.shape[2])

    return imgs/255.0


def perform_groundtruth_preprocessing(ground_truth_path, key):
    """Perform ground truth image pre-processing, resulting pixel values are between 0 and 1"""
    imgs = HDF5Reader().load_hdf5(ground_truth_path, key).astype("uint8")

    # Cut off top and bottom pixel rows to convert images to squares
    imgs = crop_image(imgs, imgs.shape[1], imgs.shape[2])

    return imgs/255.0


def save_image(img, filename):
    print(img.shape)

    if img.shape[2]==1:
        img = np.reshape(img, (img.shape[0], img.shape[1]))

    print(img.shape)

    if np.max(img)>1:
        img = Image.fromarray(img.astype(np.uint8))
    else:
        img = Image.fromarray((img*255).astype(np.uint8))

    print(filename + ".png")
    img.save(filename + ".png")

    return img