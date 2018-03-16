from dltoolkit.io import HDF5Writer, HDF5Reader
from dltoolkit.utils.image import rgb_to_gray, normalise, clahe_equalization, adjust_gamma

import numpy as np

def perform_image_preprocessing(image_path, key, is_training=True):
    """Perform image pre-processing, resulting pixel values are between 0 and 1"""
    imgs = HDF5Reader().load_hdf5(image_path, key).astype("uint8")

    # Normalise
    imgs = normalise(imgs)

    # Apply CLAHE equalization
    # imgs = clahe_equalization(imgs)

    # Apply gamma adjustment
    # imgs = adjust_gamma(imgs)

    # Cut off top and bottom pixel rows to convert images to squares when performing training
    # if is_training:
    #     imgs = crop_image(imgs, imgs.shape[1], imgs.shape[2])

    return imgs/255.0


def perform_groundtruth_preprocessing(ground_truth_path, key, is_training=True):
    """Perform ground truth image pre-processing, resulting pixel values are between 0 and 1"""
    imgs = HDF5Reader().load_hdf5(ground_truth_path, key).astype("uint8")

    # Cut off top and bottom pixel rows to convert images to squares
    # if is_training:
    #     imgs = crop_image(imgs, imgs.shape[1], imgs.shape[2])

    return imgs/255.0


def extend_images(imgs, patch_dim):
    """
    Extend images (assumed to be *square*) to the right and/or bottom with black pixels to ensure patches will cove
    the entire image as opposed to missing the bottom and/or right part of the image (because the image dimension
    divided by the patch dimension does not result in an integer)
    # TODO: needs to be able to deal with non-square images
    :param imgs: array of images to extend (images are assumed to be square)
    :param patch_dim: patch dimensions (patches are assumed to always be square)
    :return: array of extended images
    """

    img_dim = imgs.shape[1]
    new_img_dim = img_dim
    num_patches = int(img_dim / patch_dim) + 1      # number of patches across and down, total of num_patches**2 patches

    if (img_dim % patch_dim) == 0:
        # No changes required
        return imgs, new_img_dim, num_patches
    else:
        # Extension is required
        new_img_dim = int((img_dim / patch_dim) + 1) * patch_dim

    # Create a black image with the new size
    imgs_extended = np.zeros((imgs.shape[0], new_img_dim, new_img_dim, imgs.shape[3]))

    # Copy the original image, effectively extending the image to the right and bottom if required
    imgs_extended[:, 0:img_dim, 0:img_dim, :] = imgs[:, :, :, :]

    return imgs_extended, new_img_dim, num_patches