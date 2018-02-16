"""Use the trained U-Net model to make predictions on the DRIVE test data"""
from settings import settings_drive as settings
from drive_utils import perform_groundtruth_preprocessing, perform_image_preprocessing, save_image

from keras.models import load_model

import os, cv2, time
import numpy as np
import argparse


def extend_images(imgs, patch_dim):
    """
    Extend images (assumed to be *square*) to the right and/or bottom with black pixels to ensure patches will cove
    the entire image as opposed to missing the bottom and/or right part of the image (because the image dimension
    divided by the patch dimension does not result in an integer)
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


def generate_ordered_patches(imgs, patch_dim, verbose=False):
    """Generate an array of patches for an array of images"""
    start_time = time.time()

    img_dim = imgs.shape[1]
    num_patches = int(img_dim/patch_dim)

    if verbose:
        print("# patches {}, pixels remaining: {}".format(num_patches, img_dim%patch_dim))

    num_patches_total = (num_patches*num_patches)*imgs.shape[0]

    patches = np.empty((num_patches_total, patch_dim, patch_dim, imgs.shape[3]))

    # Loop over all images
    total_patch_count = 0
    for i in range(imgs.shape[0]):
        # Create patches for each individual image
        for h in range(num_patches):
            for w in range(num_patches):
                patch = imgs[i,                                 # image
                        h*patch_dim:(h*patch_dim)+patch_dim,    # height
                        w*patch_dim:(w*patch_dim)+patch_dim,    # width
                        :]                                      # color channel
                patches[total_patch_count] = patch
                total_patch_count +=1

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return patches


def convert_pred_to_img(pred, pred_num_channels, patch_dim, verbose=False):
    """Convert patch *predictions* to patch *images* (the opposite of convert_img_to_pred)"""
    start_time = time.time()

    pred_images = np.empty((pred.shape[0], pred.shape[1] * pred.shape[2]))
    pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3]))

    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            if pred[i, pix, 1] >= 0.5:
                pred_images[i, pix] = 1
            else:
                pred_images[i, pix] = 0

    pred_images = np.reshape(pred_images, (pred.shape[0], patch_dim, patch_dim, pred_num_channels))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


def reconstruct_image(patches, img_dim, verbose=False):
    """Combine patch images into single images"""
    start_time = time.time()

    patch_dim = patches.shape[1]
    num_patches = int(img_dim / patch_dim)

    patches_reconstructed = np.empty((patches.shape[0], num_patches*patch_dim, num_patches*patch_dim, 1))

    current_image = 0
    current_patch = 0

    # Loop over all patches
    while current_patch < patches.shape[0]:
        image = np.empty((num_patches*patch_dim, num_patches*patch_dim, 1))
        for p in range(num_patches):
            for w in range(num_patches):
                image[p*patch_dim:(p*patch_dim)+patch_dim, w*patch_dim:(w*patch_dim)+patch_dim, :]= patches[current_patch]
                current_patch+=1

        patches_reconstructed[current_image]=image
        current_image+=1

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return patches_reconstructed


def model_name_from_arguments():
    """Return the full path of the model to be used for making predictions"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, nargs='?',
                    const=True, required=True, help="Set to the full path of the trained model to use")
    args = vars(ap.parse_args())

    return args["model"]


if __name__ == "__main__":
    # Load and preprocess the test and ground truth images
    print("--- Pre-processing test images")
    test_imgs = perform_image_preprocessing(os.path.join(settings.TEST_PATH, settings.FOLDER_IMAGES + settings.HDF5_EXT),
                                            settings.HDF5_KEY)
    test_ground_truths = perform_groundtruth_preprocessing(os.path.join(settings.TEST_PATH, settings.FOLDER_MANUAL_1+ settings.HDF5_EXT),
                                                           settings.HDF5_KEY)

    # Extend images and ground truths to ensure patches cover the entire image
    print("\n--- Extending patches")
    test_imgs, new_img_dim, patches_dim = extend_images(test_imgs, settings.PATCH_DIM)
    test_ground_truths, _, _ = extend_images(test_ground_truths, settings.PATCH_DIM)

    # Break up images into patches that will be provided to the U-Net for predicting
    print("\n--- Generating patches")
    patch_imgs = generate_ordered_patches(test_imgs, settings.PATCH_DIM, settings.VERBOSE)
    patch_ground_truths = generate_ordered_patches(test_ground_truths, settings.PATCH_DIM, settings.VERBOSE)

    # Load the trained U-net model
    print("\n--- Loading trained model")
    model = load_model(model_name_from_arguments())

    # Make predictions on the patches
    print("\n--- Making predictions")
    predictions = model.predict(patch_imgs, batch_size=settings.BATCH_SIZE, verbose=2)

    # Convert patch predictions into patch images
    print("\n--- Reconstructing patches")
    predictions = convert_pred_to_img(predictions, settings.PATCH_CHANNELS, settings.PATCH_DIM, settings.VERBOSE)

    # Reconstruct the images from the patch images
    print("\n--- Reconstructing images from patches")
    predictions = reconstruct_image(predictions, new_img_dim, settings.VERBOSE)

    # Crop any extended pixels added by extend_images()
    # TODO

    IMG_INDEX = 0
    cv2.imshow("lala", test_imgs[IMG_INDEX])
    cv2.waitKey(0)
    save_image(test_imgs[IMG_INDEX], settings.OUTPUT_PATH + "image_org")

    cv2.imshow("lala", test_ground_truths[IMG_INDEX])
    cv2.waitKey(0)
    save_image(test_ground_truths[IMG_INDEX], settings.OUTPUT_PATH + "image_groundtruth")

    cv2.imshow("lala", predictions[IMG_INDEX])
    cv2.waitKey(0)
    save_image(predictions[IMG_INDEX], settings.OUTPUT_PATH + "image_pred")
