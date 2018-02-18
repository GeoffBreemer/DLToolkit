"""Use a trained U-Net model (using the -model parameter) to make predictions on the DRIVE test data"""
from settings import settings_drive as settings
from drive_utils import perform_groundtruth_preprocessing, perform_image_preprocessing,\
    save_image, crop_image, group_images

from dltoolkit.io import HDF5Reader

from keras.models import load_model

import os, cv2, time
import numpy as np
import argparse


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


def generate_ordered_patches(imgs, patch_dim, verbose=False):
    """Generate an array of patches for each image in an array of images"""
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
        # Create patches for each individual image covering the entire image
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


def convert_pred_to_img(pred, patch_dim, threshold=0.5, verbose=False):
    """Convert patch *predictions* to patch *images* (the opposite of convert_img_to_pred)"""
    start_time = time.time()

    pred_images = np.empty((pred.shape[0], pred.shape[1]))
    # pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1] * pred.shape[2]))

    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            if pred[i, pix, 1] > threshold:        # TODO for multiple classes > 2 use argmax
                pred_images[i, pix] = 1
            else:
                pred_images[i, pix] = 0

    pred_images = np.reshape(pred_images, (pred.shape[0], patch_dim, patch_dim, 1))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


def reconstruct_image(patches, img_dim, verbose=False):
    """Combine patch images into single images"""
    start_time = time.time()

    patch_dim = patches.shape[1]
    num_patches = int(img_dim/patch_dim)
    num_images = int(patches.shape[0] / (num_patches**2))

    # patches_reconstructed = np.empty((patches.shape[0], num_patches*patch_dim, num_patches*patch_dim, 1))
    patches_reconstructed = np.empty((num_images, num_patches*patch_dim, num_patches*patch_dim, 1))

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


def apply_masks(preds, masks):
    """Apply the masks to the predictions"""
    img_height = preds.shape[1]
    img_width = preds.shape[2]

    for i in range(preds.shape[0]):
        for x in range(img_width):
            for y in range(img_height):         # TODO: replace with cv2.bitwise_and
                if masks[i, y, x, 0] == 0:
                    preds[i, y, x, :] = 0.0


def load_masks(mask_path, key, patch_dim):
    """Load masks and crop and extend them like the images and ground truths were"""
    masks = HDF5Reader().load_hdf5(mask_path, key).astype("uint8")

    masks = crop_image(masks, masks.shape[1], masks.shape[2])
    masks, _, _ = extend_images(masks, patch_dim)

    return masks


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
    test_imgs = perform_image_preprocessing(os.path.join(settings.TEST_PATH,
                                                         settings.FOLDER_IMAGES + settings.HDF5_EXT),
                                            settings.HDF5_KEY, True)

    test_ground_truths = perform_groundtruth_preprocessing(os.path.join(settings.TEST_PATH,
                                                                        settings.FOLDER_MANUAL_1 + settings.HDF5_EXT),
                                                           settings.HDF5_KEY, True)
    # TODO: both should use is_training=False
    # print(test_imgs.shape)
    # test_imgs = test_imgs[[0],:,:,:]
    # test_ground_truths = test_ground_truths[[0],:,:,:]

    # Extend images and ground truths to ensure patches cover the entire image
    print("\n--- Extending images")
    test_imgs, new_img_dim, num_patches = extend_images(test_imgs, settings.PATCH_DIM)
    test_ground_truths, _, _ = extend_images(test_ground_truths, settings.PATCH_DIM)

    # Break up images into patches that will be provided to the U-Net for predicting
    print("\n--- Generating patches")
    patch_imgs = generate_ordered_patches(test_imgs, settings.PATCH_DIM, settings.VERBOSE)
    patch_ground_truths = generate_ordered_patches(test_ground_truths, settings.PATCH_DIM, settings.VERBOSE)

    # TODO: ORDERED - SELECT ONLY ONE/A FEW PATCHES
    # NUM_OVERFIT = 1
    # START_OVERFIT = 51
    # patch_imgs = patch_imgs[START_OVERFIT:START_OVERFIT+NUM_OVERFIT]
    # patch_ground_truths = patch_ground_truths[START_OVERFIT:START_OVERFIT+NUM_OVERFIT]
    # cv2.imshow("images", group_images(patch_imgs, NUM_OVERFIT))
    # cv2.waitKey(0)
    # cv2.imshow("ground truths", group_images(patch_ground_truths, NUM_OVERFIT))
    # cv2.waitKey(0)
    # TODO

    # Load the trained U-net model
    print("\n--- Loading trained model: {}".format(model_name_from_arguments()))
    model = load_model(model_name_from_arguments())

    # Make predictions on the patches
    print("\n--- Making predictions")
    predictions = model.predict(patch_imgs, batch_size=settings.BATCH_SIZE, verbose=2)

    # Convert patch predictions into patch images
    print("\n--- Reconstructing patch predictions to images")
    predictions_img = convert_pred_to_img(predictions,
                                      settings.PATCH_DIM,
                                      settings.PRED_THRESHOLD,
                                      settings.VERBOSE)

    print("GROUND TRUTH patches")
    print(patch_ground_truths.shape)
    # cv2.imshow("org gt", patch_ground_truths[0])
    # cv2.waitKey(0)
    tmp_img = group_images(patch_ground_truths, 5*num_patches)
    cv2.imshow("Ground truth", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "patches_ground_truth")

    print("ORIGINAL patches")
    print(patch_imgs[0].shape)
    tmp_img = group_images(patch_imgs, 5*num_patches)
    cv2.imshow("Original", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "patches_original")

    print("PREDICTED patches")
    print(predictions_img[0].shape)
    tmp_img = group_images(predictions_img, 5 * num_patches)
    cv2.imshow("Prediction", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "patches_predicted")

    # Reconstruct images from the predicted patch images
    print("\n--- Reconstructing images from patches")
    reconstructed = reconstruct_image(predictions_img, new_img_dim, settings.VERBOSE)

    tmp_img = group_images(reconstructed, 5)
    cv2.imshow("Reconstructed", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "images_reconstructed")

    exit()

    # Crop back to original resolution
    # TODO: requires updates to perform_image_preprocessing, perform_groundtruth_preprocessing and extend_images
    # test_imgs = test_imgs[:, :, 0:new_img_dim, 0:new_img_dim]
    # predictions = predictions[:, :, 0:new_img_dim, 0:new_img_dim]

    # Load and apply masks
    print("\n--- Loading masks")
    masks = load_masks(os.path.join(settings.TEST_PATH, settings.FOLDER_MASK + settings.HDF5_EXT),
                       settings.HDF5_KEY,
                       settings.PATCH_DIM)

    # Show the original, ground truth and prediction for one image
    print("\n--- Showing masked results")

    cv2.imshow("Mask", masks[0])
    cv2.waitKey(0)
    save_image(masks[0], settings.OUTPUT_PATH + "image_mask")

    # Load and apply masks
    print("\n--- Applying masks")
    apply_masks(reconstructed, masks)

    cv2.imshow("Masked reconstructed", reconstructed[0])
    cv2.waitKey(0)
    save_image(reconstructed[0], settings.OUTPUT_PATH + "image_reconstruct_masked")

    print("\n--- Predicting complete")
