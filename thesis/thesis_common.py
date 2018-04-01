"""Image handling and conversion methods"""
from dltoolkit.iomisc import HDF5Reader, HDF5Writer
from dltoolkit.utils.image import standardise_single, standardise, mean_subtraction
from dltoolkit.utils.generic import list_images

import numpy as np
import cv2
import time, os, progressbar, argparse
import matplotlib.pyplot as plt


# 3D U-Net conversions
def convert_to_hdf5_3D(img_path, img_shape, img_exts, key, ext, settings, is_mask=False):
    """Convert images present in `img_path` to HDF5 format. The HDF5 file is saved one sub folder up from where the
    images are located. Masks are binary tresholded to be 0 for background pixels and 255 for blood vessels. Images
    (slices) are expected to be stored in subfolders (volumes), one for each patient.
    :param img_path: path to the folder containing images
    :param img_shape: shape of each image (width, height, # of channels)
    :param img_ext: image extension, e.g. ".jpg"
    :param key: HDF5 data set key
    :param settings: settings
    :param is_mask: True when converting masks/ground truths, False when converting images
    :return: full path to the generated HDF5 file, class_weights (masks/ground truths only)
    """
    # Path to the HDF5 file
    output_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path)) + ext

    # Create a list of paths to the individual patient folders
    patient_folders = sorted([os.path.join(img_path, e.name) for e in os.scandir(img_path) if e.is_dir()])

    # Prepare the HDF5 writer, which expects a label vector. Because this is a segmentation problem just pass None
    hdf5_writer = HDF5Writer((len(patient_folders), settings.NUM_SLICES_TOTAL) + img_shape, output_path,
                             feat_key=key,
                             label_key=None,
                             del_existing=True,
                             buf_size=len(patient_folders),
                             dtype_feat="f" if not is_mask else "i8")
    classcounts = [0] * settings.NUM_CLASSES

    # Loop through all images
    widgets = ["Creating HDF5 database ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(patient_folders), widgets=widgets).start()

    # Loop through each patient subfolder
    for patient_ix, p_folder in enumerate(patient_folders):
        imgs_list = sorted(list(list_images(basePath=p_folder, validExts=img_exts)))
        # imgs = np.zeros((settings.NUM_SLICES_TOTAL, img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
        imgs = np.zeros((settings.NUM_SLICES_TOTAL, img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8)

        # Read each slice in the current patient's folder
        for slice_ix, slice_img in enumerate(imgs_list):
            image = cv2.imread(slice_img, cv2.IMREAD_GRAYSCALE)

            # Apply any preprocessing
            if is_mask:
                # Apply binary thresholding to ground truth masks
                _, image = cv2.threshold(image, settings.MASK_BINARY_THRESHOLD, settings.MASK_BLOODVESSEL, cv2.THRESH_BINARY)

                # Count the number of class occurrences in the ground truth image
                for ix, cl in enumerate([settings.MASK_BACKGROUND, settings.MASK_BLOODVESSEL]):
                    classcounts[ix] += len(np.where(image == cl)[0])
            # else:
            # Standardise the images
            #     image = mean_subtraction(image)
            #     image = image/255.
                # image = standardise_single(image)
                # pass

            # Reshape from (height, width) to (height, width, 1)
            image = image.reshape((img_shape[0], img_shape[1], img_shape[2]))
            imgs[slice_ix] = image

        ########################################
        # if not is_mask:
        #     print("prior group std during CONV:{} - {}".format(imgs.shape, imgs.dtype))
        #     imgs = standardise(imgs)
        #     print("after group std during CONV:{} - {}".format(imgs.shape, imgs.dtype))
        ########################################

        # Write all slices for the current patient
        hdf5_writer.add([imgs], None)
        pbar.update(patient_ix)

    if is_mask:
        total = sum(classcounts)
        for ix in range(settings.NUM_CLASSES):
            classcounts[ix] = int(total / classcounts[ix])

    pbar.finish()
    hdf5_writer.close()

    if is_mask:
        return output_path, classcounts
    else:
        return output_path


def convert_img_to_pred_3D(ground_truths, num_classes, verbose=False):
    """Convert an array of grayscale images with shape (-1, height, width, slices, 1) to an array of the same length with
    shape (-1, height, width, slices, num_classes). Does not generalise to more than two classes, and requires the ground
    truth image to only contain 0 (first class) or 255 (second class)
    Helpful: https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
    """
    start_time = time.time()

    # Convert 0 to 0 and 255 to 1, then perform one-hot encoding and squeeze the single-dimension
    tmp_truths = ground_truths/255
    new_masks = (np.arange(num_classes) == tmp_truths[..., None]).astype(np.uint8)
    new_masks = np.squeeze(new_masks, axis=4)

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return new_masks


def convert_pred_to_img_3D(pred, verbose=False):
    """Convert 3D UNet predictions to images, changing the shape from (-1, height, width, slices, num_classes) to
    (-1, slices, height, width, 1). The assumption is that only two classes are used and that they are 0 (background)
    and 255 (blood vessels). This function will not generalize to more classes and/or different class labels in its
    current state.
    """
    start_time = time.time()

    # print("pred shape: {}".format(pred.shape))
    # ix = 120
    # print(pred[0, ix:(ix+11), 100, 0, :])

    # Determine the class label for each pixel for all images
    pred_images = (np.argmax(pred, axis=-1)*255).astype(np.uint8)

    # print("pred images shape 1: {}".format(pred_images.shape))
    # print(pred_images[0, ix:(ix+11), 100, 0])

    # Add a dimension for the color channel
    pred_images = np.reshape(pred_images, tuple(pred_images.shape[0:4]) + (1,))
    # print("pred images shape 2: {}".format(pred_images.shape))

    # Permute the dimensions
    pred_images = np.transpose(pred_images, axes=(0, 3, 1, 2, 4))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


# U-Net conversions
def create_hdf5_db(imgs_list, dn_name, img_path, img_shape, key, ext, settings, is_mask=False):
    """
    Create a HDF5 file using a list of paths to individual images to be written to the data set
    :param imgs_list: list of image paths
    :param dn_name: becomes part of the HDF5 file name
    :param img_path: path to the location of the `images` and `groundtruths` subfolders
    :param img_shape: shape of the images being written to the data set
    :param key: key to use for the data set
    :param ext: extension of the HDF5 file name
    :param settings: holds settings
    :param is_mask: True if masks are being written, False if not
    :return: the full path to the HDF5 file
    """
    # Construct the name of the database
    tmp_name = dn_name + ("_masks" if is_mask else "_imgs")
    output_path = os.path.join(os.path.dirname(img_path), tmp_name) + ext
    print(output_path)

    # Prepare the HDF5 writer, which expects a label vector. Because this is a segmentation problem just pass None
    # hdf5_writer = HDF5Writer((len(imgs_list), img_shape[0], img_shape[1], img_shape[2]), output_path,
    hdf5_writer = HDF5Writer(((len(imgs_list),) + img_shape),
                             output_path=output_path,
                             feat_key=key,
                             label_key=None,
                             del_existing=True,
                             buf_size=len(imgs_list),
                             dtype_feat=np.float32 if not is_mask else np.uint8
                             )
    # Prepare for CLAHE histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))

    # Loop through all images
    widgets = ["Creating HDF5 database ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imgs_list), widgets=widgets).start()
    for i, img in enumerate(imgs_list):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # Crop to the region of interest
        image = image[settings.IMG_CROP_HEIGHT:image.shape[0] - settings.IMG_CROP_HEIGHT,
                settings.IMG_CROP_WIDTH:image.shape[1] - settings.IMG_CROP_WIDTH]

        # Apply pre-processing
        if is_mask:
            # Apply binary thresholding to ground truth masks
            _, image = cv2.threshold(image, settings.MASK_BINARY_THRESHOLD, settings.MASK_BLOODVESSEL,
                                     cv2.THRESH_BINARY)

            # Convert to the format produced by the model
            # image = convert_img_to_pred(np.array([image]), settings, settings.VERBOSE)
        else:
            # Apply preprocessing to images (not to ground truth masks)
            # Apply CLAHE histogram equalization
            image = clahe.apply(image)

            # Normalise between -0.5 and 0.5
            # image = (image / 255.0 - 0.5).astype(np.float32)

            # Standardise
            image = standardise_single(image)

        # Reshape from (height, width) to (height, width, 1)
        image = image.reshape((img_shape[0], img_shape[1], img_shape[2]))

        hdf5_writer.add([image], None)
        pbar.update(i)

    pbar.finish()
    hdf5_writer.close()

    return output_path


def convert_img_to_pred(ground_truths, num_classes, verbose=False):
    """Convert an array of grayscale images with shape (-1, height, width, 1) to an array of the same length with
    shape (-1, height, width, num_classes).
    :param ground_truths: array of grayscale images, pixel values are integers 0 (background) or 255 (blood vessels)
    :param settings:
    :param verbose: True if additional information is to be printed to the console during training
    :return: one-hot encoded version of the image
    """

    start_time = time.time()

    # Convert 0 to 0 and 255 to 1, then perform one-hot encoding and squeeze the single-dimension
    tmp_truths = ground_truths/255
    new_masks = (np.arange(num_classes) == tmp_truths[..., None]).astype(np.uint8)
    new_masks = np.squeeze(new_masks, axis=3)

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return new_masks


    start_time = time.time()

    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    new_masks = np.empty((ground_truths.shape[0], img_height, img_width, settings.NUM_CLASSES), dtype=np.uint8)

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


def convert_pred_to_img(pred, verbose=False):
    """Convert U-Net predictions from (-1, height, width, num_classes) to (-1, height, width, 1)"""
    start_time = time.time()

    # print("pred shape: {}".format(pred.shape))
    # ix = 120
    # print(pred[0, ix:(ix+11), 100, 0, :])

    # Determine the class label for each pixel for all images
    pred_images = (np.argmax(pred, axis=-1)*255).astype(np.uint8)

    # print("pred images shape 1: {}".format(pred_images.shape))
    # print(pred_images[0, ix:(ix+11), 100, 0])

    # Add a dimension for the color channel
    pred_images = np.reshape(pred_images, tuple(pred_images.shape[0:3]) + (1,))
    # print("pred images shape 2: {}".format(pred_images.shape))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


def convert_img_to_pred_flatten(ground_truths, settings, verbose=False):
    """Similar to convert_img_to_pred, but converts from (-1, height, width, 1) to (-1, height * width, num_classes)"""
    start_time = time.time()

    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    print("gt from: {}".format(ground_truths.shape))
    ground_truths = np.reshape(ground_truths, (ground_truths.shape[0], img_height * img_width))
    print("  gt to: {} ".format(ground_truths.shape))

    new_masks = np.empty((ground_truths.shape[0], img_height * img_width, settings.NUM_CLASSES), dtype=np.uint8)

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


def convert_pred_to_img_flatten(pred, settings, threshold=0.5, verbose=False):
    """Convert U-Net predictions from (-1, height * width, num_classes) to (-1, height, width, 1)"""
    start_time = time.time()

    pred_images = np.empty((pred.shape[0], pred.shape[1]), dtype=np.uint8)
    # pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1] * pred.shape[2]))

    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            if pred[i, pix, settings.ONEHOT_BLOODVESSEL] > threshold:
                # print("from {} to {}".format(pred[i, pix, 1], 1))
                pred_images[i, pix] = settings.MASK_BLOODVESSEL
            else:
                # print("from {} to {}".format(pred[i, pix, 1], 0))
                pred_images[i, pix] = settings.MASK_BACKGROUND

    pred_images = np.reshape(pred_images, (pred.shape[0], settings.IMG_HEIGHT, settings.IMG_WIDTH, 1))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


# Image loading and preprocessing
def read_preprocess_image(image_path, key, is_3D=False):
    """Perform image pre-processing, resulting pixel values are between 0.0 and 1.0"""
    imgs = HDF5Reader().load_hdf5(image_path, key)
    print("Loading image HDF5: {} with dtype = {}\n".format(image_path, imgs.dtype))

    # Permute array dimensions for the 3D U-Net model so that the shape becomes: (-1, height, width, slices, channels)
    if is_3D:
        # Standardise
        # print("prior group std during READ:{} - {}".format(imgs.shape, imgs.dtype))
        imgs = standardise(imgs)
        # imgs = mean_subtraction(imgs)
        # print("after group std during READ:{} - {}".format(imgs.shape, imgs.dtype))

        imgs = np.transpose(imgs, axes=(0, 2, 3, 1, 4))

    return imgs


def read_preprocess_groundtruth(ground_truth_path, key, is_3D=False):
    """Perform ground truth image pre-processing, resulting pixel values are between 0 and 255"""
    imgs = HDF5Reader().load_hdf5(ground_truth_path, key).astype("uint8")
    print("Loading ground truth HDF5: {} with dtype = {}\n".format(ground_truth_path, imgs.dtype))

    # Permute array dimensions for the 3D U-Net model so that the shape becomes: (-1, height, width, slices, channels)
    if is_3D:
        imgs = np.transpose(imgs, axes=(0, 2, 3, 1, 4))

    return imgs


# Visualisation
def group_images(imgs, num_per_row, empty_color=255, show=False, save_path=None):
    """Combines an array of images into a single image using a grid with num_per_row columns, the number of rows is
    calculated using the number of images in the array and the number of requested columns. Grid cells without an
    image are replaced with an empty image using a specified color.
    :param imgs: numpy array of images , shape: (-1, height, width, channels)
    :param num_per_row: number of images shown in each row
    :param empty_color: color to use for empty grid cells, e.g. 255 for white (grayscale images)
    :param show: True if the resulting image should be displayed on screen, False otherwise
    :param save_path: full path for the image, None otherwise
    :return: resulting grid image
    """
    all_rows= []
    img_height = imgs.shape[1]
    img_width = imgs.shape[2]
    img_channels = imgs.shape[3]

    num_rows = (imgs.shape[0] // num_per_row) + (1 if imgs.shape[0] % num_per_row else 0)
    for i in range(num_rows):
        # Add the first image to the current row
        row = imgs[i * num_per_row]

        if i == (num_rows-1):
            # Ensure the last row does not use more images than available in the array
            remaining = num_rows * num_per_row - len(imgs)
            rng = range(i * num_per_row + 1, i * num_per_row + num_per_row - remaining)
        else:
            rng = range(i * num_per_row + 1, i * num_per_row + num_per_row)

        # Concatenate the remaining images to the current row
        for k in rng:
            row = np.concatenate((row, imgs[k]), axis=1)

        if i == (num_rows-1):
            # For the last row use white images for any empty cells
            row = np.concatenate((row, np.full((img_height, remaining*img_width, img_channels),
                                               empty_color,
                                               dtype=imgs[0].dtype)),
                                 axis=1)

        all_rows.append(row)

    # Create the grid image by concatenating all rows
    final_image = all_rows[0]
    for i in range(1, len(all_rows)):
        final_image = np.concatenate((final_image, all_rows[i]),axis=0)

    # Plot the image
    plt.figure(figsize=(20.48, 15.36))
    plt.axis('off')
    plt.imshow(final_image[:, :, 0], cmap="gray")

    # Save the plot to a file if desired
    if save_path is not None:
        save_path = save_path + ".png"
        plt.savefig(save_path, dpi=100)

    # Show the plot if desired
    if show:
        plt.show()

    plt.close()

    return final_image


def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.grid(False)
    plt.title(title)
    plt.show()


# Parameter parsing
def model_name_from_arguments():
    """Return the full path of the model to be used for making predictions"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, nargs='?',
                    const=True, required=True, help="Set to the full path of the trained model to use")
    args = vars(ap.parse_args())

    return args["model"]