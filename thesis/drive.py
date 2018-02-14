from settings import settings_drive as settings

from dltoolkit.io import HDF5Writer, HDF5Reader
from dltoolkit.utils.generic import list_images
from dltoolkit.utils.image import rgb_to_gray, normalise, clahe_equalization, adjust_gamma
from dltoolkit.nn.segment import UNet_NN

import numpy as np
import os, progressbar, cv2

from PIL import Image                                   # for reading .gif images


def _convert_to_hdf5(img_path, img_shape, exts):
    """
    Convert images present in `img_path` to HDF5 format. The HDF5 file is created in the same folder as
    where the folder containing the images is located (i.e. one level up from the images)
    :param img_path: path to the folder containing images
    :param img_shape: shape of each image (width, height, # of channels)
    :return: full path to the HDF5 file
    """
    output_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path)) + settings.HDF5_EXT
    imgs = list(list_images(basePath=img_path, validExts=exts))

    # Prepare the HDF5 writer, a label vector is not available because this is a segmentation problem
    hdf5_writer = HDF5Writer((len(imgs), img_shape[0], img_shape[1], img_shape[2]), output_path,
                             feat_key=settings.HDF5_KEY,
                             label_key=None,
                             del_existing=True)

    # Loop through all images
    widgets = ["Creating HDF5 database ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imgs), widgets=widgets).start()
    for i, img in enumerate(imgs):
        if exts == ".gif":
            # Ground truth and masks are .gif files
            image = np.asarray(Image.open(img).convert("L"))
            image = image.reshape((settings.IMG_HEIGHT,
                                   settings.IMG_WIDTH,
                                   settings.MODEL_CHANNELS))
        else:
            # Actual images are .tiff files
            image = cv2.imread(img)

        hdf5_writer.add(image, None)
        pbar.update(i)

    pbar.finish()
    hdf5_writer.close()

    return output_path


def perform_hdf5_conversion():
    output_paths = []

    # Convert training images in each sub folder to a single HDF5 file
    output_paths.append(_convert_to_hdf5(os.path.join(settings.PATH_TRAINING, settings.FOLDER_IMAGES),
                                         (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_TIF),
                                         exts=".tif"))
    output_paths.append(_convert_to_hdf5(os.path.join(settings.PATH_TRAINING, settings.FOLDER_MANUAL_1),
                                         (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_GIF),
                                         exts=".gif"))
    output_paths.append(_convert_to_hdf5(os.path.join(settings.PATH_TRAINING, settings.FOLDER_MASK),
                                         (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_GIF),
                                         exts=".gif"))

    # Do the same for the test images
    output_paths.append(_convert_to_hdf5(os.path.join(settings.PATH_TEST, settings.FOLDER_IMAGES),
                                         (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_TIF),
                                         exts=".tif"))
    output_paths.append(_convert_to_hdf5(os.path.join(settings.PATH_TEST, settings.FOLDER_MANUAL_1),
                                         (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_GIF),
                                         exts=".gif"))
    output_paths.append(_convert_to_hdf5(os.path.join(settings.PATH_TEST, settings.FOLDER_MASK),
                                         (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_GIF),
                                         exts=".gif"))

    return output_paths


def crop_image(imgs):
    """Cut off the top and bottom pixel rows so that image height and width are the same"""
    new_top = int((settings.IMG_HEIGHT-settings.IMG_WIDTH)/2)
    new_bottom = settings.IMG_HEIGHT-round((settings.IMG_HEIGHT-settings.IMG_WIDTH)/2)

    return imgs[:, new_top:new_bottom, :, :]


def perform_image_preprocessing(training_image_path):
    """Perform training image pre-processing, pixels are between 0 and 1"""
    imgs = HDF5Reader().load_hdf5(training_image_path, settings.HDF5_KEY).astype("uint8")

    # Convert RGB to gray scale
    imgs = rgb_to_gray(imgs)

    # Normalise
    imgs = normalise(imgs)

    # Apply CLAHE equalization
    imgs = clahe_equalization(imgs)

    # Apply gamma adjustment
    imgs = adjust_gamma(imgs)

    # Cut off top and bottom pixel rows to convert images to squares
    imgs = crop_image(imgs)

    return imgs/255.0


def perform_groundtruth_preprocessing(training_mask_path):
    """Perform ground truth image pre-processing, pixels are between 0 and 1"""
    imgs = HDF5Reader().load_hdf5(training_mask_path, settings.HDF5_KEY).astype("uint8")

    # Cut off top and bottom pixel rows to convert images to squares
    imgs = crop_image(imgs)

    return imgs/255.0


if __name__ == "__main__":
    # Convert images to HDF5 format
    # hdf5_paths = perform_hdf5_conversion()

    # Hard code paths to HDF5 files during development
    hdf5_paths = ['../data/DRIVE/training/images.hdf5',
                    '../data/DRIVE/training/1st_manual.hdf5',
                    '../data/DRIVE/training/mask.hdf5',
                    '../data/DRIVE/test/images.hdf5',
                    '../data/DRIVE/test/1st_manual.hdf5',
                    '../data/DRIVE/test/mask.hdf5']

    # Perform training image and ground truth pre-processing. All images are 565 x 565 gray scale after this
    training_imgs = perform_image_preprocessing(hdf5_paths[0])
    training_ground_truths = perform_groundtruth_preprocessing(hdf5_paths[1])

    # Instantiate the U-Net model
    # unet = UNet_NN(img_height=settings.IMG_HEIGHT,
    #                img_width=settings.IMG_WIDTH,
    #                img_channels=settings.MODEL_CHANNELS)

    unet = UNet_NN(img_height=572,
                   img_width=572,
                   img_channels=1)

    model = unet.build_model()
    model.summary()

    # model2 = unet.get_unet()
    # model2.summary()