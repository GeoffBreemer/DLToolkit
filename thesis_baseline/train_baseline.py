# TODO:
# - imbalanced classes:
#    https://github.com/keras-team/keras/issues/6261

import settings_baseline as settings

from dltoolkit.nn.cnn import AlexNetNN
from dltoolkit.io import HDF5Writer, HDF5Reader
from dltoolkit.utils.generic import list_images, model_architecture_to_file, model_summary_to_file
from dltoolkit.nn.segment import UNet_NN, FCN32_VGG16_NN
from dltoolkit.utils.visual import plot_training_history
from dltoolkit.utils.image import rgb_to_gray, normalise, clahe_equalization, adjust_gamma, standardise
from dltoolkit.preprocess import ResizePreprocessor

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, Conv2DTranspose, Activation, Reshape
from keras.models import Model
from keras.models import load_model

import cv2
import numpy as np
import os, progressbar, cv2, random, time
from PIL import Image                                   # for reading .gif images


def convert_img_to_pred(ground_truths, num_model_channels, verbose=False):
    """Convert ground truth *images* into the shape of the *predictions* produced by the U-Net (the opposite of
    convert_pred_to_img in drive_test.py)
    """
    start_time = time.time()

    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    ground_truths = np.reshape(ground_truths, (ground_truths.shape[0], img_height * img_width))
    new_masks = np.empty((ground_truths.shape[0], img_height * img_width, num_model_channels))

    for image in range(ground_truths.shape[0]):
        if verbose and image % 1000 == 0:
            print("{}/{}".format(image, ground_truths.shape[0]))

        for pix in range(img_height*img_width):
            if ground_truths[image, pix] == 0.:      # TODO: update for num_model_channels > 2
                new_masks[image, pix, 0] = 1.0
                new_masks[image, pix, 1] = 0.0
            else:
                new_masks[image, pix, 0] = 0.0
                new_masks[image, pix, 1] = 1.0

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return new_masks


def convert_to_hdf5(img_path, img_shape, img_exts, key, ext, is_mask=False):
    """
    Convert images present in `img_path` to HDF5 format. The HDF5 file is one sub folder up from where the
    images are located
    :param img_path: path to the folder containing images
    :param img_shape: shape of each image (width, height, # of channels)
    :return: full path to the generated HDF5 file
    """
    output_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path)) + ext
    imgs_list = sorted(list(list_images(basePath=img_path, validExts=img_exts)))

    # Prepare the HDF5 writer, which expects a label vector. Because this is a segmentation problem just pass None
    hdf5_writer = HDF5Writer((len(imgs_list), img_shape[0], img_shape[1], img_shape[2]), output_path,
                             feat_key=key,
                             label_key=None,
                             del_existing=True,
                             buf_size=len(imgs_list)
                             )

    # Loop through all images
    widgets = ["Creating HDF5 database ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imgs_list), widgets=widgets).start()
    for i, img in enumerate(imgs_list):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # Apply thresholding to ground truth masks only, not to images
        if is_mask:
            # print("before", np.min(image), np.max(image))
            _, image = cv2.threshold(image, settings.MASK_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
            # print("after", np.min(image), np.max(image))

        # Resize to AlexNet dimensions
        # image = cv2.resize(image, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_AREA)

        image = image.reshape((img_shape[0],
                               img_shape[1],
                               img_shape[2]))

        hdf5_writer.add([image], None)
        pbar.update(i)

    pbar.finish()
    hdf5_writer.close()

    return output_path


def perform_hdf5_conversion():
    """Convert the training and test images, ground truths and masks to HDF5 format. For the DRIVE data set the
    assumption is that filenames start with a number and that images/ground truths/masks share the same number
    """
    output_paths = []

    # Convert training images in each sub folder to a single HDF5 file
    # output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FLDR_IMAGES),
    #                                     (settings.IMG_RESIZE_DIM, settings.IMG_RESIZE_DIM, settings.IMG_CHANNELS),
    #                                     img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))
    output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))

    # Training ground truths
    # output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FLDR_GROUND_TRUTH),
    #                                     (settings.IMG_RESIZE_DIM_GT, settings.IMG_RESIZE_DIM_GT, settings.IMG_CHANNELS),
    #                                     img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))
    output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FLDR_GROUND_TRUTH),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        is_mask=True))

    # Do the same for the test images
    # output_paths.append(convert_to_hdf5(os.path.join(settings.TEST_PATH, settings.FLDR_IMAGES),
    #                                     (settings.IMG_RESIZE_DIM, settings.IMG_RESIZE_DIM, settings.IMG_CHANNELS),
    #                                     img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))
    output_paths.append(convert_to_hdf5(os.path.join(settings.TEST_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))

    return output_paths


def perform_image_preprocessing(image_path, key, is_training=True):
    """Perform image pre-processing, resulting pixel values are between 0.0 and 1.0"""
    imgs = HDF5Reader().load_hdf5(image_path, key).astype("uint8")

    # Standardise
    imgs = standardise(imgs)

    # Apply CLAHE equalization
    # imgs = clahe_equalization(imgs)

    # Apply gamma adjustment
    # imgs = adjust_gamma(imgs)

    # Cut off top and bottom pixel rows to convert images to squares when performing training
    # if is_training:
    #     imgs = crop_image(imgs, imgs.shape[1], imgs.shape[2])

    return imgs             # /255.0


def perform_groundtruth_preprocessing(ground_truth_path, key, is_training=True):
    """Perform ground truth image pre-processing, resulting pixel values are between 0 and 1"""
    imgs = HDF5Reader().load_hdf5(ground_truth_path, key).astype("uint8")

    # Cut off top and bottom pixel rows to convert images to squares
    # if is_training:
    #     imgs = crop_image(imgs, imgs.shape[1], imgs.shape[2])

    return imgs/255.0


if __name__ == "__main__":
    if  settings.IS_DEVELOPMENT:
        hdf5_paths = perform_hdf5_conversion()
    else:
        # During development avoid performing the HDF5 conversion for every run
        hdf5_paths = ["../data/MSC8002/training/images.h5",
                      "../data/MSC8002/training/groundtruths.h5",
                      "../data/MSC8002/test/images.hdf5"]

    # Read the training images
    train_imgs = perform_image_preprocessing(hdf5_paths[0], settings.HDF5_KEY)
    train_grndtr = perform_groundtruth_preprocessing(hdf5_paths[1], settings.HDF5_KEY)

    # Show one image and associated ground truth
    cv2.imshow("image", train_imgs[9])
    cv2.waitKey(0)
    cv2.imshow("ground truth", train_grndtr[9])
    cv2.waitKey(0)

    # Only train using a handful images
    PRED_IX = range(9, 19)
    train_imgs = train_imgs[[PRED_IX]]
    train_grndtr = train_grndtr[[PRED_IX]]

    # Shuffle
    idx = np.random.permutation(len(train_imgs))
    train_imgs, train_grndtr= train_imgs[idx], train_grndtr[idx]

    # CHECK DEZE:
    # https://github.com/heuritech/convnets-keras
    # https://devblogs.nvidia.com/image-segmentation-using-digits-5/
    # For the AlexNet, the images(for the mode without the heatmap) have to be of shape (227, 227).It is recommended to
    # resize the images with a size of (256, 256), and then do a crop of size (227, 227).The colors are in RGB order.

    # Instantiate the U-Net model
    # alex_seg = AlexNetNN(settings.NUM_CLASSES)
    # model, output_size = alex_seg.build_model_conv(settings.IMG_CHANNELS)
    # model.summary()

    unet = UNet_NN(img_height=settings.IMG_HEIGHT,
                   img_width=settings.IMG_WIDTH,
                   img_channels=settings.IMG_CHANNELS,
                   num_classes=settings.NUM_CLASSES)
    model = unet.build_model()

    print("model.output_shape: {}".format(model.output_shape))
    # print("       output_size: {}".format(output_size))

    # Prepare some path strings
    model_path = os.path.join(settings.MODEL_PATH, unet.title + "_ep{}.model".format(settings.TRN_NUM_EPOCH))
    csv_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_training_ep{}_bs{}.csv".format(
        settings.TRN_NUM_EPOCH,
        settings.TRN_BATCH_SIZE))
    summ_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_model_summary.txt")

    # Print the architecture to the console, a text file and an image
    model.summary()
    model_summary_to_file(model, summ_path)
    model_architecture_to_file(unet.model, settings.OUTPUT_PATH + unet.title + "_BRAIN_base_training")

    # Convert the random patches into the same shape as the predictions the U-net produces
    print("--- \nEncoding training ground truths")
    print(train_grndtr.shape)
    train_grndtr_ext_conv = convert_img_to_pred(train_grndtr, settings.NUM_CLASSES, settings.VERBOSE)
    # train_grndtr_ext_conv = train_grndtr
    print(train_grndtr_ext_conv.shape)

    # Train the model
    print("\n--- Start training")
    opt = Adam()
    # model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    # Prepare callbacks
    callbacks = [ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
                 EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode="auto"),
                 CSVLogger(csv_path, append=False),
                 # TensorBoard(settings.OUTPUT_PATH + "/tensorboard", histogram_freq=1, batch_size=1)
                 ]

    hist = model.fit(train_imgs, train_grndtr_ext_conv,
              epochs=settings.TRN_NUM_EPOCH,
              batch_size=settings.TRN_BATCH_SIZE,
              verbose=1,
              shuffle=True,
              validation_split=settings.TRN_TRAIN_VAL_SPLIT,
              callbacks=callbacks)

    print("\n--- Training complete")

    # Plot the training results - currently breaks if training stopped early
    plot_training_history(hist, settings.TRN_NUM_EPOCH, show=True, save_path=settings.OUTPUT_PATH + unet.title, time_stamp=True)

    # TODO: calculate performance metrics
