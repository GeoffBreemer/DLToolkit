import settings_DECiSION as settings
from common_DECiSION import perform_image_preprocessing, perform_groundtruth_preprocessing, extend_images

from dltoolkit.io import HDF5Writer
from dltoolkit.utils.generic import list_images, model_architecture_to_file, model_summary_to_file
from dltoolkit.nn.segment import UNet_NN, FCN32_VGG16_NN
from dltoolkit.utils.visual import plot_training_history

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD, Adam

import numpy as np
import os, progressbar, cv2, random, time
from PIL import Image                                   # for reading .gif images


def convert_img_to_pred(ground_truths, num_model_channels, verbose=False):
    """Convert ground truth *images* into the shape of the *predictions* produced by the U-Net (the opposite of
    convert_pred_to_img_3D in drive_test.py)
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


def convert_to_hdf5(img_path, img_shape, img_exts, key, ext):
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
    output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))

    # Training ground truths
    output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FLDR_GROUND_TRUTH),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))


    # Do the same for the test images
    output_paths.append(convert_to_hdf5(os.path.join(settings.TEST_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))

    return output_paths


if __name__ == "__main__":
    if not settings.IS_DEVELOPMENT:
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
    cv2.imshow("image", train_imgs[150])
    cv2.waitKey(0)
    cv2.imshow("ground truth", train_grndtr[150])
    cv2.waitKey(0)

    # Extend images
    train_imgs_ext, _, _ = extend_images(train_imgs, settings.IMG_DIM_EXT)
    train_grndtr_ext, _, _ = extend_images(train_grndtr, settings.IMG_DIM_EXT)

    # train_imgs_ext = train_imgs
    # train_grndtr_ext = train_grndtr

    # Instantiate the U-Net model
    unet = UNet_NN(img_height=settings.IMG_DIM_EXT,
                   img_width=settings.IMG_DIM_EXT,
                   img_channels=settings.IMG_CHANNELS,
                   num_classes=settings.NUM_CLASSES
                   )
    model = unet.build_model_DRIVE()

    # Prepare some path strings
    model_path = os.path.join(settings.MODEL_PATH, unet.title + "_BRAIN_ep{}.model".format(settings.TRN_NUM_EPOCH))
    csv_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_BRAIN_training_ep{}_bs{}.csv".format(
        settings.TRN_NUM_EPOCH,
        settings.TRN_BATCH_SIZE))
    summ_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_BRAIN_model_summary.txt")

    # Print the architecture to the console, a text file and an image
    model.summary()
    model_summary_to_file(model, summ_path)
    model_architecture_to_file(unet.model, settings.OUTPUT_PATH + unet.title + "_BRAIN_training")

    # Convert the random patches into the same shape as the predictions the U-net produces
    print("--- \nEncoding training ground truths")
    train_grndtr_ext_conv = convert_img_to_pred(train_grndtr_ext, settings.NUM_CLASSES, settings.VERBOSE)

    # Train the model
    print("\n--- Start training")
    opt = Adam()
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    # Prepare callbacks
    callbacks = [ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
                 EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode="auto"),
                 CSVLogger(csv_path, append=False),
                 # TensorBoard(settings.OUTPUT_PATH + "/tensorboard", histogram_freq=1, batch_size=1)
                 ]

    hist = model.fit(train_imgs_ext, train_grndtr_ext_conv,
              epochs=settings.TRN_NUM_EPOCH,
              batch_size=settings.TRN_BATCH_SIZE,
              verbose=1,
              shuffle=True,
              validation_split=settings.TRN_TRAIN_VAL_SPLIT,
              callbacks=callbacks)

    print("\n--- Training complete")

    # Plot the training results - currently breaks if training stopped early
    plot_training_history(hist, settings.TRN_NUM_EPOCH, show=False, save_path=settings.OUTPUT_PATH + unet.title + "_BRAIN", time_stamp=True)

    # TODO: calculate performance metrics
