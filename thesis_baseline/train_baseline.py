# TODO:
# - imbalanced classes:
#    ----------> focal loss PLUS benjamin-robbins: https://github.com/keras-team/keras/issues/6261
#    https://github.com/keras-team/keras/issues/8308
#    ----------> https://github.com/keras-team/keras/issues/2115
#    https://github.com/keras-team/keras/issues/5116
#    https://github.com/keras-team/keras/issues/6538#issuecomment-302964746
#    --> https://github.com/keras-team/keras/issues/3653
#    https://stackoverflow.com/questions/43033436/how-to-do-point-wise-categorical-crossentropy-loss-in-keras
#    https://stackoverflow.com/questions/43968028/how-to-use-weighted-categorical-crossentropy-on-fcn-u-net-in-keras?rq=1
#    https://stackoverflow.com/questions/46504371/how-to-do-weight-imbalanced-classes-for-cross-entropy-loss-in-keras
#    GOOD --> https://github.com/keras-team/keras/issues/3653
#    GOOD --> https://github.com/keras-team/keras/issues/6538
#
#
#   Loss functions:
#   https://datascience.stackexchange.com/questions/23968/99-validation-accuracy-but-0-prediction-results-unet-architecture
#   https://github.com/keras-team/keras/issues/2115
#   --> https://github.com/keras-team/keras/issues/2115
#   https://github.com/keras-team/keras/issues/9395
#
#
#
#   Class weights
#   https://github.com/keras-team/keras/issues/5116
#
#
#   Other approaches:
#   https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/22951
#
#
#
#
#
# AlexNet
# CHECK DEZE:
# https://github.com/heuritech/convnets-keras
# https://devblogs.nvidia.com/image-segmentation-using-digits-5/
# For the AlexNet, the images(for the mode without the heatmap) have to be of shape (227, 227).It is recommended to
# resize the images with a size of (256, 256), and then do a crop of size (227, 227).The colors are in RGB order.
#

# - cannot do pixel level softmax:
#    https://stackoverflow.com/questions/42118821/cross-entropy-loss-for-semantic-segmentation-keras?noredirect=1&lq=1

RANDOM_STATE = 42
from numpy.random import seed
seed(RANDOM_STATE)

from tensorflow import set_random_seed
set_random_seed(RANDOM_STATE)

import random
random.seed = RANDOM_STATE

import settings_baseline as settings

from dltoolkit.io import HDF5Writer, HDF5Reader
from dltoolkit.utils.generic import list_images, model_architecture_to_file, model_summary_to_file
from dltoolkit.nn.segment import UNet_NN, FCN32_VGG16_NN
from dltoolkit.utils.visual import plot_training_history

from common_baseline import perform_image_preprocessing, perform_groundtruth_preprocessing,\
    convert_img_to_pred_4D, convert_pred_to_img_4D,\
    convert_img_to_pred_3D, convert_pred_to_img_3D,\
    dice_coef, dice_coef_loss, focal_loss, weighted_pixelwise_crossentropy_loss

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras import backend as K

import numpy as np
import os, progressbar, cv2


def convert_to_hdf5(img_path, img_shape, img_exts, key, ext, is_mask=False):
    """
    Convert images present in `img_path` to HDF5 format. The HDF5 file is one sub folder up from where the
    images are located. Masks are binary tresholded to be 0 for background pixels and 255 for blood vessels.
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
                             buf_size=len(imgs_list),
                             dtype_feat="f" if not is_mask else "i8"
                             )

    # Loop through all images
    widgets = ["Creating HDF5 database ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imgs_list), widgets=widgets).start()
    for i, img in enumerate(imgs_list):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # Apply binary thresholding to ground truth masks
        if is_mask:
            _, image = cv2.threshold(image, settings.MASK_BINARY_THRESHOLD, settings.MASK_BLOODVESSEL, cv2.THRESH_BINARY)

        # Reshape from (height, width) to (height, width, 1)
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
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        is_mask=True))

    # Do the same for the test images
    output_paths.append(convert_to_hdf5(os.path.join(settings.TEST_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT))

    return output_paths


if __name__ == "__main__":
    if  settings.IS_DEVELOPMENT:
        hdf5_paths = perform_hdf5_conversion()
    else:
        # During development avoid performing the HDF5 conversion for every run
        hdf5_paths = ["../data/MSC8002/training/images.h5",
                      "../data/MSC8002/training/groundtruths.h5",
                      # "../data/MSC8002/test/images.hdf5"
                      ]

    # Read the training images and ground truths
    train_imgs = perform_image_preprocessing(hdf5_paths[0], settings.HDF5_KEY)
    train_grndtr = perform_groundtruth_preprocessing(hdf5_paths[1], settings.HDF5_KEY)

    # Show an image plus its ground truth to check
    IX = 69
    cv2.imshow("CHECK image", train_imgs[IX])
    cv2.imshow("CHECK ground truth", train_grndtr[IX])
    print("       Max image intensity: {} - {} - {}".format(np.max(train_imgs[IX]), train_imgs.dtype, train_imgs.shape))
    print("Max ground truth intensity: {} - {} - {}".format(np.max(train_grndtr[IX]), train_grndtr.dtype, train_grndtr.shape))
    cv2.waitKey(0)

    # Only train using 10 images to test the pipeline
    PRED_IX = range(69, 79)
    # PRED_IX = range(59, 89)
    train_imgs = train_imgs[[PRED_IX]]
    train_grndtr = train_grndtr[[PRED_IX]]

    # Print class distribution
    print("total: {}".format(320*320))
    unique, counts = np.unique(train_grndtr[0], return_counts=True)
    print("countr: {}".format(dict(zip(unique, counts))))

    # Shuffle the data set
    idx = np.random.permutation(len(train_imgs))
    train_imgs, train_grndtr= train_imgs[idx], train_grndtr[idx]

    # Instantiate the U-Net model
    unet = UNet_NN(img_height=settings.IMG_HEIGHT,
                   img_width=settings.IMG_WIDTH,
                   img_channels=settings.IMG_CHANNELS,
                   num_classes=settings.NUM_CLASSES)

    # model = unet.build_model_sigmoid()                # flatten
    # model = unet.build_model_3D_soft()
    model = unet.build_model_4D_soft()

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

    # Convert the ground truths into the same shape as the predictions the U-net produces
    print("--- \nEncoding training ground truths")
    print("Ground truth shape before conversion: {}".format(train_grndtr.shape))

    # train_grndtr_ext_conv = train_grndtr                                                                            # no conversion for sigmoid
    # train_grndtr_ext_conv = convert_img_to_pred_3D(train_grndtr, settings.NUM_CLASSES, settings.VERBOSE)            # softmax: 3D
    train_grndtr_ext_conv = convert_img_to_pred_4D(train_grndtr, settings.NUM_CLASSES, settings.VERBOSE)  # softmax: 4D

    print(" Ground truth shape AFTER conversion: {} of type {}\n".format(train_grndtr_ext_conv.shape, train_grndtr_ext_conv.dtype))

    # Train the model
    print("\n--- Start training")
    # Prepare callbacks
    callbacks = [ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
                 EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode="auto"),
                 CSVLogger(csv_path, append=False),
                 ]

    # Set the optimiser, loss function and metrics
    opt = Adam()
    metrics = [dice_coef]

    # loss = "categorical_crossentropy"                             # Works only when MASK_BINARY_THRESHOLD is 1
    loss = weighted_pixelwise_crossentropy_loss([1., 50.])
    # loss = focal_loss

    # Compile and fit
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    hist = model.fit(train_imgs, train_grndtr_ext_conv,
                     epochs=settings.TRN_NUM_EPOCH,
                     # epochs=2,
                     batch_size=settings.TRN_BATCH_SIZE,
                     verbose=1,
                     shuffle=True,
                     validation_split=settings.TRN_TRAIN_VAL_SPLIT,
                     callbacks=callbacks)

    # Plot the training results - currently breaks if training stopped early
    plot_training_history(hist,
                          settings.TRN_NUM_EPOCH,
                          show=False,
                          save_path=settings.OUTPUT_PATH + unet.title,
                          time_stamp=True,
                          metric="dice_coef")

    print("\n--- Training complete")


    # For pipeline testing only, predict on one training image
    predictions = model.predict(train_imgs[[0]], batch_size=settings.TRN_BATCH_SIZE, verbose=2)

    # predictions = predictions
    # predictions = convert_pred_to_img_3D(predictions, settings.IMG_HEIGHT, settings.TRN_PRED_THRESHOLD)
    predictions = convert_pred_to_img_4D(predictions, settings.IMG_HEIGHT, settings.TRN_PRED_THRESHOLD)

    cv2.imshow("PRED org image", train_imgs[0])
    cv2.imshow("PRED org ground truth", train_grndtr[0])
    cv2.imshow("PRED predicted mask", predictions[0])
    print("  original {} dtype {}".format(np.max(train_imgs[0]), train_imgs[0].dtype))
    print("  gr truth {} dtype {}".format(np.max(train_grndtr[0]), train_grndtr[0].dtype))
    print("prediction {} dtype {}".format(np.max(predictions[0]), predictions[0].dtype))
    cv2.waitKey(0)
