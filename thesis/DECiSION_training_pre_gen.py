# Relevant links used during development (TODO delete when finished):
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
#   Loss functions:
#   https://datascience.stackexchange.com/questions/23968/99-validation-accuracy-but-0-prediction-results-unet-architecture
#   https://github.com/keras-team/keras/issues/2115
#   --> https://github.com/keras-team/keras/issues/2115
#   https://github.com/keras-team/keras/issues/9395
#
#   Class weights
#   https://github.com/keras-team/keras/issues/5116
#
#   Other approaches:
#   https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/22951
#
#   AlexNet:
#   https://github.com/heuritech/convnets-keras
#   https://devblogs.nvidia.com/image-segmentation-using-digits-5/
#   For the AlexNet, the images(for the mode without the heatmap) have to be of shape (227, 227).It is recommended to
#   resize the images with a size of (256, 256), and then do a crop of size (227, 227).The colors are in RGB order.
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

import DECiSION_settings as settings

from dltoolkit.utils.generic import model_architecture_to_file, model_summary_to_file
from dltoolkit.utils.image import standardise
from dltoolkit.nn.segment import UNet_NN
from dltoolkit.utils.visual import plot_training_history

from thesis_common import read_preprocess_image, read_preprocess_groundtruth,\
    convert_img_to_pred, convert_pred_to_img, convert_to_hdf5, group_images
from thesis_metric_loss import dice_coef, weighted_pixelwise_crossentropy_loss

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os, cv2


def perform_hdf5_conversion(settings):
    """Convert the training and test images, ground truths and masks to HDF5 format. The assumption is that images
    are all placed in the same folder, regardless of the patient.
    """
    output_paths = []

    # Convert training images in each sub folder to a single HDF5 file
    output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        settings=settings))

    # Training ground truths
    path, class_weights = convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FLDR_GROUND_TRUTH),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        settings=settings, is_mask=True)

    output_paths.append(path)

    # Do the same for the test images
    output_paths.append(convert_to_hdf5(os.path.join(settings.TEST_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        settings=settings))

    return output_paths, class_weights


# def create_data_generators(X, Y, batch_size):
#     """Create Keras image data generators"""
#     # Split the training set into a training and validation set
#     # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
#
#     RANDOM_STATE = 122177
#     # Create generator arguments
#     data_gen_args = dict(
#         rotation_range=2.,
#                          width_shift_range=0.01,
#                          height_shift_range=0.01,
#                          shear_range=1.2,
#                          zoom_range=0.01,
#                          # horizontal_flip=False,
#                          # vertical_flip=False,
#                          fill_mode='nearest')
#
#     # Create and fit the training data generators
#     train_image_datagen = ImageDataGenerator(**data_gen_args)
#     train_mask_datagen = ImageDataGenerator(**data_gen_args)
#
#     # train_image_datagen.fit(X, augment=True, seed=RANDOM_STATE)
#     # train_mask_datagen.fit(Y, augment=True, seed=RANDOM_STATE)
#
#     train_image_gen = train_image_datagen.flow(X, batch_size=batch_size, shuffle=True, seed=RANDOM_STATE)
#     train_mask_gen = train_mask_datagen.flow(Y, batch_size=batch_size, shuffle=True, seed=RANDOM_STATE)
#
#     train_generator = zip(train_image_gen, train_mask_gen)
#
#     # Create the validation data generators (do not apply any augmentation)
#     val_image_datagen = ImageDataGenerator()
#     val_mask_datagen = ImageDataGenerator()
#
#     # val_image_gen = val_image_datagen.flow(X_val, batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_STATE)
#     # val_mask_gen = val_mask_datagen.flow(Y_val, batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_STATE)
#
#     # val_generator = zip(val_image_gen, val_mask_gen)
#
#     return train_generator
#     # return train_image_gen, train_mask_gen


if __name__ == "__main__":
    if settings.IS_DEVELOPMENT:
        print("\n--- Converting images to HDF5")
        hdf5_paths, class_weights = perform_hdf5_conversion(settings)
    else:
        # During development avoid performing HDF5 conversion for every run
        hdf5_paths = ["../data/MSC8002/training/images.h5",
                      "../data/MSC8002/training/groundtruths.h5",
                      ]

    # Read the training images and ground truths
    print("\n--- Read and preprocess images")
    train_imgs = read_preprocess_image(hdf5_paths[0], settings.HDF5_KEY)
    train_grndtr = read_preprocess_groundtruth(hdf5_paths[1], settings.HDF5_KEY)

    # Show one image plus its ground truth as a quick check
    print("\n--- Show example image")
    IX = 69
    cv2.imshow("CHECK image", train_imgs[IX])
    cv2.imshow("CHECK ground truth", train_grndtr[IX])
    print("Max image intensity: {} - {} - {}".format(np.max(train_imgs[IX]), train_imgs.dtype, train_imgs.shape))
    print("Max grtrh intensity: {} - {} - {}".format(np.max(train_grndtr[IX]), train_grndtr.dtype, train_grndtr.shape))
    cv2.waitKey(0)

    # Only train using a small number of images to test the pipeline
    print("\n--- Limiting training set size for pipeline testing")
    PRED_IX = range(69, 79)
    # PRED_IX = range(0, 95)
    train_imgs = train_imgs[[PRED_IX]]
    train_grndtr = train_grndtr[[PRED_IX]]

    # Print class distribution
    class_weights = [settings.CLASS_WEIGHT_BACKGROUND, settings.CLASS_WEIGHT_BLOODVESSEL]
    print("Class distribution: {}".format(class_weights))

    # Shuffle the data set
    idx = np.random.permutation(len(train_imgs))
    train_imgs, train_grndtr= train_imgs[idx], train_grndtr[idx]

    # Instantiate the U-Net model
    unet = UNet_NN(img_height=settings.IMG_HEIGHT,
                   img_width=settings.IMG_WIDTH,
                   img_channels=settings.IMG_CHANNELS,
                   num_classes=settings.NUM_CLASSES)

    # model = unet.build_model_sigmoid()
    # model = unet.build_model_flatten()
    model = unet.build_model_softmax()

    # Prepare some path strings
    model_path = os.path.join(settings.MODEL_PATH, unet.title + "_ep{}.model".format(settings.TRN_NUM_EPOCH))
    summ_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_model_summary.txt")
    csv_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_training_ep{}_bs{}.csv".format(settings.TRN_NUM_EPOCH,
                                                                                                settings.TRN_BATCH_SIZE))

    # Print the architecture to the console, a text file and an image
    model.summary()
    model_summary_to_file(model, summ_path)
    model_architecture_to_file(unet.model, settings.OUTPUT_PATH + unet.title + "_BRAIN_base_training")

    # Convert the ground truths into the same shape as the predictions the U-net produces
    print("--- \nEncoding training ground truths")
    print("Ground truth shape before conversion: {} of type {}".format(train_grndtr.shape, train_grndtr.dtype))
    # train_grndtr_ext_conv = train_grndtr        # no conversion for sigmoid
    # train_grndtr_ext_conv = convert_img_to_pred_flatten(train_grndtr, settings, settings.VERBOSE)  # softmax: 3D
    train_grndtr_ext_conv = convert_img_to_pred(train_grndtr, settings, settings.VERBOSE)  # softmax: 4D
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
    loss = weighted_pixelwise_crossentropy_loss(class_weights)

    # Compile and fit
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    # train_gen, train_mask_gen = create_data_generators(train_imgs, train_grndtr, settings.TRN_BATCH_SIZE)

    hist = model.fit(train_imgs, train_grndtr_ext_conv,
                     epochs=settings.TRN_NUM_EPOCH,
                     batch_size=settings.TRN_BATCH_SIZE,
                     verbose=1,
                     shuffle=True,
                     validation_split=settings.TRN_TRAIN_VAL_SPLIT,
                     callbacks=callbacks)

    # hist = model.fit_generator(train_imgs, train_grndtr_ext_conv,
    #                  epochs=settings.TRN_NUM_EPOCH,
    #                  batch_size=settings.TRN_BATCH_SIZE,
    #                  verbose=1,
    #                  shuffle=True,
    #                  validation_split=settings.TRN_TRAIN_VAL_SPLIT,
    #                  callbacks=callbacks)

    # Plot the training results
    plot_training_history(hist,
                          settings.TRN_NUM_EPOCH,
                          show=False,
                          save_path=settings.OUTPUT_PATH + unet.title,
                          time_stamp=True,
                          metric="dice_coef")

    print("\n--- Training complete")


    print("\n--- Pipeline test")
    # For pipeline testing only, predict on one training image
    predictions = model.predict(train_imgs[[0]], batch_size=settings.TRN_BATCH_SIZE, verbose=2)

    # predictions = predictions
    # predictions = convert_pred_to_img_flatten(predictions, settings.TRN_PRED_THRESHOLD)
    predictions = convert_pred_to_img(predictions, settings, settings.TRN_PRED_THRESHOLD)

    cv2.imshow("PRED org image", train_imgs[0])
    cv2.imshow("PRED org ground truth", train_grndtr[0])
    cv2.imshow("PRED predicted mask", predictions[0])
    print("  original {} dtype {}".format(np.max(train_imgs[0]), train_imgs[0].dtype))
    print("  gr truth {} dtype {}".format(np.max(train_grndtr[0]), train_grndtr[0].dtype))
    print("prediction {} dtype {}".format(np.max(predictions[0]), predictions[0].dtype))
    cv2.waitKey(0)
