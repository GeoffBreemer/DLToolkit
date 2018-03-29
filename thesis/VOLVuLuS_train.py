RANDOM_STATE = 421
from numpy.random import seed
seed(RANDOM_STATE)

from tensorflow import set_random_seed
set_random_seed(RANDOM_STATE)

import random
random.seed = RANDOM_STATE

import VOLVuLuS_settings as settings

from dltoolkit.utils.generic import model_architecture_to_file, model_summary_to_file
from dltoolkit.nn.segment import UNet_3D_NN
from dltoolkit.utils.visual import plot_training_history

from thesis_common import read_preprocess_image, read_preprocess_groundtruth, \
    convert_img_to_pred_3D, convert_pred_to_img_3D, convert_to_hdf5_3D
from thesis_metric_loss import dice_coef, weighted_pixelwise_crossentropy_loss, focal_loss

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam

import numpy as np
import os, cv2, time

# Useful links:
#
# generators: https://github.com/ncullen93/Unet-ants


# TODO:
# - data augmentation: slight rotating, translating and zooming. Slight skewing also makes sense as the patient head
# position is not always in the exact position.
# -

def perform_hdf5_conversion_3D(settings):
    """Convert the training and test images, ground truths and masks to HDF5 format. The assumption is that images
    are all places in the same folder, regardless of the patient.
    """
    output_paths = []

    print("training images")
    # Convert training images in each sub folder to a single HDF5 file
    output_paths.append(convert_to_hdf5_3D(os.path.join(settings.TRAINING_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        settings=settings))

    print("training ground truths")
    # Training ground truths
    path, class_weights = convert_to_hdf5_3D(os.path.join(settings.TRAINING_PATH, settings.FLDR_GROUND_TRUTH),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        settings=settings, is_mask=True)

    output_paths.append(path)

    # Do the same for the test images
    print("test images")
    output_paths.append(convert_to_hdf5_3D(os.path.join(settings.TEST_PATH, settings.FLDR_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        img_exts=".jpg", key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        settings=settings))

    return output_paths, class_weights


if __name__ == "__main__":
    if settings.IS_DEVELOPMENT:
        print("\n--- Converting images to HDF5")
        hdf5_paths, class_weights = perform_hdf5_conversion_3D(settings)
    else:
        # During development avoid performing HDF5 conversion for every run
        hdf5_paths = ["../data/MSC8002/training_3d/images.h5",
                      "../data/MSC8002/training_3d/groundtruths.h5",
                      ]

    # Read the training images and ground truths
    print("\n--- Read and preprocess images")
    train_imgs = read_preprocess_image(hdf5_paths[0], settings.HDF5_KEY, is_3D=True)
    train_grndtr = read_preprocess_groundtruth(hdf5_paths[1], settings.HDF5_KEY, is_3D=True)

    # Only train using a small number of images to test the pipeline
    print("\n--- Show example image")
    PATIENT_ID = 0
    IX_START = 0
    print(train_imgs.shape)
    cv2.imshow("CHECK image", train_imgs[PATIENT_ID, :, :, IX_START, :])
    cv2.imshow("CHECK ground truth", train_grndtr[PATIENT_ID, :, :, IX_START, :])
    print("Max image intensity: {} - {} - {}".format(np.max(train_imgs[PATIENT_ID, :, :, IX_START, :]),
                                                            train_imgs.dtype,
                                                            train_imgs.shape))
    print("Max grtrh intensity: {} - {} - {}".format(np.max(train_grndtr[PATIENT_ID, :, :, IX_START, :]),
                                                            train_grndtr.dtype,
                                                            train_grndtr.shape))


    lala = train_imgs[PATIENT_ID, :, :, IX_START, :]
    print(lala.shape)
    print(lala[120:120+10, 100])

    cv2.waitKey(0)

    # Only train using a small number of images to test the pipeline
    print("\n--- Limiting training set size for pipeline testing")
    # PRED_IX = range(IX_START, IX_START + settings.NUM_SLICES)
    PRED_IX = range(45, 45+settings.NUM_SLICES)
    train_imgs = train_imgs[:, :, :, PRED_IX]
    train_grndtr = train_grndtr[:, :, :, PRED_IX]

    # Print class distribution
    class_weights = [settings.CLASS_WEIGHT_BACKGROUND, settings.CLASS_WEIGHT_BLOODVESSEL]
    print("Class distribution: {}".format(class_weights))

    # Shuffle the data set
    idx = np.random.permutation(len(train_imgs))
    train_imgs, train_grndtr = train_imgs[idx], train_grndtr[idx]

    # Instantiate the 3D U-Net model
    unet = UNet_3D_NN(img_height=settings.IMG_HEIGHT,
                      img_width=settings.IMG_WIDTH,
                      num_slices=settings.NUM_SLICES,
                      img_channels=settings.IMG_CHANNELS,
                      num_classes=settings.NUM_CLASSES)
    model = unet.build_model_no_BN()
    print("Input shape: {}, output shape: {}".format(model.input_shape, model.output_shape))

    # Prepare some path strings
    model_path = os.path.join(settings.MODEL_PATH, "VOLVuLuS_" + unet.title + "_ep{}.model".format(settings.TRN_NUM_EPOCH))
    summ_path = os.path.join(settings.OUTPUT_PATH, "VOLVuLuS_" + unet.title + "_model_summary.txt")
    csv_path = os.path.join(settings.OUTPUT_PATH, "VOLVuLuS_" + unet.title + "_training_ep{}_bs{}.csv".format(settings.TRN_NUM_EPOCH,
                                                                                                settings.TRN_BATCH_SIZE))

    # Print the architecture to the console, a text file and an image
    model.summary()
    model_summary_to_file(model, summ_path)
    model_architecture_to_file(unet.model, settings.OUTPUT_PATH + "VOLVuLuS_" + unet.title)

    # Convert the ground truths into the same shape as the predictions the 3D U-net produces
    print("\n--- Encoding training ground truths")
    print("Ground truth shape before conversion: {} of type {}".format(train_grndtr.shape, train_grndtr.dtype))
    train_grndtr_ext_conv = convert_img_to_pred_3D(train_grndtr, settings.NUM_CLASSES, settings.VERBOSE)
    print(" Ground truth shape after conversion: {} of type {}".format(train_grndtr_ext_conv.shape, train_grndtr_ext_conv.dtype))

    # Test the pred to image conversion
    # print("\n--- Test prediction to image conversion")
    # back = convert_pred_to_img_3D(train_grndtr_ext_conv)
    # print("back={}, dtype={}".format(back.shape, back.dtype))
    # cv2.imshow("back image", back[0, 0, :, :])
    # cv2.waitKey(0)
    # exit()

    # Train the model
    print("\n--- Start training")
    # Prepare callbacks

    # WITH a validation set:
    # callbacks = [
    #     ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    #     EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode="auto"),
    #     CSVLogger(csv_path, append=False),
    #     ]

    # WITHOUT a validation set:
    callbacks = [
        ModelCheckpoint(model_path, monitor="loss", mode="min", save_best_only=True, verbose=1),
        EarlyStopping(monitor='loss', min_delta=0, patience=settings.TRN_EARLY_PATIENCE, verbose=0, mode="auto"),
        CSVLogger(csv_path, append=False),
        ]

    # Set the optimiser, loss function and metrics
    opt = Adam()
    metrics = [dice_coef]
    loss = weighted_pixelwise_crossentropy_loss(class_weights)
    # loss = dice_coefficient_loss

    start_time = time.time()

    # Compile and fit
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    hist = model.fit(train_imgs, train_grndtr_ext_conv,
                     epochs=settings.TRN_NUM_EPOCH,
                     # epochs=2,
                     batch_size=settings.TRN_BATCH_SIZE,
                     verbose=2,
                     shuffle=True,
                     # validation_split=settings.TRN_TRAIN_VAL_SPLIT,
                     callbacks=callbacks)

    # Save the last model
    # model.save_weights(model_path)

    # Plot the training results
    # plot_training_history(hist,
    #                       settings.TRN_NUM_EPOCH,
    #                       show=False,
    #                       save_path=settings.OUTPUT_PATH + unet.title,
    #                       time_stamp=True,
    #                       metric="dice_coef")

    print("\n--- Training complete")

    # For pipeline testing only, predict on one training image
    print("\n--- Predicting")
    predictions = model.predict(train_imgs, batch_size=settings.TRN_BATCH_SIZE, verbose=2)
    print("pred shape 1: {}".format(predictions.shape))

    print("\n--- Visualising")
    predictions = convert_pred_to_img_3D(predictions)
    print("pred shape 2: {}".format(predictions.shape))

    # Transpose images and ground truths to the correct oder
    print("  train_imgs 1: {}".format(train_imgs.shape))
    print("train_grndtr 1: {}".format(train_grndtr.shape))

    train_imgs = np.transpose(train_imgs, axes=(0, 3, 1, 2, 4))
    train_grndtr = np.transpose(train_grndtr, axes=(0, 3, 1, 2, 4))

    print("  train_imgs 2: {}".format(train_imgs.shape))
    print("train_grndtr 2: {}".format(train_grndtr.shape))

    # Show a single image, ground truth and prediction
    cv2.imshow("PRED org image", train_imgs[0, 0])
    cv2.imshow("PRED org ground truth", train_grndtr[0, 0])
    cv2.imshow("PRED predicted mask", predictions[0, 0])
    print("  original {} dtype {}".format(np.max(train_imgs[0, 0]), train_imgs[0, 0].dtype))
    print("  gr truth {} dtype {}".format(np.max(train_grndtr[0, 0]), train_grndtr[0, 0].dtype))
    print("prediction {} dtype {}".format(np.max(predictions[0, 0]), predictions[0, 0].dtype))
    cv2.waitKey(0)

    print("Elapsed training time: {} min".format(int((time.time() - start_time))/60))
