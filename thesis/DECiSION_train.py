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

from dltoolkit.utils.generic import model_architecture_to_file, model_summary_to_file, list_images
from dltoolkit.utils.image import standardise, standardise_single, mean_subtraction
from dltoolkit.nn.segment import UNet_NN
from dltoolkit.utils.visual import plot_training_history
from dltoolkit.io import HDF5Writer, HDF5Reader, HDF5Generator_Segment

from thesis_common import convert_img_to_pred, convert_pred_to_img #convert_to_hdf5, group_images
from thesis_metric_loss import dice_coef, weighted_pixelwise_crossentropy_loss

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import numpy as np
import os, cv2, time, progressbar


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

    classcounts = [0] * settings.NUM_CLASSES

    # Loop through all images
    widgets = ["Creating HDF5 database ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imgs_list), widgets=widgets).start()
    for i, img in enumerate(imgs_list):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # Apply binary thresholding to ground truth masks
        if is_mask:
            _, image = cv2.threshold(image, settings.MASK_BINARY_THRESHOLD, settings.MASK_BLOODVESSEL, cv2.THRESH_BINARY)

            # Convert to the format produced by the model
            print(image.shape)
            print(np.array([image]).shape)
            image = convert_img_to_pred(np.array([image]), settings, settings.VERBOSE)

            for ix, cl in enumerate([settings.MASK_BACKGROUND, settings.MASK_BLOODVESSEL]):
                classcounts[ix] += len(np.where(image == cl)[0])
        else:
            # Apply preprocessing to images (not to ground truth masks)
            # print("before, min {} max {}".format(np.min(image), np.max(image)))
            # print("Max prior: {}".format(np.max(image)))
            # image = standardise_single(image)
            # image = mean_subtraction(image)
            # print("Max after: {}".format(np.max(image)))
            # print(" after, min {} max {}".format(np.min(image), np.max(image)))
            pass

        # Reshape from (height, width) to (height, width, 1)
        image = image.reshape((img_shape[0], img_shape[1], img_shape[2]))

        if not is_mask:
        #     image = standardise_single(image)
            image = mean_subtraction(image)
            # image=image/255.

        hdf5_writer.add([image], None)
        pbar.update(i)

    if is_mask:
        total = sum(classcounts)
        for i in range(settings.NUM_CLASSES):
            classcounts[i] = int(total / classcounts[i])

    pbar.finish()
    hdf5_writer.close()

    if is_mask:
        return output_path#, classcounts
    else:
        return output_path


def perform_hdf5_conversion(settings):
    # Prepare the path to the training images and ground truths
    img_exts = ".jpg"
    img_path = os.path.join(settings.TRAINING_PATH, settings.FLDR_IMAGES)
    msk_path = os.path.join(settings.TRAINING_PATH, settings.FLDR_GROUND_TRUTH)
    test_path = os.path.join(settings.TEST_PATH, settings.FLDR_IMAGES)

    # Create a list of paths to the individual patient folders
    patient_fld_imgs = sorted([os.path.join(img_path, e.name) for e in os.scandir(img_path) if e.is_dir()])
    patient_fld_masks = sorted([os.path.join(msk_path, e.name) for e in os.scandir(msk_path) if e.is_dir()])
    test_imgs = sorted(list(list_images(basePath=test_path, validExts=img_exts)))

    # Obtain a list of paths to the training images and ground truths for each patient
    img_list = []
    msk_list = []
    for patient_ix, (p_fld_imgs, p_fld_masks) in enumerate(zip(patient_fld_imgs, patient_fld_masks)):
        img_list.extend(sorted(list(list_images(basePath=p_fld_imgs,
                                                validExts=img_exts)))
                        [settings.SLICE_START:settings.SLICE_END])
        msk_list.extend(sorted(list(list_images(basePath=p_fld_masks,
                                                validExts=img_exts)))
                        [settings.SLICE_START:settings.SLICE_END])

    assert(len(img_list) == len(msk_list))

    # Split the training set into a training and validation set
    train_img, val_img, train_msk, val_msk = train_test_split(img_list, msk_list,
                                                              test_size=settings.TRN_TRAIN_VAL_SPLIT,
                                                              random_state=settings.RANDOM_STATE,
                                                              shuffle=True)

    print("Check train data: {} = {}".format(train_img[0], train_msk[0]))
    print("  Check val data: {} = {}".format(val_img[0], val_msk[0]))
    print("Num train: {}, num val: {}".format(len(train_img), len(val_img)))

    # Create the HDF5 data sets
    output_paths = []

    # Training images
    output_paths.append(create_hdf5_db(train_img, "train", img_path,
                                       (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                       key=settings.HDF5_KEY, ext=settings.HDF5_EXT, settings=settings))

    # Training ground truths
    output_paths.append(create_hdf5_db(train_msk, "train", msk_path,
                                       (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                       key=settings.HDF5_KEY, ext=settings.HDF5_EXT, settings=settings,
                                       is_mask=True))

    # Validation images
    output_paths.append(create_hdf5_db(val_img, "val", img_path,
                                       (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                       key=settings.HDF5_KEY, ext=settings.HDF5_EXT, settings=settings))

    # Validation ground truths
    output_paths.append(create_hdf5_db(val_msk, "val", msk_path,
                                       (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                       key=settings.HDF5_KEY, ext=settings.HDF5_EXT, settings=settings,
                                       is_mask=True))

    # Test images (no ground truths available, no need to split)
    output_paths.append(create_hdf5_db(test_imgs, "test", test_path,
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS),
                                        key=settings.HDF5_KEY, ext=settings.HDF5_EXT,
                                        settings=settings))

    return output_paths


# Read all into memory:
if __name__ == "__main__":
    if settings.IS_DEVELOPMENT:
        print("\n--- Converting images to HDF5")
        hdf5_paths = perform_hdf5_conversion(settings)
    else:
        # During development avoid performing HDF5 conversion for every run
        hdf5_paths = ["../data/MSC8002/training/train_imgs.h5",
                      "../data/MSC8002/training/train_masks.h5",
                      "../data/MSC8002/training/val_imgs.h5",
                      "../data/MSC8002/training/val_masks.h5"
                      "../data/MSC8002/test/test_imgs.h5"
                      ]

    # Read the training images and ground truths
    print("\n--- Read and preprocess images")
    train_imgs = read_preprocess_image(hdf5_paths[0], settings.HDF5_KEY)
    train_grndtr = read_preprocess_groundtruth(hdf5_paths[1], settings.HDF5_KEY)
    val_imgs = read_preprocess_image(hdf5_paths[2], settings.HDF5_KEY)
    val_grndtr = read_preprocess_groundtruth(hdf5_paths[3], settings.HDF5_KEY)

    # Show one image plus its ground truth as a quick check
    print("\n--- Show TRAIN example image")
    IX = 0
    cv2.imshow("CHECK image", train_imgs[IX])
    cv2.imshow("CHECK ground truth", train_grndtr[IX])
    print("Max image intensity: {} - {} - {} - {}".format(np.max(train_imgs[IX]), np.min(train_imgs[IX]),
                                                          train_imgs.dtype, train_imgs.shape))
    print("Max grtrh intensity: {} - {} - {} - {}".format(np.max(train_grndtr[IX]), np.min(train_grndtr[IX]),
                                                          train_grndtr.dtype, train_grndtr.shape))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n--- Show VAL example image")
    IX = 0
    cv2.imshow("CHECK image", val_imgs[IX])
    cv2.imshow("CHECK ground truth", val_grndtr[IX])
    print("Max image intensity: {} min: {} - {} - {}".format(np.max(train_imgs[IX]), np.min(train_imgs[IX]),
                                                             val_imgs.dtype, val_imgs.shape))
    print("Max grtrh intensity: {} min: {} - {} - {}".format(np.max(train_grndtr[IX]), np.min(train_grndtr[IX]),
                                                             val_grndtr.dtype, val_grndtr.shape))
    cv2.waitKey(0)

    # Print class distribution
    class_weights = [settings.CLASS_WEIGHT_BACKGROUND, settings.CLASS_WEIGHT_BLOODVESSEL]
    print("Class distribution: {}".format(class_weights))

    # Shuffle the data set
    # idx = np.random.permutation(len(train_imgs))
    # train_imgs, train_grndtr= train_imgs[idx], train_grndtr[idx]

    # Instantiate the U-Net model
    unet = UNet_NN(img_height=settings.IMG_HEIGHT,
                   img_width=settings.IMG_WIDTH,
                   img_channels=settings.IMG_CHANNELS,
                   num_classes=settings.NUM_CLASSES)

    # model = unet.build_model_sigmoid()
    # model = unet.build_model_flatten()
    model = unet.build_model_softmax()

    # Prepare some path strings
    model_path = os.path.join(settings.MODEL_PATH, "DECiSION_" + unet.title + "_ep{}.model".format(settings.TRN_NUM_EPOCH))
    summ_path = os.path.join(settings.OUTPUT_PATH, "DECiSION_" + unet.title + "_model_summary.txt")
    csv_path = os.path.join(settings.OUTPUT_PATH, "DECiSION_" + unet.title + "_training_ep{}_bs{}.csv".format(settings.TRN_NUM_EPOCH,
                                                                                                settings.TRN_BATCH_SIZE))

    # Print the architecture to the console, a text file and an image
    model.summary()
    model_summary_to_file(model, summ_path)
    model_architecture_to_file(unet.model, settings.OUTPUT_PATH + "DECiSION_" + unet.title)

    # Convert the ground truths into the same shape as the predictions the U-net produces
    print("--- \nEncoding training ground truths")
    print("Ground truth shape before conversion: {} of type {}".format(train_grndtr.shape, train_grndtr.dtype))
    # train_grndtr_ext_conv = train_grndtr        # no conversion for sigmoid
    # train_grndtr_ext_conv = convert_img_to_pred_flatten(train_grndtr, settings, settings.VERBOSE)  # softmax: 3D
    train_grndtr_ext_conv = convert_img_to_pred(train_grndtr, settings, settings.VERBOSE)  # softmax: 4D
    val_grndtr_ext_conv = convert_img_to_pred(val_grndtr, settings, settings.VERBOSE)  # softmax: 4D
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
    start_time = time.time()

    hist = model.fit(train_imgs, train_grndtr_ext_conv,
                     epochs=settings.TRN_NUM_EPOCH,
                     batch_size=settings.TRN_BATCH_SIZE,
                     verbose=1,
                     shuffle=True,
                     validation_data=(val_imgs, val_grndtr_ext_conv),
                     # validation_split=settings.TRN_TRAIN_VAL_SPLIT,
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

    print("Elapsed training time: {} min".format(int((time.time() - start_time))/60))


    print("\n--- Pipeline test")
    # For pipeline testing only, predict on one training image
    predictions = model.predict(train_imgs[[0]], batch_size=settings.TRN_BATCH_SIZE, verbose=2)

    # predictions = predictions
    # predictions = convert_pred_to_img_flatten(predictions, settings.TRN_PRED_THRESHOLD)
    predictions = convert_pred_to_img(predictions, settings, settings.TRN_PRED_THRESHOLD)

    cv2.imshow("PRED TRAIN org image", train_imgs[0])
    cv2.imshow("PRED TRAIN org ground truth", train_grndtr[0])
    cv2.imshow("PRED TRAIN predicted mask", predictions[0])
    print("  original {} dtype {}".format(np.max(train_imgs[0]), train_imgs[0].dtype))
    print("  gr truth {} dtype {}".format(np.max(train_grndtr[0]), train_grndtr[0].dtype))
    print("prediction {} dtype {}".format(np.max(predictions[0]), predictions[0].dtype))
    cv2.waitKey(0)


# Use data generators
# if __name__ == "__main__":
#     if settings.IS_DEVELOPMENT:
#         print("\n--- Converting images to HDF5")
#         hdf5_paths = perform_hdf5_conversion(settings)
#     else:
#         # During development avoid performing HDF5 conversion for every run
#         hdf5_paths = ["../data/MSC8002/training/train_imgs.h5",
#                       "../data/MSC8002/training/train_masks.h5",
#                       "../data/MSC8002/training/val_imgs.h5",
#                       "../data/MSC8002/training/val_masks.h5"
#                       "../data/MSC8002/test/test_imgs.h5"
#                       ]
#
#     # Print class distribution
#     class_weights = [settings.CLASS_WEIGHT_BACKGROUND, settings.CLASS_WEIGHT_BLOODVESSEL]
#     print("Class distribution: {}".format(class_weights))
#
#     # Instantiate the U-Net model
#     unet = UNet_NN(img_height=settings.IMG_HEIGHT,
#                    img_width=settings.IMG_WIDTH,
#                    img_channels=settings.IMG_CHANNELS,
#                    num_classes=settings.NUM_CLASSES)
#
#     model = unet.build_model_softmax()
#
#     # Prepare some path strings
#     model_path = os.path.join(settings.MODEL_PATH, "DECiSION_" + unet.title + "_ep{}.model".format(settings.TRN_NUM_EPOCH))
#     summ_path = os.path.join(settings.OUTPUT_PATH, "DECiSION_" + unet.title + "_model_summary.txt")
#     csv_path = os.path.join(settings.OUTPUT_PATH, "DECiSION_" + unet.title + "_training_ep{}_bs{}.csv".format(settings.TRN_NUM_EPOCH,
#                                                                                                 settings.TRN_BATCH_SIZE))
#
#     # Print the architecture to the console, a text file and an image
#     model.summary()
#     model_summary_to_file(model, summ_path)
#     model_architecture_to_file(unet.model, settings.OUTPUT_PATH + "DECiSION_" + unet.title)
#
#     # Train the model
#     print("\n--- Start training")
#     # Prepare callbacks
#     callbacks = [ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
#                  EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode="auto"),
#                  CSVLogger(csv_path, append=False),
#                  ]
#
#     # Set the optimiser, loss function and metrics
#     opt = Adam()
#     metrics = [dice_coef]
#     loss = weighted_pixelwise_crossentropy_loss(class_weights)
#
#     # Compile and fit
#     model.compile(optimizer=opt, loss=loss, metrics=metrics)
#
#
#     # Create data generators
#     training_gen = HDF5Generator_Segment(hdf5_paths[0], hdf5_paths[1], batch_size=settings.TRN_BATCH_SIZE)
#     validation_gen = HDF5Generator_Segment(hdf5_paths[2], hdf5_paths[3], batch_size=settings.TRN_BATCH_SIZE)
#
#     # train_gen, train_mask_gen = create_data_generators(train_imgs, train_grndtr, settings.TRN_BATCH_SIZE)
#     start_time = time.time()
#
#     hist = model.fit_generator(training_gen.generator(settings.TRN_NUM_EPOCH),
#                                epochs=settings.TRN_NUM_EPOCH,
#                                verbose=2,
#                                shuffle=True,
#                                validation_data=validation_gen(settings.TRN_NUM_EPOCH),
#                                callbacks=callbacks)
#
#     # Plot the training results
#     plot_training_history(hist,
#                           settings.TRN_NUM_EPOCH,
#                           show=False,
#                           save_path=settings.OUTPUT_PATH + unet.title,
#                           time_stamp=True,
#                           metric="dice_coef")
#
#     print("\n--- Training complete")
#
#     print("Elapsed training time: {} min".format(int((time.time() - start_time))/60))
