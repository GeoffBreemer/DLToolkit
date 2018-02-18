from settings import settings_drive as settings
from drive_utils import perform_image_preprocessing, perform_groundtruth_preprocessing, group_images
from drive_test import reconstruct_image, convert_pred_to_img

from dltoolkit.io import HDF5Writer
from dltoolkit.utils.generic import list_images, model_architecture_to_file, model_summary_to_file
from dltoolkit.nn.segment import UNet_NN
from dltoolkit.utils.visual import plot_training_history

from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Cropping2D, Dropout, Conv2DTranspose, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD
from keras import backend as K
from keras.metrics import mean_squared_error

import numpy as np
import os, progressbar, cv2, random, time

from PIL import Image                                   # for reading .gif images

smooth = 1.

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


def convert_img_to_pred(ground_truths, num_model_channels, verbose=False):
    """Convert ground truth *images* into the shape of the *predictions* produced by the U-Net (the opposite of
    convert_pred_to_img)
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
            if ground_truths[image, pix] == 0:
                new_masks[image, pix, 0] = 1
                new_masks[image, pix, 1] = 0
            else:
                new_masks[image, pix, 0] = 0
                new_masks[image, pix, 1] = 1

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return new_masks


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


if __name__ == "__main__":
    # Convert images to HDF5 format (without applying any preprocessing), this is only required once
    hdf5_paths = ["../data/DRIVE/training/images.hdf5",
                    "../data/DRIVE/training/1st_manual.hdf5",
                    "../data/DRIVE/training/mask.hdf5",
                    "../data/DRIVE/test/images.hdf5",
                    "../data/DRIVE/test/1st_manual.hdf5",
                    "../data/DRIVE/test/mask.hdf5"]

    # Perform training image and ground truth pre-processing. All images are square and gray scale after this
    print("--- Pre-processing training images")
    training_imgs = perform_image_preprocessing(hdf5_paths[0],
                                                settings.HDF5_KEY)

    training_ground_truths = perform_groundtruth_preprocessing(hdf5_paths[1],
                                                               settings.HDF5_KEY)

    # TODO
    print("NUmber of training images: {}".format(training_imgs.shape))
    cv2.imshow("preprocessed image", training_imgs[0])
    cv2.waitKey(0)
    cv2.imshow("preprocessed ground dtruth", training_ground_truths[0])
    cv2.waitKey(0)
    # TODO

    # Create the random patches that will serve as the training set
    print("\n--- Generating random training patches")
    patch_imgs = generate_ordered_patches(training_imgs, settings.PATCH_DIM, settings.VERBOSE)
    patch_ground_truths = generate_ordered_patches(training_ground_truths, settings.PATCH_DIM, settings.VERBOSE)

    # TODO
    print("NUmber of pacthes: {}".format(patch_imgs.shape))
    cv2.imshow("selected patch", patch_imgs[50])
    cv2.waitKey(0)
    cv2.imshow(" selected grtr", patch_ground_truths[50])
    cv2.waitKey(0)
    print("patch_img dtype: {}".format(patch_imgs.dtype))
    print(" grtr     dtype: {}".format(patch_ground_truths.dtype))
    # TODO

    # Pick only one
    patch_imgs = patch_imgs[[50]]
    patch_ground_truths = patch_ground_truths[[50]]

    voor_inference = patch_imgs.copy()
    voor_inference_ground_truths = patch_ground_truths.copy()

    print("========== patch shape {}".format(patch_imgs.shape))
    print("==========  grtr shape {}".format(patch_ground_truths.shape))

    # Instantiate the U-Net model
    unet = UNet_NN(img_height=settings.PATCH_DIM,
                   img_width=settings.PATCH_DIM,
                   img_channels=settings.PATCH_CHANNELS,
                   dropout_rate=settings.DROPOUT_RATE)

    model = unet.build_model()
    # model = get_unet()

    # Prepare some path strings
    model_path = os.path.join(settings.MODEL_PATH, unet.title + "_DRIVE_ep{}_np{}.model".format(settings.NUM_EPOCH,
                                                                                                settings.PATCHES_NUM_RND))
    csv_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_DRIVE_training_ep{}_np{}.csv".format(settings.NUM_EPOCH,
                                                                                            settings.PATCHES_NUM_RND))
    summ_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_DRIVE_model_summary.txt")

    # Convert the random patches into the same shape as the predictions the U-net produces
    print("--- \nEncoding training ground truths")

    # Train the model
    print("\n--- Start training")
    # opt = SGD(momentum=settings.MOMENTUM)
    opt = SGD()
    # model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    # model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()
    # model_architecture_to_file(unet.model, settings.OUTPUT_PATH + unet.title + "_DRIVE_trainingkut")

    # Prepare callbacks
    callbacks = [ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
                 EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode="auto"),
                 CSVLogger(csv_path, append=False)
                 ]

    # TODO use a validation set!!!
    # hist = model.fit(patch_imgs, patch_ground_truths,
    #           epochs=settings.NUM_EPOCH,
    #           batch_size=settings.BATCH_SIZE,
    #           verbose=1,
    #           shuffle=True,
    #           validation_split=settings.TRAIN_VAL_SPLIT,
    #           callbacks=callbacks)
    print("========== fit {} and {}".format(patch_imgs.shape, patch_ground_truths.shape))


    labels = convert_img_to_pred(patch_ground_truths, settings.NUM_OUTPUT_CLASSES, False)

    hist = model.fit(patch_imgs, labels,
              epochs=settings.NUM_EPOCH,
              batch_size=settings.BATCH_SIZE,
              verbose=1,
              shuffle=True,
              callbacks=callbacks)

    # Plot the training results - currently breaks if training stopped early
    # plot_training_history(hist, settings.NUM_EPOCH, show=False, save_path=settings.OUTPUT_PATH + unet.title + "_DRIVE", time_stamp=True)

    print("\n--- Training complete")

    # Immediately make predictions on the single patch
    # predictions = model.predict(lala, batch_size=settings.BATCH_SIZE, verbose=2)
    predictions = model.predict(voor_inference, batch_size=settings.BATCH_SIZE, verbose=2)

    print(predictions.shape)


    # def convert_pred_to_img(pred, pred_num_channels, patch_dim, threshold=0.5, verbose=False):


    predictions = convert_pred_to_img(predictions, settings.NUM_OUTPUT_CLASSES, settings.PATCH_DIM)

    cv2.imshow("Org", voor_inference[0])
    cv2.waitKey(0)

    cv2.imshow("Grtr", voor_inference_ground_truths[0])
    cv2.waitKey(0)

    cv2.imshow("Pred", predictions[0])
    cv2.waitKey(0)

    print("===== ground turhts")
    print(voor_inference_ground_truths[0])
    print("===== predictions")
    print(predictions)
    print(predictions.shape)
    print("-----")
    print(predictions[0,0,0])
    print("-----")
    print(predictions[0, 0, 0,:])
