"""Train the U-Net model on DRIVE training data"""
from settings import settings_drive as settings
from drive_utils import perform_image_preprocessing, perform_groundtruth_preprocessing

from dltoolkit.io import HDF5Writer
from dltoolkit.utils.generic import list_images, model_architecture_to_file, model_summary_to_file
from dltoolkit.nn.segment import UNet_NN
from dltoolkit.utils.visual import plot_training_history

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD

import numpy as np
import os, progressbar, cv2, random, time

from PIL import Image                                   # for reading .gif images


def convert_to_hdf5(img_path, img_shape, exts, key):
    """
    Convert images present in `img_path` to HDF5 format. The HDF5 file is created in the same folder as
    where the folder containing the images is located (i.e. one level up from the images)
    :param img_path: path to the folder containing images
    :param img_shape: shape of each image (width, height, # of channels)
    :return: full path to the HDF5 file
    """
    output_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path)) + key
    imgs_list = sorted(list(list_images(basePath=img_path, validExts=exts)))

    # Prepare the HDF5 writer, a label vector is not available because this is a segmentation problem
    hdf5_writer = HDF5Writer((len(imgs_list), img_shape[0], img_shape[1], img_shape[2]), output_path,
                             feat_key=settings.HDF5_KEY,
                             label_key=None,
                             del_existing=True,
                             buf_size=len(imgs_list)
                             )

    # Loop through all images
    widgets = ["Creating HDF5 database ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imgs_list), widgets=widgets).start()
    for i, img in enumerate(imgs_list):
        print(img)
        if exts == ".gif":
            # Ground truth and masks are single colour channel .gif files
            image = np.asarray(Image.open(img).convert("L"))
            image = image.reshape((img_shape[0],
                                   img_shape[1],
                                   img_shape[2]))
        else:
            # Actual images are .tiff files with three channels
            image = cv2.imread(img)

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
    output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FOLDER_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_TIF),
                                        exts=".tif", key=settings.HDF5_EXT))

    output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FOLDER_MANUAL_1),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_GIF),
                                        exts=".gif", key=settings.HDF5_EXT))

    output_paths.append(convert_to_hdf5(os.path.join(settings.TRAINING_PATH, settings.FOLDER_MASK),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_GIF),
                                        exts=".gif", key=settings.HDF5_EXT))

    # Do the same for the test images
    output_paths.append(convert_to_hdf5(os.path.join(settings.TEST_PATH, settings.FOLDER_IMAGES),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_TIF),
                                        exts=".tif", key=settings.HDF5_EXT))

    output_paths.append(convert_to_hdf5(os.path.join(settings.TEST_PATH, settings.FOLDER_MANUAL_1),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_GIF),
                                        exts=".gif", key=settings.HDF5_EXT))

    output_paths.append(convert_to_hdf5(os.path.join(settings.TEST_PATH, settings.FOLDER_MASK),
                                        (settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS_GIF),
                                        exts=".gif", key=settings.HDF5_EXT))

    return output_paths


def is_patch_within_boundary(x, y, img_dim, patch_dim):
    """"
    Return True if the patch with center (x, y) and dimensions patch_dim is fully contained within an image
    of the retina (i.e. a circle) with dimensions img_dim
    """
    x = x - int(img_dim/2)        # origin (0,0) shifted to image center
    y = y - int(img_dim/2)       # origin (0,0) shifted to image center

    R_inside = 270 - int(patch_dim * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV

    radius = np.sqrt((x*x)+(y*y))

    if radius < R_inside:
        return True
    else:
        return False


def generate_random_patches(imgs, ground_truths, num_rnd_patches, patch_dim, patch_channels, verbose=False):
    """Randomly sample small patches from the preprocessed images and corresponding ground truths"""
    start_time = time.time()

    img_dim = imgs.shape[1]
    patch_per_img = int(num_rnd_patches / imgs.shape[0])        # number of random patches to generate per image

    # Placeholder arrays for the image patches and corresponding ground truth patches
    patches = np.empty((num_rnd_patches, patch_dim, patch_dim, patch_channels))
    patches_ground_truths = np.empty((num_rnd_patches, patch_dim, patch_dim, patch_channels))

    # Loop over all images
    total_patch_count = 0
    for i in range(imgs.shape[0]):
        # Sample random patches from each individual image
        p = 0
        while p < patch_per_img:
            # Pick a random center point inside the original image
            x = random.randint(0 + int(patch_dim/2), img_dim - int(patch_dim / 2))
            y = random.randint(0 + int(patch_dim/2), img_dim - int(patch_dim / 2))

            # Ensure the patch is completely within the image boundary
            # if not is_patch_within_boundary(x, y, img_dim, patch_dim):
            #     Discard the patch if it is not
                # continue

            # Grab the patch from the original image and ground truth
            img_patch = imgs[i,                                     # current image
                   y - int(patch_dim/2):y + int(patch_dim/2),       # height
                   x - int(patch_dim/2):x + int(patch_dim/2),       # width
                   :]                                               # color channel
            ground_truth_patch = ground_truths[i,
                   y - int(patch_dim/2):y + int(patch_dim/2),
                   x - int(patch_dim/2):x + int(patch_dim/2),
                   :]

            # Copy to placeholders that will be returned from the function
            patches[total_patch_count] = img_patch
            patches_ground_truths[total_patch_count] = ground_truth_patch

            total_patch_count+=1
            p+=1

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return patches, patches_ground_truths


def convert_img_to_pred(ground_truths, num_model_channels, verbose=False):
    """Convert ground truth *images* into the shape of the *predictions* produced by the U-Net (the opposite of
    convert_pred_to_img)
    """
    start_time = time.time()

    im_h = ground_truths.shape[1]
    im_w = ground_truths.shape[2]

    ground_truths = np.reshape(ground_truths, (ground_truths.shape[0], im_h * im_w))
    new_masks = np.empty((ground_truths.shape[0], im_h * im_w, num_model_channels))

    for i in range(ground_truths.shape[0]):
        if verbose and i % 1000 == 0:
            print("{}/{}".format(i, ground_truths.shape[0]))

        for j in range(im_h*im_w):
            if  ground_truths[i, j] == 0:
                new_masks[i,j, 0]=1
                new_masks[i,j, 1]=0
            else:
                new_masks[i,j, 0]=0
                new_masks[i,j, 1]=1

    new_masks = np.reshape(new_masks, (ground_truths.shape[0], im_h, im_w, num_model_channels))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return new_masks


if __name__ == "__main__":
    # Convert images to HDF5 format (without applying any preprocessing), this is only required once

    # if settings.DEVELOPMENT:
    if False:
        # Hard code paths to the HDF5 files during development instead
        hdf5_paths = ['../data/DRIVE/training/images.hdf5',
                        '../data/DRIVE/training/1st_manual.hdf5',
                        '../data/DRIVE/training/mask.hdf5',
                        '../data/DRIVE/test/images.hdf5',
                        '../data/DRIVE/test/1st_manual.hdf5',
                        '../data/DRIVE/test/mask.hdf5']
    else:
        hdf5_paths = perform_hdf5_conversion()

    # Perform training image and ground truth pre-processing. All images are square and gray scale after this
    print("--- Pre-processing training images")
    training_imgs = perform_image_preprocessing(hdf5_paths[0],
                                                settings.HDF5_KEY)
    training_ground_truths = perform_groundtruth_preprocessing(hdf5_paths[1],
                                                               settings.HDF5_KEY)

    # Create the random patches that will serve as the training set
    print("\n--- Generating random training patches")
    patch_imgs, patch_ground_truths = generate_random_patches(training_imgs, training_ground_truths,
                                                              settings.PATCHES_NUM_RND,
                                                              settings.PATCH_DIM,
                                                              settings.PATCH_CHANNELS,
                                                              settings.VERBOSE)

    # Instantiate the U-Net model
    unet = UNet_NN(img_height=settings.PATCH_DIM,
                   img_width=settings.PATCH_DIM,
                   img_channels=settings.PATCH_CHANNELS,
                   dropout_rate=settings.DROPOUT_RATE)

    model = unet.build_model()

    # Prepare some path strings
    model_path = os.path.join(settings.MODEL_PATH, unet.title + "_DRIVE_ep{}_np{}.model".format(settings.NUM_EPOCH,
                                                                                                settings.PATCHES_NUM_RND))
    csv_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_DRIVE_training_ep{}_np{}.csv".format(settings.NUM_EPOCH,
                                                                                            settings.PATCHES_NUM_RND))
    summ_path = os.path.join(settings.OUTPUT_PATH, unet.title + "_DRIVE_model_summary.txt")

    # Print the architecture to the console, a text file and an image
    model.summary()
    model_summary_to_file(model, summ_path)
    model_architecture_to_file(unet.model, settings.OUTPUT_PATH + unet.title + "_DRIVE_training")

    # Convert the random patches into the same shape as the predictions the U-net produces
    print("--- \nEncoding training ground truths")
    patch_ground_truths = convert_img_to_pred(patch_ground_truths, settings.NUM_OUTPUT_CHANNELS, settings.VERBOSE)

    # Train the model
    print("\n--- Start training")
    opt = SGD(momentum=settings.MOMENTUM)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    # Prepare callbacks
    callbacks = [ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
                 EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode="auto"),
                 CSVLogger(csv_path, append=False)
                 ]

    hist = model.fit(patch_imgs, patch_ground_truths,
              epochs=settings.NUM_EPOCH,
              batch_size=settings.BATCH_SIZE,
              verbose=1,
              shuffle=True,
              validation_split=settings.TRAIN_VAL_SPLIT,
              callbacks=callbacks)

    # Plot the training results
    plot_training_history(hist, settings.NUM_EPOCH, show=False, save_path=settings.OUTPUT_PATH + unet.title + "_DRIVE", time_stamp=True)

    print("\n--- Training complete")
