import VOLVuLuS_settings as settings
from thesis_common import read_preprocess_image, read_preprocess_groundtruth, group_images,\
    convert_pred_to_img_3D, model_name_from_arguments

from dltoolkit.nn.segment import UNet_3D_NN

import os, cv2
import numpy as np


if __name__ == "__main__":
    # Load and preprocess the test and ground truth images (the latter will not be used during inference,
    # only for visualisation)
    print("--- Pre-processing test NO TRAINING images")
    test_imgs = read_preprocess_image(os.path.join(settings.TRAINING_PATH,
                                                   settings.FLDR_IMAGES + settings.HDF5_EXT),
                                      settings.HDF5_KEY, is_3D=True)
    test_ground_truths = read_preprocess_groundtruth(os.path.join(settings.TRAINING_PATH,
                                                                  settings.FLDR_GROUND_TRUTH + settings.HDF5_EXT),
                                                     settings.HDF5_KEY, is_3D=True)

    # Show an image plus its ground truth to check
    IX_START = 69
    # cv2.imshow("CHECK image", test_imgs[IX])
    # cv2.imshow("CHECK ground truth", test_ground_truths[IX])
    # print("       Max image intensity: {} - {} - {}".format(np.max(test_imgs[IX]), test_imgs.dtype, test_imgs.shape))
    # print("Max ground truth intensity: {} - {} - {}".format(np.max(test_ground_truths[IX]), test_ground_truths.dtype, test_ground_truths.shape))
    # cv2.waitKey(0)

    # Only predict for some images
    # PRED_IX = IX
    PRED_IX = range(IX_START, IX_START + settings.NUM_SLICES)
    test_imgs = test_imgs[:, :, :, PRED_IX]
    test_ground_truths = test_ground_truths[:, :, :, PRED_IX]

    # Load the trained model
    print("\n--- Loading trained model: {}".format(model_name_from_arguments()))

    # Create the UNet model and load its saved weights
    unet = UNet_3D_NN(img_height=settings.IMG_HEIGHT,
                      img_width=settings.IMG_WIDTH,
                      num_slices=settings.NUM_SLICES,
                      img_channels=settings.IMG_CHANNELS,
                      num_classes=settings.NUM_CLASSES)
    model = unet.build_model_no_BN()
    model.load_weights(model_name_from_arguments())
    model.summary()

    # Make predictions
    print("\n--- Making predictions")
    predictions = model.predict(test_imgs, batch_size=settings.TRN_BATCH_SIZE, verbose=2)

    # Convert predictions to images
    predictions = convert_pred_to_img_3D(predictions)
    print(" predictions.shape AFTER conv: {} ".format(predictions.shape))

    tmp_img = test_ground_truths[0, :, :, 0]
    cv2.imshow("Ground truth", tmp_img)
    print("gr truth max {} type {} shape {}".format(np.max(tmp_img), tmp_img.dtype, tmp_img.shape))

    tmp_img = test_imgs[0, :, :, 0]
    cv2.imshow("Original", tmp_img)
    print("original {} type {} shape {}".format(np.max(tmp_img), tmp_img.dtype, tmp_img.shape))

    tmp_img = predictions[0, 0]
    cv2.imshow("Prediction", tmp_img)
    print("prediction {} type {} shape {}".format(np.max(tmp_img), tmp_img.dtype, tmp_img.shape))
    cv2.waitKey(0)

    print("\n--- Producing output images")
    test_imgs = np.transpose(test_imgs, axes=(0, 3, 1, 2, 4))
    group_images(test_imgs[0, 0:settings.NUM_SLICES], 4, 1.0, False, "../output/"+unet.title+"_grp_images_w50")

    test_ground_truths = np.transpose(test_ground_truths, axes=(0, 3, 1, 2, 4))
    group_images(test_ground_truths[0, 0:settings.NUM_SLICES], 4, 255, False, "../output/"+unet.title+"_grp_originals_w50")

    group_images(predictions[0, 0:settings.NUM_SLICES], 4, 1.0, False, "../output/"+unet.title+"_grp_predictions_w50")

    print("\n--- Predicting complete")
