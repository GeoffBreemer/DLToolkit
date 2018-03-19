import DECiSION_settings as settings
from thesis_common import perform_image_preprocessing, perform_groundtruth_preprocessing,\
    convert_img_to_pred_4D, convert_pred_to_img_4D,\
    convert_img_to_pred_3D, convert_pred_to_img_3D, group_images

from dltoolkit.nn.segment import UNet_NN

from keras.models import load_model

import os, cv2, argparse
import numpy as np


def model_name_from_arguments():
    """Return the full path of the model to be used for making predictions"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, nargs='?',
                    const=True, required=True, help="Set to the full path of the trained model to use")
    args = vars(ap.parse_args())

    return args["model"]


if __name__ == "__main__":
    # Load and preprocess the test and ground truth images (the latter will not be used during inference,
    # only for visualisation)
    print("--- Pre-processing test NO TRAINING images")
    test_imgs = perform_image_preprocessing(os.path.join(settings.TRAINING_PATH,
                                                         settings.FLDR_IMAGES + settings.HDF5_EXT),
                                            settings.HDF5_KEY)
    test_ground_truths = perform_groundtruth_preprocessing(os.path.join(settings.TRAINING_PATH,
                                                                        settings.FLDR_GROUND_TRUTH + settings.HDF5_EXT),
                                                           settings.HDF5_KEY)


    # Show an image plus its ground truth to check
    IX = 69
    # cv2.imshow("CHECK image", test_imgs[IX])
    # cv2.imshow("CHECK ground truth", test_ground_truths[IX])
    # print("       Max image intensity: {} - {} - {}".format(np.max(test_imgs[IX]), test_imgs.dtype, test_imgs.shape))
    # print("Max ground truth intensity: {} - {} - {}".format(np.max(test_ground_truths[IX]), test_ground_truths.dtype, test_ground_truths.shape))
    # cv2.waitKey(0)

    # Only predict for some images
    # PRED_IX = IX
    PRED_IX = range(59, 89)
    test_imgs = test_imgs[[PRED_IX]]
    test_ground_truths = test_ground_truths[[PRED_IX]]

    # Load the trained model
    print("\n--- Loading trained model: {}".format(model_name_from_arguments()))

    # Create the UNet model and load its saved weights
    unet = UNet_NN(settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS, settings.NUM_CLASSES)
    # model = unet.build_model()
    # model = unet.build_model_3D_soft()
    model = unet.build_model_4D_soft()
    model.load_weights(model_name_from_arguments())

    # Load the entire model (incl. loss function etc.), does not require prior instantiation
    # model = load_model(model_name_from_arguments())           no need to instantiate the model first
    # model = load_model(model_name_from_arguments(), custom_objects = {'loss': weighted_pixelwise_crossentropy([1, 1])})
    # model = load_model(model_name_from_arguments(), custom_objects = {'loss': dice_coef_loss, 'metric': dice_coef})

    model.summary()

    # Make predictions
    print("\n--- Making predictions")
    predictions = model.predict(test_imgs, batch_size=settings.TRN_BATCH_SIZE, verbose=2)

    # Convert predictions to images
    predictions = convert_pred_to_img_4D(predictions, settings, settings.TRN_PRED_THRESHOLD)
    # predictions = convert_pred_to_img_3D(predictions, settings, settings.TRN_PRED_THRESHOLD)
    # print(predictions[0, 100:110, 100:110])

    print(" predictions.shape AFTER conv: {} ".format(predictions.shape))

    # tmp_img = test_ground_truths[0]
    # cv2.imshow("Ground truth", tmp_img)
    # print("gr truth max {} type {}".format(np.max(tmp_img), tmp_img.dtype))

    # tmp_img = test_imgs[0]
    # cv2.imshow("Original", tmp_img)
    # print("original {} type {}".format(np.max(tmp_img), tmp_img.dtype))

    # tmp_img = predictions[0]
    # cv2.imshow("Prediction", tmp_img)
    # print("prediction {} type {}".format(np.max(tmp_img), tmp_img.dtype))

    print("\n--- Producing output images")
    group_images(test_ground_truths[0:15], 5, 255, False, "../output/"+unet.title+"_grp_originals_w50")
    group_images(predictions[0:15], 5, 1.0, False, "../output/"+unet.title+"_grp_predictions_w50")

    print("\n--- Predicting complete")
