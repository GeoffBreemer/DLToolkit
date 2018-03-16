import settings_DECiSION as settings
from common_DECiSION import perform_image_preprocessing, extend_images

from keras.models import load_model

import os, cv2, time
import numpy as np
import argparse


def convert_pred_to_img(pred, patch_dim, threshold=0.5, verbose=False):
    """Convert patch *predictions* to patch *images* (the opposite of convert_img_to_pred)"""
    start_time = time.time()

    pred_images = np.empty((pred.shape[0], pred.shape[1]))
    # pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1] * pred.shape[2]))

    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            if pred[i, pix, 1] > threshold:        # TODO for multiple classes > 2 use argmax
                pred_images[i, pix] = 1
            else:
                pred_images[i, pix] = 0

    pred_images = np.reshape(pred_images, (pred.shape[0], patch_dim, patch_dim, 1))

    if verbose:
        print("Elapsed time: {}".format(time.time() - start_time))

    return pred_images


def model_name_from_arguments():
    """Return the full path of the model to be used for making predictions"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, nargs='?',
                    const=True, required=True, help="Set to the full path of the trained model to use")
    args = vars(ap.parse_args())

    return args["model"]


if __name__ == "__main__":
    # Load and preprocess the test and ground truth images (the latter will not be used during inference,
    # just for visualisation)
    print("--- Pre-processing test images")
    test_imgs = perform_image_preprocessing(os.path.join(settings.TRAINING_PATH,
                                                         settings.FLDR_IMAGES + settings.HDF5_EXT),
                                            settings.HDF5_KEY, True)

    test_ground_truths = perform_image_preprocessing(os.path.join(settings.TRAINING_PATH,
                                                                        settings.FLDR_GROUND_TRUTH + settings.HDF5_EXT),
                                                           settings.HDF5_KEY, True)

    test_imgs = test_imgs[[150]]
    cv2.imshow("omg", test_imgs[0])
    cv2.waitKey(0)

    test_ground_truths = test_ground_truths[[150]]
    cv2.imshow("gt", test_ground_truths[0])
    cv2.waitKey(0)

    print(test_imgs[[150]].shape)
    print(test_ground_truths[[150]].shape)

    # TODO: both should use is_training=False
    # print(test_imgs.shape)
    # test_imgs = test_imgs[[0],:,:,:]
    # test_ground_truths = test_ground_truths[[0],:,:,:]

    # Extend images and ground truths to ensure patches cover the entire image
    print("\n--- Extending images")
    test_imgs, new_img_dim, num_patches = extend_images(test_imgs, settings.IMG_DIM_EXT)
    test_ground_truths, _, _ = extend_images(test_ground_truths, settings.IMG_DIM_EXT)

    # Keras specific code - START

    # Load the trained U-net model
    print("\n--- Loading trained model: {}".format(model_name_from_arguments()))
    model = load_model(model_name_from_arguments())

    # Make predictions on the patches
    print("\n--- Making predictions")
    predictions = model.predict(test_imgs, batch_size=settings.TRN_BATCH_SIZE, verbose=2)

    predictions = convert_pred_to_img(predictions, settings.IMG_DIM_EXT, settings.TRN_PRED_THRESHOLD)

    # Keras specific code - END

    print("GROUND TRUTH")
    tmp_img = test_ground_truths[0]
    cv2.imshow("Ground truth", tmp_img)
    cv2.waitKey(0)

    print("ORIGINAL ")
    tmp_img = test_imgs[0]
    cv2.imshow("Original", tmp_img)
    cv2.waitKey(0)

    print("PREDICTED")
    tmp_img = predictions[0]
    cv2.imshow("Prediction", tmp_img)
    cv2.waitKey(0)

    print("\n--- Predicting complete")
