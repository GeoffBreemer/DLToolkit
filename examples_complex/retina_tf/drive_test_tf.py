"""Use a trained U-Net model (using the --m parameter) to make DRIVE test data predictions - TensorFlow version"""
from settings import settings_drive as settings
from drive_utils import perform_groundtruth_preprocessing, perform_image_preprocessing,\
    save_image, group_images
from drive_train_tf import build_unet_tf
from drive_test import extend_images, generate_ordered_patches, convert_pred_to_img, reconstruct_image,\
    apply_masks, load_masks, model_name_from_arguments

import os, cv2
import numpy as np
import tensorflow as tf


def load_model(sess, model_path):
    """Load previously saved TensorFlow variables"""
    saver = tf.train.Saver()
    saver.restore(sess, model_path)


if __name__ == "__main__":
    # Load and preprocess the test and ground truth images (the latter will not be used during inference,
    # just for visualisation)
    print("--- Pre-processing test images")
    test_imgs = perform_image_preprocessing(os.path.join(settings.TEST_PATH,
                                                         settings.FOLDER_IMAGES + settings.HDF5_EXT),
                                            settings.HDF5_KEY, True)

    test_ground_truths = perform_groundtruth_preprocessing(os.path.join(settings.TEST_PATH,
                                                                        settings.FOLDER_MANUAL_1 + settings.HDF5_EXT),
                                                           settings.HDF5_KEY, True)

    # Extend images and ground truths to ensure patches cover the entire image
    print("\n--- Extending images")
    test_imgs, new_img_dim, num_patches = extend_images(test_imgs, settings.PATCH_DIM)
    test_ground_truths, _, _ = extend_images(test_ground_truths, settings.PATCH_DIM)

    # Break up images into patches that will be provided to the U-Net for predicting
    print("\n--- Generating patches")
    patch_imgs = generate_ordered_patches(test_imgs, settings.PATCH_DIM, settings.VERBOSE)
    patch_ground_truths = generate_ordered_patches(test_ground_truths, settings.PATCH_DIM, settings.VERBOSE)

    patch_imgs = patch_imgs.astype("float32")
    patch_ground_truths = patch_ground_truths.astype("float32")

    # TensorFlow specific code - START

    # Load the trained U-net model
    print("\n--- Loading trained model: {}".format(model_name_from_arguments()))

    X = tf.placeholder(tf.float32, (None, settings.PATCH_DIM, settings.PATCH_DIM, settings.PATCH_CHANNELS))
    Y = tf.placeholder(tf.float32, (None, settings.PATCH_DIM * settings.PATCH_DIM, settings.NUM_OUTPUT_CLASSES))
    Y_test = np.zeros(shape=(patch_imgs.shape[0], settings.PATCH_DIM**2, settings.NUM_OUTPUT_CLASSES))

    logits = build_unet_tf(X, settings.PATCH_DIM, settings.NUM_OUTPUT_CLASSES)
    pred_probs = tf.nn.softmax(logits)

    # Make predictions on the patches
    print("\n--- Making predictions")
    with tf.Session() as sess:
        load_model(sess, model_name_from_arguments())
        predictions = sess.run(pred_probs, feed_dict={X: patch_imgs, Y: Y_test})

    # TensorFlow specific code - END

    # Convert patch predictions into patch images
    print("\n--- Reconstructing patch predictions to images")
    predictions_img = convert_pred_to_img(predictions,
                                      settings.PATCH_DIM,
                                      settings.PRED_THRESHOLD,
                                      settings.VERBOSE)

    print("GROUND TRUTH patches")
    tmp_img = group_images(patch_ground_truths, 5*num_patches)
    cv2.imshow("Ground truth", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "patches_ground_truth")

    print("ORIGINAL patches")
    print(patch_imgs[0].shape)
    tmp_img = group_images(patch_imgs, 5*num_patches)
    cv2.imshow("Original", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "patches_original")

    print("PREDICTED patches")
    print(predictions_img[0].shape)
    tmp_img = group_images(predictions_img, 5 * num_patches)
    cv2.imshow("Prediction", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "patches_predicted")

    # Reconstruct images from the predicted patch images
    print("\n--- Reconstructing images from patches")
    reconstructed = reconstruct_image(predictions_img, new_img_dim, settings.VERBOSE)

    tmp_img = group_images(reconstructed, 5)
    cv2.imshow("Reconstructed", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "images_reconstructed")

    # Crop back to original resolution
    # TODO: requires updates to read_preprocess_image, read_preprocess_groundtruth and extend_images
    # test_imgs = test_imgs[:, :, 0:new_img_dim, 0:new_img_dim]
    # predictions = predictions[:, :, 0:new_img_dim, 0:new_img_dim]

    # Load and apply masks
    print("\n--- Loading masks")
    masks = load_masks(os.path.join(settings.TEST_PATH, settings.FOLDER_MASK + settings.HDF5_EXT),
                       settings.HDF5_KEY,
                       settings.PATCH_DIM)

    # Show the original, ground truth and prediction for one image
    print("\n--- Showing masked results")

    cv2.imshow("Mask", masks[2])
    cv2.waitKey(0)
    save_image(masks[0], settings.OUTPUT_PATH + "image_mask")

    # Load and apply masks
    print("\n--- Applying masks")
    apply_masks(reconstructed, masks)

    tmp_img = group_images(reconstructed, 5)
    cv2.imshow("Reconstructed with masks", tmp_img)
    cv2.waitKey(0)
    save_image(tmp_img, settings.OUTPUT_PATH + "images_reconstructed_with_masks")


    cv2.imshow("Masked reconstructed", reconstructed[2])
    cv2.waitKey(0)
    save_image(reconstructed[0], settings.OUTPUT_PATH + "image_reconstruct_masked")

    print("\n--- Predicting complete")
