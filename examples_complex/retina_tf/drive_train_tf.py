"""Train the U-Net model on DRIVE training data using TensorFlow"""
from settings import settings_drive as settings
from drive_utils import perform_image_preprocessing, perform_groundtruth_preprocessing
from drive_train import generate_random_patches, convert_img_to_pred

import tensorflow as tf
from sklearn.model_selection import train_test_split

import os, cv2, time, sys
from math import ceil


def batches(X, y, batch_size):
    """Returns a generator that splits the two datasets into batches"""
    n = int(ceil(len(X) / batch_size))
    for k in range(n):
        a = k * batch_size
        b = a + batch_size
        yield (k, X[a:b], y[a:b])


def eval_data(data_x, data_y, batch_size, sess, loss_op, accuracy_op, X, Y):
    """Calculate the loss and accuracy for two datasets, dividing them into batches"""
    total_acc, total_loss, total_obs = 0, 0, 0

    # Calculate the mean loss and accuracy for each batch, keep a tally of the total (i.e not
    # the mean) of the loss and accuracy
    for (k, batch_x, batch_y) in batches(data_x, data_y, batch_size):
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={X: batch_x, Y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
        total_obs += len(batch_x)

    # Return the mean loss and accuracy
    return total_loss/total_obs, total_acc/total_obs


def build_unet_tf(input, patch_dim, num_output_classes):
    """Build a U-Net model that returns logits for each pixel"""
    with tf.name_scope('Model') as scope:
        # Contracting path
        conv_contr1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=(3, 3), strides=1, padding='SAME', activation=tf.nn.relu, use_bias=True)
        conv_contr1 = tf.layers.conv2d(inputs=conv_contr1, filters=32, kernel_size=(3, 3), strides=1, padding='SAME', activation=tf.nn.relu, use_bias=True)
        pool_contr1 = tf.layers.max_pooling2d(inputs=conv_contr1, pool_size=2, strides=2, padding="VALID")

        conv_contr2 = tf.layers.conv2d(inputs=pool_contr1, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)
        conv_contr2 = tf.layers.conv2d(inputs=conv_contr2, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)
        pool_contr2 = tf.layers.max_pooling2d(inputs=conv_contr2, pool_size=2, strides=2, padding="VALID")

        # "Bottom" layer
        conv_bottom = tf.layers.conv2d(inputs=pool_contr2, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)
        conv_bottom = tf.layers.conv2d(inputs=conv_bottom, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)

        # Expansive path
        conv_scale_up2 = tf.layers.conv2d_transpose(inputs=conv_bottom, filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")
        merge_up2 = tf.concat(axis=3, values=[conv_scale_up2, conv_contr2])
        conv_up2 = tf.layers.conv2d(inputs=merge_up2, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)
        conv_up2 = tf.layers.conv2d(inputs=conv_up2, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)

        conv_scale_up1 = tf.layers.conv2d_transpose(inputs=conv_up2, filters=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")
        merge_up1 = tf.concat(axis=3, values=[conv_scale_up1, conv_contr1])
        conv_up1 = tf.layers.conv2d(inputs=merge_up1, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)
        conv_up1 = tf.layers.conv2d(inputs=conv_up1, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)

        # Final 1x1 conv layer
        conv_final = tf.layers.conv2d(inputs=conv_up1, filters=num_output_classes, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, use_bias=True)
        conv_final = tf.reshape(conv_final, shape=[-1, patch_dim**2, num_output_classes])

    return conv_final


def save_model(sess, epoch, model_path):
    """Save TensorFlow model variables to disk, the current epoch becomes part of the name"""
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path + '/tf_retina_model-epoch' + str(epoch) +
                           '_np' + str(settings.PATCHES_NUM_RND) +
                           'bs' + str(settings.BATCH_SIZE) +
                           '.ckpt')
    print("Model saved in file: %s" % save_path)


def train_model(X_train, Y_train, X_val, Y_val, model_path, patch_dim, num_output_classes):
    """Train the model"""
    # Placeholders: X = original image, Y = segmented image
    X = tf.placeholder(tf.float32, (None, settings.PATCH_DIM, settings.PATCH_DIM, settings.PATCH_CHANNELS))
    Y = tf.placeholder(tf.float32, (None, settings.PATCH_DIM * settings.PATCH_DIM, settings.NUM_OUTPUT_CLASSES))

    # Instantiate the U-net
    logits = build_unet_tf(X, patch_dim, num_output_classes)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
    train_op = tf.train.AdamOptimizer().minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(logits, 2), tf.argmax(Y, 2))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    with tf.Session() as sess:
        total_time = 0

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(settings.NUM_EPOCH):
            print("Epoch: {}".format(i))
            start_time = time.time()

            # Fit the training set
            for (k, batch_x, batch_y) in batches(X_train, Y_train, settings.BATCH_SIZE):
                if k%100 == 0:
                    print("batch {}".format(k), end=" ")

                _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
                sys.stdout.flush()

            # Evaluate on the validation set
            val_loss, val_acc = eval_data(X_val, Y_val, settings.BATCH_SIZE, sess, loss_op, accuracy_op, X, Y)
            tf.summary.scalar('validation_loss', val_loss)
            tf.summary.scalar('validation_accuracy', val_acc)

            # Save the every model
            save_model(sess, i, model_path)

            total_time += time.time() - start_time
            print('elapsed time = {:<.2f} min.\n'.format((time.time() - start_time) / 60.0))
            print("val_loss: {}, val_acc: {}".format(val_loss, val_acc))
            print()

    sess.close()
    print('         Total time = {:<.2f} min'.format(total_time / 60.0))


if __name__ == "__main__":
    # Convert images to HDF5 format (without applying any preprocessing), this is only required once
    # Hard code paths to the HDF5 files during development instead
    hdf5_paths = ["../../data/DRIVE/training/images.hdf5",
                    "../../data/DRIVE/training/1st_manual.hdf5",
                    "../../data/DRIVE/training/mask.hdf5",
                    "../../data/DRIVE/test/images.hdf5",
                    "../../data/DRIVE/test/1st_manual.hdf5",
                    "../../data/DRIVE/test/mask.hdf5"]

    # Perform training image and ground truth pre-processing. All images are square and gray scale after this
    print("--- Pre-processing training images")
    training_imgs = perform_image_preprocessing(hdf5_paths[0],
                                                settings.HDF5_KEY)

    training_ground_truths = perform_groundtruth_preprocessing(hdf5_paths[1],
                                                               settings.HDF5_KEY)

    print("--- Showing example image and ground truth")
    cv2.imshow("Preprocessed image", training_imgs[0])
    cv2.waitKey(0)
    cv2.imshow("Preprocessed ground dtruth", training_ground_truths[0])
    cv2.waitKey(0)

    # Generate random patches that will serve as the training set
    print("\n--- Generating random training patches")
    patch_imgs, patch_ground_truths = generate_random_patches(training_imgs, training_ground_truths,
                                                              settings.PATCHES_NUM_RND,
                                                              settings.PATCH_DIM,
                                                              settings.PATCH_CHANNELS,
                                                              settings.VERBOSE)

    # Prepare some path strings
    uunet_title = "U-net_tf"
    model_path = os.path.join(settings.MODEL_PATH, uunet_title + "_DRIVE_ep{}_np{}.model".format(settings.NUM_EPOCH, settings.PATCHES_NUM_RND))
    csv_path = os.path.join(settings.OUTPUT_PATH, uunet_title + "_DRIVE_training_ep{}_np{}_bs{}.csv".format(
        settings.NUM_EPOCH,
        settings.PATCHES_NUM_RND,
        settings.BATCH_SIZE))
    summ_path = os.path.join(settings.OUTPUT_PATH, uunet_title + "_DRIVE_model_summary.txt")

    # Convert the random patches into the same shape as the predictions the U-net produces
    print("--- \nEncoding training ground truths")
    patch_ground_truths_conv = convert_img_to_pred(patch_ground_truths, settings.NUM_OUTPUT_CLASSES, settings.VERBOSE)

    # TensorFlow specific code - START

    patch_imgs = patch_imgs.astype("float32")
    patch_ground_truths = patch_ground_truths.astype("float32")

    # Split the training set into a training and validation set
    X_train, X_val, Y_train, Y_val= train_test_split(patch_imgs,
                                                     patch_ground_truths_conv,
                                                     test_size=settings.TRAIN_VAL_SPLIT,
                                                     # shuffle=False,
                                                     random_state=0)

    # Train the model
    print("\n--- Start training")
    train_model(X_train, Y_train, X_val, Y_val, model_path, settings.PATCH_DIM, settings.NUM_OUTPUT_CLASSES)

    # TensorFlow specific code - END

    print("\n--- Training complete")

    # Plot the training results - currently breaks if training stopped early
    # plot_training_history(hist, settings.NUM_EPOCH, show=False, save_path=settings.OUTPUT_PATH + unet.title + "_DRIVE", time_stamp=True)

    # TODO: calculate performance metrics
