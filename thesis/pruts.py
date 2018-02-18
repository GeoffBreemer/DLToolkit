from dltoolkit.nn.segment import UNet_NN
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import numpy as np


def convert_pred_to_img_ALT(pred, patch_dim, threshold=0.5, verbose=False):
    """Convert patch *predictions* to patch *images* (the opposite of convert_img_to_pred)"""
    pred_images = np.empty((pred.shape[0], pred.shape[1], pred.shape[1]))
    print(pred_images.shape)

    for i in range(pred.shape[0]):
        for pix1 in range(pred.shape[1]):
            for pix2 in range(pred.shape[2]):
                if pred[i, pix1, pix2, 1] > threshold:        # TODO for multiple classes > 2 use argmax
                    pred_images[i, pix1, pix2] = 1.0
                else:
                    pred_images[i, pix1, pix2] = 0.0

    # pred_images = np.reshape(pred_images, (pred.shape[0], patch_dim, patch_dim, 1))
    print(pred_images.shape)
    return pred_images





def convert_pred_to_img(pred, patch_dim, threshold=0.5, verbose=False):
    """Convert patch *predictions* to patch *images* (the opposite of convert_img_to_pred)"""
    pred_images = np.empty((pred.shape[0], pred.shape[1]))
    print(pred_images.shape)

    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            if pred[i, pix, 1] > threshold:        # TODO for multiple classes > 2 use argmax
                pred_images[i, pix] = 1.0
            else:
                pred_images[i, pix] = 0.0

    pred_images = np.reshape(pred_images, (pred.shape[0], patch_dim, patch_dim, 1))

    return pred_images


def convert_img_to_pred(ground_truths, num_model_channels, verbose=False):
    """Convert ground truth *images* into the shape of the *predictions* produced by the U-Net (the opposite of
    convert_pred_to_img)
    """
    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    ground_truths = np.reshape(ground_truths, (ground_truths.shape[0], img_height * img_width))
    new_masks = np.empty((ground_truths.shape[0], img_height * img_width, num_model_channels))

    for image in range(ground_truths.shape[0]):
        if verbose and image % 1000 == 0:
            print("{}/{}".format(image, ground_truths.shape[0]))

        for pix in range(img_height*img_width):
            if ground_truths[image, pix] == 0.:      # TODO: update for multiple classes > 2
                new_masks[image, pix, 0] = 1.0
                new_masks[image, pix, 1] = 0.0
            else:
                new_masks[image, pix, 0] = 0.0
                new_masks[image, pix, 1] = 1.0

    return new_masks




def convert_img_to_pred_ALT(ground_truths, num_model_channels, verbose=False):
    """Convert ground truth *images* into the shape of the *predictions* produced by the U-Net (the opposite of
    convert_pred_to_img)
    """
    img_height = ground_truths.shape[1]
    img_width = ground_truths.shape[2]

    ground_truths = np.reshape(ground_truths, (ground_truths.shape[0], img_height * img_width))
    new_masks = np.empty((ground_truths.shape[0], img_height * img_width, num_model_channels))

    for image in range(ground_truths.shape[0]):
        if verbose and image % 1000 == 0:
            print("{}/{}".format(image, ground_truths.shape[0]))

        for pix in range(img_height*img_width):
            if ground_truths[image, pix] == 0.:      # TODO: update for multiple classes > 2
                new_masks[image, pix, 0] = 1.0
                new_masks[image, pix, 1] = 0.0
            else:
                new_masks[image, pix, 0] = 0.0
                new_masks[image, pix, 1] = 1.0

    new_masks = np.reshape(new_masks , (ground_truths.shape[0], img_height, img_width, num_model_channels))

    return new_masks



if __name__ == "__main__":
    img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread("groundtruth.png", cv2.IMREAD_GRAYSCALE)

    print(type(img))
    img = img/255.
    img = img.reshape((48, 48, 1))
    lala = np.empty((1,48,48,1))
    lala[0]=img
    img = lala
    cv2.imshow("image", img[0])
    cv2.waitKey(0)

    print(type(gt))
    gt = gt/255.
    gt = gt.reshape((48, 48, 1))
    lala = np.empty((1,48,48,1))
    lala[0]=gt
    gt = lala
    cv2.imshow("gt", gt[0])
    cv2.waitKey(0)

    unet = UNet_NN(img_height=48,
                   img_width=48,
                   img_channels=1,
                   num_classes=2)

    # model = unet.build_model()
    model = unet.get_unet()
    model.summary()

    opt = SGD()
    opt = Adam()
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    # Prepare callbacks
    callbacks = [ModelCheckpoint("pruts.model", monitor="loss", mode="min", save_best_only=True, verbose=1),
                 EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode="auto"),
                 ]

    # TODO no validation set
    # gt_conv = convert_img_to_pred(gt, 2)

    print(convert_img_to_pred(gt, 2).shape)
    print(convert_img_to_pred_ALT(gt, 2).shape)

    gt_conv = convert_img_to_pred_ALT(gt, 2)
    # gt_conv = gt
    # print(gt_conv[0])
    hist = model.fit(img, gt_conv,
              epochs=200,
              batch_size=1,
              verbose=1,
              # shuffle=True,
              callbacks=callbacks)

    print("\n--- Training complete")

    predictions = model.predict(img, batch_size=1, verbose=2)
    print(predictions.shape)
    print(predictions[0])
    # predictions_img = convert_pred_to_img(predictions,
    #                                      48,
    #                                      0.5)

    predictions_img = convert_pred_to_img_ALT(predictions,
                                         48,
                                         0.5)


    # predictions_img = predictions
    print(predictions_img.shape)
    print(predictions_img.dtype)
    cv2.imshow("pred", predictions_img[0])
    cv2.waitKey(0)
    print(predictions_img[0])