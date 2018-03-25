import numpy as np
import cv2


def rgb_to_gray(rgb):
    """
    Convert an array of RGB images to gray scale using the ITU-R 601-2 luma transform
    :param rgb: list of images
    :return: gray scale version of the images
    """
    bn_imgs = rgb[:,:,:,0]*0.299 + rgb[:,:,:,1]*0.587 + rgb[:,:,:,2]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))

    return bn_imgs.astype("uint8")


def mean_subtraction(img):
    # print(img.shape, img.dtype)
    # lala = np.mean(img.astype("float32"), axis=0)
    # img = img.astype("float32")
    # img-= np.mean(img)
    lala = img - np.mean(img)
    return lala/255.


def standardise_single(image):
    """Standardise a single images, values are float32 between 0.0 and 1.0"""
    imgs_std = np.std(image)
    imgs_mean = np.mean(image)
    imgs_standardised = (image - imgs_mean) / imgs_std

    imgs_standardised = ((imgs_standardised - np.min(imgs_standardised)) / (np.max(imgs_standardised)-np.min(imgs_standardised)))

    return imgs_standardised.astype("float32")


def standardise(imgs):
    """Standardise an array of images, values are float32 between 0.0 and 1.0"""
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_standardised = (imgs-imgs_mean)/imgs_std

    for i in range(imgs.shape[0]):
        imgs_standardised[i] = ((imgs_standardised[i] - np.min(imgs_standardised[i])) / (np.max(imgs_standardised[i])-np.min(imgs_standardised[i])))

    return imgs_standardised.astype("float32")


def normalise(imgs):
    """Normalise an array of RGB images, values are float32 between 0.0 and 255.0"""
    imgs_normalized = np.empty(imgs.shape)

    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs[i] - np.min(imgs[i])) / (np.max(imgs[i])-np.min(imgs[i])))

    return imgs_normalized.astype("float32")


def clahe_equalization(imgs):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an array of images"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    imgs_equalized = np.empty(imgs.shape)

    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))

    return imgs_equalized.astype("uint8")


def adjust_gamma(imgs, gamma=1.0):
    """Apply gamma adjustment"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    new_imgs = np.empty(imgs.shape)

    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)

    return new_imgs.astype("uint8")
