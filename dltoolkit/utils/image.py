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

    # return bn_imgs
    return bn_imgs.astype("uint8")


# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def normalise(imgs):
    """Normalise an array of RGB images"""
    imgs_normalized = np.empty(imgs.shape)

    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std

    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255

    return imgs_normalized.astype("uint8")


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
