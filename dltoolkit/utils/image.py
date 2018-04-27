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

    return bn_imgs.astype(np.uint8)


def mean_subtraction(img):
    tmp_img = img - np.mean(img)
    return tmp_img/255.


def standardise_single(image):
    """Standardise a single images, values are float32 between 0.0 and 1.0"""
    imgs_std = np.std(image)
    imgs_mean = np.mean(image)
    imgs_standardised = (image - imgs_mean) / imgs_std

    imgs_standardised = ((imgs_standardised - np.min(imgs_standardised)) / (np.max(imgs_standardised)-np.min(imgs_standardised)))

    # return imgs_standardised.astype(np.float16)
    return imgs_standardised.astype(np.float32)


def standardise(imgs):
    """Standardise an array of images, values are float32 between 0.0 and 1.0"""
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_standardised = (imgs-imgs_mean)/imgs_std

    for i in range(imgs.shape[0]):
        imgs_standardised[i] = standardise_single(imgs_standardised[i])

    # return imgs_standardised.astype(np.float16)
    return imgs_standardised.astype(np.float32)


def normalise_single(img):
    """Normalise an array of RGB images, values are float16 between 0.0 and 255.0"""
    img_normalized = ((img - np.min(img)) / (np.max(img)-np.min(img)))

    return img_normalized.astype(np.uint8)


def normalise(imgs):
    """Normalise an array of RGB images, values are float32 between 0.0 and 255.0"""
    imgs_normalized = np.empty(imgs.shape)

    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs[i] - np.min(imgs[i])) / (np.max(imgs[i])-np.min(imgs[i])))

    # return imgs_normalized.astype(np.float16)
    return imgs_normalized.astype(np.float32)


def gray_to_rgb(imgs):
    """Turn an array of greyscale images into RGB images by copying the greyscale dimension twice
    """
    ret = np.empty(imgs.shape[:3] + (3,), dtype=imgs.dtype)
    ret[:, :, :, 0] = imgs[:, :, :, 0]
    ret[:, :, :, 1] = imgs[:, :, :, 0]
    ret[:, :, :, 2] = imgs[:, :, :, 0]

    return ret