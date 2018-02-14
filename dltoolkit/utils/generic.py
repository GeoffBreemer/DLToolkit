"""Generic utility functions"""
from keras.utils import plot_model
import argparse, os
import numpy as np


def str2bool(v):
    """Attempt to convert a string to a boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ranked_accuracy(predictions, labels):
    """Return the rank 1 and rank 5 accuracy"""
    rank1 = 0
    rank5 = 0

    for (pred, lbl) in zip(predictions, labels):
        pred = np.argsort(pred)[::-1]

        if lbl in pred[:5]:
            rank5+=1

        if lbl == pred[0]:
            rank1+=1

    rank1 /= float(len(labels))
    rank5 /= float(len(labels))

    return (rank1, rank5)


def save_model_architecture(model, save_path, show_shapes=True):
    """Save a picture of the model architecture to disk"""
    plot_model(model, to_file=save_path + "_architecture.png", show_shapes=show_shapes)


def list_images(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    """

    Original code by Adrian Rosebrock, https://github.com/jrosebr1/imutils
    :param basePath: path to search recursively
    :param validExts: limit the search to specific file extensions
    :param contains: limit the search to files containing a specific string
    :return: generator yielding the full path to each image
    """
    return _list_files(basePath, validExts, contains=contains)

def _list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    """Helper function for list_images performing the actual search"""
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath