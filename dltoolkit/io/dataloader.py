"""Data loading functions
"""
import numpy as np
import cv2
import os


class DataLoader:
    """Loads images using full image paths

    Attributes:
        preprocessors: array of image preprocessors to apply to images upon loading
    """
    def __init__(self, preprocessors=None):
        """Initialise the class"""
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        """Load the data set, both images and labels

        :param imagePaths: list holding the full path to each image
        :param verbose: non-zero integer to log informatio to the console during loading

        :return: a tuple of NumPy arrays holding image data and their associated labels

        :raises: N/A
        """
        X = []
        Y = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            X.append(image)
            Y.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("Processed {}/{}".format(i+1, len(imagePaths)))

        return np.array(X), np.array(Y)

