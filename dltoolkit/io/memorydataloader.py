"""Data loading functions
"""
import numpy as np
import cv2
import os


class MemoryDataLoader:
    """Loads images using full image paths

    Attributes:
        preprocessors: array of image preprocessors to apply to images upon loading
    """
    def __init__(self, preprocessors=None):
        """Initialise the class"""
        if preprocessors is None:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors

    def load(self, imagePaths, verbose=-1):
        """Load the data set, both images and their labels, returning two lists *in memory*

        :param imagePaths: list holding the full path to each image. Each image is stored in a subfolder
        with the name of the class the image belongs to:
            /full path/<class name>/<image name>.jpg
        :param verbose: non-zero integer to log information to the console during loading, use -1 for no
        logging at all, any positive number to log information every verbose number of records processed

        :return: a tuple of NumPy arrays holding image data (X) and their associated labels (Y)

        :raises: N/A
        """
        X = []
        Y = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # Apply any preprocessors
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            X.append(image)
            Y.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("Loaded and processed image {}/{}".format(i+1, len(imagePaths)))

        return np.array(X), np.array(Y)
