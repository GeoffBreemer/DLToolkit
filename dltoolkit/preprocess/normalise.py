"""Normalise an image"""
import cv2


class NormalisePreprocessor:
    """Normalise an image to be between 0 and 1"""
    def preprocess(self, image):
        """Preprocess the image by normalising the data

        :param N/A

        :return: normalised image data

        :raises: N/A
        """
        return image.astype("float") / 255.0
