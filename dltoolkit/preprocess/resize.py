"""Resize an image"""
import cv2


class ResizePreprocessor:
    """Resize an image to a new height, width and interpolation method

    Attributes:
        width: new width of the image
        height: new height of the image
        interp: resize interpolation method
    """
    def __init__(self, width, height, interp=cv2.INTER_AREA):
        """Initialise the class"""
        self.width = width
        self.height = height
        self.interp = interp

    def preprocess(self, image):
        """Preprocess the image by resizing it to the new width and height using the chosen interpolation method

        :param image: the image data to preprocess

        :return: preprocessed image data

        :raises: N/A
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.interp)
