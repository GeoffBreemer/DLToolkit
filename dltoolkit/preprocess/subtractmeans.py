"""Subtract mean RGB values (calculated across the entire data set) from an individual image"""
import cv2


class SubtractMeansPreprocessor:
    def __init__(self, R_mean, G_mean, B_mean):
        """
        Initialise the class
        :param R_mean: mean Red value across the entire data set
        :param G_mean: mean Green value across the entire data set
        :param B_mean: mean Blue valie across the entire data set
        """
        self.R_mean = R_mean
        self.G_mean = G_mean
        self.B_mean = B_mean

    def preprocess(self, image):
        """
        Perform the subtraction
        :param image: image data
        :return: converted
        """
        (B, G, R) = cv2.split(image.astype("float32"))

        R -= self.R_mean
        G -= self.G_mean
        B -= self.B_mean

        return cv2.merge([B, G, R])
