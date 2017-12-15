"""Subtract mean RGB values from an image"""
import cv2


class SubtractMeansPreprocessor:
    def __init__(self, R_mean, G_mean, B_mean):
        self.R_mean = R_mean
        self.G_mean = G_mean
        self.B_mean = B_mean

    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype("float32"))

        R -= self.R_mean
        G -= self.G_mean
        B -= self.B_mean

        return cv2.merge([B, G, R])
