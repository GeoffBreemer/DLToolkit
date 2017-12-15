"""Resize an image while maintaining its aspect ratio, cropping the image if/when required
"""
import cv2
import imutils


class ResizeWithAspectRatioPreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        (height, width) = image.shape[:2]
        cropWidth = 0
        cropHeight = 0

        # Determine whether to crop the height or width
        if width < height:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            cropHeight = int((image.shape[0] - self.height)/2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            cropWidth = int((image.shape[1] - self.width)/2.0)

        # Crop the image
        (height, width) = image.shape[:2]
        image = image[cropHeight:height - cropHeight, cropWidth:width - cropWidth]

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)