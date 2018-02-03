"""Extract a random patch of a specific size from an image that may have larger dimensions"""
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, img_width, img_height):
        """
        Initialise the class
        :param img_width: desired patch width
        :param img_height: desired patch height
        """
        self.img_width = img_width
        self.img_height = img_height

    def preprocess(self, image):
        """
        Perform patch extraction
        :param image: image data
        :return: one random patch
        """
        return extract_patches_2d(image, (self.img_height, self.img_width), max_patches=1)[0]
