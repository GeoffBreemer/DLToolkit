"""Create five crops (center plus four corners) of an image, optionally including horizontal flips as well"""
import cv2
import numpy as np


class CropPreprocessor:
    def __init__(self, img_width, img_height, flip_horiz=True, inter=cv2.INTER_AREA):
        """
        Initialise the class
        :param img_width: desired image width
        :param img_height: desired image height
        :param flip_horiz: whether horizontal flips are also required
        :param inter: desired interpolation method
        """
        self.img_width = img_width
        self.img_height = img_height
        self.flip_horiz = flip_horiz
        self.inter = inter

    def preprocess(self, image):
        """
        Perform patch extraction
        :param image: image data
        :return: NumPy array containing the crops
        """
        crops = []

        (height, width) = image.shape[:2]

        # Determine the image's corners
        corners = [
            [0, 0, self.img_width, self.img_height],                            # top left
            [width - self.img_width, 0, width, self.img_height],                # top right
            [width - self.img_width, height - self.img_height, width, height],  # bottom right
            [0, height - self.img_height, self.img_width, height]               # bottom left
        ]

        # Get the center crop
        crop_width = int(0.5 * (width - self.img_width))
        crop_height = int(0.5, (height - self.img_height))
        corners.append([crop_width, crop_height, width - crop_width, height - crop_height])

        # Get the corner crops
        for (x_start, y_start, x_end, y_end) in corners:
            crop = image[y_start:y_end, x_start:x_end]
            crop = cv2.resize(crop, (self.img_width, self.img_height), interpolation=self.inter)
            crops.append(crop)

        # Create the horizontal flips if required
        if self.flip_horiz:
            mirrors = [cv2.flip(crop, 1) for crop in crops]
            crops.extend(mirrors)

        return np.array(crops)
