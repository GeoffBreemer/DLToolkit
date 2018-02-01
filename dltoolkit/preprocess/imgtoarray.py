"""Change the image dimension ordering, uses Keras' image_data_format setting in ~/.keras/keras.json except if
a specific format ("channels_first" or "channels_last") is provided.
"""
from keras.preprocessing.image import img_to_array


class ImgToArrayPreprocessor:

    def __init__(self, format=None):
        """
        Initialise the class
        :param format: desired dimension ordering ("channels_first" or "channels_last")
        """
        self.format = format

    def preprocess(self, image):
        """
        Perform the conversion
        :param image: image data
        :return: image data converted to the required dimension ordering
        """
        return img_to_array(image, data_format=self.format)
