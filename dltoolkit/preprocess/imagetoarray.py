"""Change the image dimension ordering, uses Keras' image_data_format setting in ~/.keras/keras.json except if
a specific format ("channels_first" or "channels_last") is provided.
"""
from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:

    def __init__(self, format=None):
        self.format = format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.format)
