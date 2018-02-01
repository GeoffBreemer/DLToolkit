"""Scale pixel intensities to be bin the range [0, 1]"""


class NormalisePreprocessor:
    def preprocess(self, image):
        """
        Preprocess the image by normalising the data
        :param image: image date
        :return: normalised image data
        """
        return image.astype("float") / 255.0
