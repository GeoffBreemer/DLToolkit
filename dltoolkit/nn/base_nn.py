"""Base class for all neural network architectures"""
import abc

class BaseNN(abc.ABC):
    # Title of the neural network
    _title = ""

    @property
    def title(self):
        return self._title

    # The Keras model
    _model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    # Shape of the input image
    _img_width = 0
    _img_height = 0
    _img_channels = 0

    # Number of classes to classify
    _num_classes = 0

    @abc.abstractmethod
    def build_model(self):
        pass
