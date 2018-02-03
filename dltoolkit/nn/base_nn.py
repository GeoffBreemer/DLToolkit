"""Base class for all neural network architectures"""
from abc import ABC, abstractmethod

class BaseNN(ABC):
    _title = ""
    @property
    def title(self):
        # A brief title of the neural network that can be used for logging purposes
        return self._title

    _model = None
    @property
    def model(self):
        # The Keras model
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    # Input shape dimensions
    _img_width = -1
    _img_height = -1
    _img_channels = -1

    # Number of classes to classify
    _num_classes = -1

    @abstractmethod
    def build_model(self):
        """Build the Keras network and assign it to _model"""
        pass

    def __str__(self):
        return self._title + " architecture, input shape: {} x {} x {}, {} classes".format(self._img_width,
                                                                                           self._img_height,
                                                                                           self._img_channels,
                                                                                           self._num_classes)
