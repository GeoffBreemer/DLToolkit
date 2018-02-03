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

    @abstractmethod
    def build_model(self):
        """Build the Keras network and assign it to _model"""
        pass

    def __str__(self):
        return self._title + " architecture"
