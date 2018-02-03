"""Base class for all convolutional neural network architectures"""
from abc import abstractmethod
from dltoolkit.nn.base_nn import BaseNN


class BaseConvNN(BaseNN):
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
