"""Implementations of various neural network architectures"""
from .lenet import LeNetNN, LENET_IMG_CHANNELS, LENET_IMG_HEIGHT, LENET_IMG_WIDTH
from .miniVGG import MiniVGGNN
from .alexnet import AlexNetNN, ALEX_IMG_CHANNELS, ALEX_IMG_HEIGHT, ALEX_IMG_WIDTH
from .shallownet import ShallowNetNN
from .nvidia import NVIDIA_NN
