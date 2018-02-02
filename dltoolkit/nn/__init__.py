"""Implementations of various neural network architectures"""
from .lenet import LeNetNN, LENET_IMG_CHANNELS, LENET_IMG_HEIGHT, LENET_IMG_WIDTH, LENET_NUM_CLASSES
from .miniVGGnet import MiniVGGNN, MINIVGGNET_IMG_CHANNELS
from .alexnet import AlexNetNN, ALEX_IMG_CHANNELS, ALEX_IMG_HEIGHT, ALEX_IMG_WIDTH
from .shallownet import ShallowNetNN, SHALLOWNET_IMG_CHANNELS, SHALLOWNET_IMG_HEIGHT, SHALLOWNET_IMG_WIDTH
from .nvidia import NVIDIA_NN
from .generic import CIFAR10_CLASSES, ANIMALS_CLASSES, FLOWERS17_CLASSES
