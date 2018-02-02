"""Various utility functions and constants"""
from .constants import CIFAR10_CLASS_NAMES, ANIMALS_CLASS_NAMES, FLOWERS17_CLASS_NAMES
from .generic import str2bool, ranked_accuracy
from .callback import TrainingMonitor
from .visual import plot_history, save_model_architecture
