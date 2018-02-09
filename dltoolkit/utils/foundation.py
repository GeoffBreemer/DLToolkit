"""Basic functions"""
import numpy as np

def softmax(x):
    """Apply softmax to numpy vector x"""
    x_exp = np.exp(x-np.max(x))
    return x_exp/x_exp.sum(axis=0)


def sigmoid(x):
    """Apply the sigmoid function to numpy vector x"""
    return 1/(1+np.exp(-x))
