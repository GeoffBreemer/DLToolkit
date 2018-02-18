"""Basic functions"""
import numpy as np
from keras import backend as K


def softmax(x):
    """Apply softmax to numpy vector x"""
    x_exp = np.exp(x-np.max(x))
    return x_exp/x_exp.sum(axis=0)


def sigmoid(x):
    """Apply the sigmoid function to numpy vector x"""
    return 1/(1+np.exp(-x))


def cosine_similarity(u, v):
    """
    Calculate the cosine similarity between two numpy vectors
    :param u: first vector
    :param v: second vector
    :return: cosine similarity
    """
    norm_u = np.sqrt(np.sum(np.square(u)))
    norm_v = np.sqrt(np.sum(np.square(v)))

    return np.dot(u, v) / (norm_u * norm_v)


def clip(gradients, max_value):
    """
    Clip a dictionary of numpy arrays to a maximum value, clipping is done *in-place*
    :param gradients: dictionary of numpy arrays
    :param max_value: value to clip array elements to
    """
    for key, value in gradients.items():
        np.clip(value, -max_value, max_value, out=gradients[key])


if __name__ == '__main__':
    grads = {"dx": np.array([1,2, 3]),
             "dy": np.array([-1,-2, -3]),
             "dz": np.array([3,4, -5])}

    clip(grads, 2)
    print(grads)


def dice_coef(y_true, y_pred):
    smooth = 1.0

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)
