"""Generic utility functions"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def str2bool(v):
    """Determine whether a string means True or False"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_history(hist, epochs):
    """Plot Keras training results"""
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), hist.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), hist.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), hist.history["val_acc"], label="val_acc")
    plt.title("Loss/accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/accuracy")
    plt.legend()
    plt.show()
