"""Generic utility functions and constants"""
from keras.utils import plot_model
import argparse, datetime
import numpy as np
import matplotlib.pyplot as plt

CIFAR10_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ANIMALS_CLASS_NAMES = ["cats", "dogs", "panda"]
FLOWERS17_CLASS_NAMES = ["bluebell", "buttercup", "coltsfoot", "cowslip", "crocus", "daffodil", "daisy", "dandelion",
                     "fritillary", "iris", "lilyvalley", "pansy", "snowdrop", "sunflower", "tigerlily",
                     "tulip", "windflower"]


def str2bool(v):
    """Attempt to convert a string to a boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_history(hist, epochs, show=True, save_path=None, time_stamp=False):
    """
    Plot Keras training results to a figure and display and/or save it
    :param hist: a Keras History object
    :param epochs: number of epochs used
    :param show: True to show the figure, False if not
    :param save_path: full path to save the figure to, None if no saving required
    """
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

    if show:
        plt.show()

    if save_path is not None:
        save_path = save_path + "_training"
        if time_stamp:
            current_dt = datetime.datetime.now()
            save_path = save_path + "_{}_{}".format(current_dt.strftime("%Y%m%d"),
            current_dt.strftime("%H%M%S"))

        save_path = save_path + ".png"
        plt.savefig(save_path)

    plt.close()


def save_model_architecture(model, save_path, show_shapes=True):
    """Save a picture of the model architecture to disk"""
    plot_model(model, to_file=save_path + "_architecture.png", show_shapes=show_shapes)
