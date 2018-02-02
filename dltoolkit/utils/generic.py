"""Generic utility functions and constants"""
import argparse
import numpy as np


def str2bool(v):
    """Attempt to convert a string to a boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ranked_accuracy(predictions, labels):
    """Return the rank 1 and rank 5 accuracy"""
    rank1 = 0
    rank5 = 0

    for (pred, lbl) in zip(predictions, labels):
        pred = np.argsort(pred)[::-1]

        if lbl in pred[:5]:
            rank5+=1

        if lbl == pred[0]:
            rank1+=1

    rank1 /= float(len(labels))
    rank5 /= float(len(labels))

    return (rank1, rank5)