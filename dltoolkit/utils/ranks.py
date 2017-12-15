import numpy as np


def ranked_accuracy(predictions, labels):
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
