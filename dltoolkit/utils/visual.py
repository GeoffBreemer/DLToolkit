"""Utility functions for visualisation"""
from keras.utils import plot_model
from sklearn.metrics import classification_report
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def plot_training_history(hist, show=True, save_path=None, time_stamp=False, metric="acc"):
    """
    Plot Keras training results to a figure and display and/or save it
    :param hist: a Keras History object
    :param show: True to show the figure, False if not
    :param save_path: full path to save the figure to, None if no saving required
    :param time_stamp: whether to add a date/time stamp to the file name
    """
    import seaborn as sns
    import datetime

    # Don't include validation results if there aren't any (assumes only one loss/metric pair is used)
    includes_val = len(hist.history) > 2

    # Set the style
    sns.set_style("whitegrid")
    sns.color_palette("viridis")
    plt.style.use("thesis_style.use")

    if metric == "acc":
        ylabel = "Accuracy"
    elif metric == "dice_coef":
        ylabel = "Dice coefficient"

    # Create a dual axis graph
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax2 = ax1.twinx()

    # Plot the metric
    p1 = ax1.plot(hist.history[metric], '-', linewidth=3.0, label="Training "+ylabel)
    if includes_val:
        p2 = ax1.plot(hist.history["val_"+metric], '-', linewidth=3.0, label="Validation "+ylabel)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(ylabel)

    # Plot the loss
    p3 = ax2.plot(hist.history['loss'], '--', linewidth=3.0, label="Training loss")
    if includes_val:
        p4 = ax2.plot(hist.history['val_loss'], '--', linewidth=3.0, label="Validation loss")
    ax2.set_ylabel('Loss')

    # Combine into one legend
    if includes_val:
        p = p1+p2+p3+p4
        min_ix = np.argmin(hist.history["val_loss"])
        min_val = hist.history["val_loss"][min_ix]
        min_text = "Lowest validation loss"
    else:
        p = p1+p3
        min_ix = np.argmin(hist.history["loss"])
        min_val = hist.history["loss"][min_ix]
        min_text = "Lowest loss"

    labs = [l.get_label() for l in p]
    ax1.legend(p, labs, loc="center right")

    # Mark the epoch with the lowest (val_)loss
    ax2.plot([min_ix], [min_val], 'ko', markersize=15, alpha=.5)
    ax2.plot([min_ix], [min_val], 'ko', markersize=15, markerfacecolor="None")
    ax2.annotate(min_text, xy=(min_ix, min_val),
            xytext=(0.5, 0.05), textcoords='axes fraction',
            arrowprops=dict(facecolor='black'),#, shrink=0.15),
            horizontalalignment='right', verticalalignment='top')

    # Align grids
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    ax2.grid(None)

    # Set ticks
    major_ticks = np.arange(0, len(hist.history[metric]), 1)
    ax1.set_xticks(major_ticks)

    plt.title("Training loss/" + ylabel + " by epoch")

    if show:
        plt.show()

    if save_path is not None:
        save_path = save_path + "_training"
        if time_stamp:
            current_dt = datetime.datetime.now()
            save_path = save_path + "_{}_{}".format(current_dt.strftime("%Y%m%d"),
            current_dt.strftime("%H%M%S"))

        save_path = save_path + ".png"
        fig.savefig(save_path)

    plt.close()


def model_performance(nn, test_set, test_labels, class_names, batch_size):
    """Make predictions on a test set and print the classification report, return the predictions"""
    pred = nn.model.predict(x=test_set, batch_size=batch_size)
    print(classification_report(test_labels.argmax(axis=-1), pred.argmax(axis=1), target_names=class_names))

    return pred


def visualise_results(test_set, test_labels, pred_labels, class_names):
    # Visualise a few random  images, increase their size for better visualisation
    idxs = np.random.randint(0, len(test_set), size=(10,))
    for (i, image) in enumerate(test_set[idxs]):
        print("Image {} is a {} predicted to be a {}".format(i + 1,
                                                             class_names[test_labels[idxs[i]].argmax(axis=0)],
                                                             class_names[pred_labels[idxs[i]].argmax(axis=0)]))
        image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Image", image)
        cv2.waitKey(0)


def plot_roc_curve(ground_truth_imgs, predicted_scores_pos, show=True, save_path=None, time_stamp=False):
    """
    Plot the ROC curve.
    :param ground_truth_imgs: ground truth images
    :param predicted_scores_pos: predicted scores for the positive class (i.e. NOT for both classes, and not the image)
    :param show: True to show the image, False otherwise
    :param save_path: full path to save the image to, None to not save the image
    :param time_stamp: True to add a timestamp to the saved file, False otherwise
    :return: N/A
    """
    y_true = np.reshape(ground_truth_imgs, (-1, 1))
    y_true = (y_true / 255).astype(np.uint8)
    y_scores = np.reshape(predicted_scores_pos, (-1, 1))

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    AUC_ROC = roc_auc_score(y_true, y_scores, average='weighted')

    print("\nArea under ROC curve: {:0.4f}".format(AUC_ROC))

    # Set the style
    sns.set_style("whitegrid")
    sns.color_palette("viridis")
    plt.style.use("thesis_style.use")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(fpr, tpr, '-', label="Area Under the Curve (AUC) = {:0.4f}".format(AUC_ROC))

    plt.title("Receiver Operator Curve (ROC)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower center")

    if show:
        plt.show()

    if save_path is not None:
        save_path = save_path + "_roc_curve"
        if time_stamp:
            current_dt = datetime.datetime.now()
            save_path = save_path + "_{}_{}".format(current_dt.strftime("%Y%m%d"),
            current_dt.strftime("%H%M%S"))

        save_path = save_path + ".png"
        fig.savefig(save_path)

    plt.close()


def plot_precision_recall_curve(ground_truth_imgs, predictions, num_classes, show=True, save_path=None, time_stamp=False):
    """
    Plot the precision/recall curve.
    :param ground_truth_imgs: ground truth images
    :param predictions: prediction scores (i.e. NOT the images)
    :param num_classes: number of classes
    :param show: True to show the image, False otherwise
    :param save_path: full path to save the image to, None to not save the image
    :param time_stamp: True to add a timestamp to the saved file, False otherwise
    :return: N/A
    """
    y_true = np.reshape(ground_truth_imgs, (-1, 1))
    y_true = (y_true / 255).astype(np.uint8)
    y_scores = np.reshape(predictions, (-1, num_classes))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores[:, 1], pos_label=1)

    precision = np.fliplr([precision])[0]
    recall = np.fliplr([recall])[0]

    AUC_prec_rec = np.trapz(precision, recall)

    print("\nArea under Precision-Recall curve: {:0.4f}".format(AUC_prec_rec))

    # Set the style
    sns.set_style("whitegrid")
    sns.color_palette("viridis")
    plt.style.use("thesis_style.use")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(recall, precision, '-', label="Area Under the Curve (AUC) = {:0.4f}".format(AUC_prec_rec))

    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower center")

    if show:
        plt.show()

    if save_path is not None:
        save_path = save_path + "_precision_recall_curve"
        if time_stamp:
            current_dt = datetime.datetime.now()
            save_path = save_path + "_{}_{}".format(current_dt.strftime("%Y%m%d"),
            current_dt.strftime("%H%M%S"))

        save_path = save_path + ".png"
        fig.savefig(save_path)

    plt.close()


def print_confusion_matrix(ground_truth_imgs, predicted_imgs):
    """
    Print the confusion matrix.
    :param ground_truth_imgs: ground truth images
    :param predicted_imgs: predicted images (i.e. NOT the scores)
    :return:
    """
    y_true = np.reshape(ground_truth_imgs, (-1, 1))
    y_pred = np.reshape(predicted_imgs, (-1, 1))

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n{}".format(conf_matrix))
    print("   Row/Column: Negative Class, Positive Class, i.e.:")
    print("             TN FP")
    print("             FN TP")

    (tn, fp, fn, tp) = conf_matrix.ravel()
    print("\n True Negative (TN): {}".format(tn))
    print(" True Positive (TP): {}".format(tp))
    print("False Negative (FN): {}".format(fn))
    print("False Positive (FP): {}".format(fp))


    print("\nBlood vessel precision: {:.2f}".format((tp / (tp + fp))))
    print("   Blood vessel recall: {:.2f}".format((tp / (tp + fn))))

    return conf_matrix, conf_matrix.ravel()


def print_classification_report(ground_truth_imgs, predicted_imgs):
    """
    Print the classification report
    :param ground_truth_imgs: ground truth images
    :param predicted_imgs: predicted images (i.e. NOT the scores)
    :return:
    """
    y_true = np.reshape(ground_truth_imgs, (-1, 1))
    y_pred = np.reshape(predicted_imgs, (-1, 1))

    y_true = (y_true / 255).astype(np.uint8)
    y_pred = (y_pred / 255).astype(np.uint8)

    prec = precision_score(y_true, y_pred, pos_label=1, average='binary')
    print("Precision: {:.2f}".format(prec))

    rec = recall_score(y_true, y_pred, pos_label=1, average='binary')
    print("   Recall: {:.2f}".format(rec))

    print(classification_report(y_true, y_pred, target_names=("background", "vessel")))
