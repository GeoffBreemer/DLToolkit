"""Utility functions for visualisation"""
from keras.utils import plot_model
from sklearn.metrics import classification_report
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


def plot_training_history(hist, epochs, show=True, save_path=None, time_stamp=False, metric='acc'):
    """
    Plot Keras training results to a figure and display and/or save it
    :param hist: a Keras History object
    :param epochs: number of epochs used
    :param show: True to show the figure, False if not
    :param save_path: full path to save the figure to, None if no saving required
    :param time_stamp: whether to add a date/time stamp to the file name
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    # plt.plot(np.arange(0, epochs), hist.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, epochs), hist.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, epochs), hist.history["val_acc"], label="val_acc")

    # plt.figure(figsize=[8, 6])
    plt.plot(hist.history['loss'], 'r--', linewidth=3.0, label="train_loss")
    plt.plot(hist.history['val_loss'], 'b--', linewidth=3.0, label="val_loss")
    plt.plot(hist.history[metric], 'r', linewidth=3.0, label="train_"+metric)
    plt.plot(hist.history["val_"+metric], 'b', linewidth=3.0, label="val_"+metric)
    # plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    # plt.xlabel('Epochs ', fontsize=16)
    # plt.ylabel('Loss', fontsize=16)
    # plt.title('Loss Curves', fontsize=16)

    plt.title("Loss/"+metric)
    plt.xlabel("Epoch")
    plt.ylabel("Loss/"+metric)
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


def plot_roc_curve(ground_truth_imgs, predicted_imgs, show=True, save_path=None, time_stamp=False):
    y_true = np.reshape(ground_truth_imgs, (-1, 1))
    y_true = (y_true / 255).astype(np.uint8)

    y_scores = np.reshape(predicted_imgs, (-1, 1))
    y_scores = (y_scores / 255).astype(np.uint8)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    AUC_ROC = roc_auc_score(y_true, y_scores, average='weighted')

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, '-', label='Area Under the Curve (AUC) = %0.4f' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")

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
    y_true = np.reshape(ground_truth_imgs, (-1, 1))
    y_true = (y_true / 255).astype(np.uint8)
    y_scores = np.reshape(predictions, (-1, num_classes))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores[:, 1], pos_label=1)

    precision = np.fliplr([precision])[0]
    recall = np.fliplr([recall])[0]

    AUC_prec_rec = np.trapz(precision, recall)

    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")

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
    y_true = np.reshape(ground_truth_imgs, (-1, 1))
    y_pred = np.reshape(predicted_imgs, (-1, 1))

    y_true = (y_true / 255).astype(np.uint8)
    y_pred = (y_pred / 255).astype(np.uint8)

    prec = precision_score(y_true, y_pred, pos_label=1, average='binary')
    print("Precision: {:.2f}".format(prec))

    rec = recall_score(y_true, y_pred, pos_label=1, average='binary')
    print("   Recall: {:.2f}".format(rec))

    print(classification_report(y_true, y_pred, target_names=("background", "vessel")))
