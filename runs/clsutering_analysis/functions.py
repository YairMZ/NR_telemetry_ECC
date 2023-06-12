import numpy as np
from sklearn.metrics import confusion_matrix


def relabel(predicted_labels, n_classes, real_labels=None):
    """Relabels the predicted labels so that the predicted labels are the same as the real labels.

    :param predicted_labels: array of predicted labels
    :param n_classes: number of classes
    :param real_labels: array of real labels. If None, then the real labels are assumed to be in consecutive order.
    :return:
    """
    if real_labels is None:
        predictions = np.zeros((n_classes, len(predicted_labels)//n_classes))
    else:
        predictions = -np.ones((n_classes, len(predicted_labels)))
    for real_label in range(n_classes):
        if real_labels is None:
            predictions[real_label] = predicted_labels[real_label::n_classes]
        else:
            p = predicted_labels[real_labels == real_label]
            predictions[real_label, :len(p)] = p
    # now find which predicted cluster has the most samples of each real label
    # and assign that cluster to that real label
    prediction_counts = np.zeros((n_classes, n_classes))
    for real_label in range(n_classes):
        for predicted_label in range(n_classes):
            prediction_counts[real_label, predicted_label] = np.sum(predictions[real_label] == predicted_label)
    # find the largest value in prediction_counts. The column index label should be mapped to the row index label.
    # if there are multiple values that are the same, it doesn't matter which one we choose.
    reduced_prediction_counts = prediction_counts.copy()
    labels_map = {}
    for _ in range(n_classes-1):
        true_label, predicted_label = np.unravel_index(reduced_prediction_counts.argmax(), reduced_prediction_counts.shape)
        labels_map[predicted_label] = true_label
        reduced_prediction_counts[:, predicted_label] = -1
    tmp, remaining_predicted_label = np.unravel_index(reduced_prediction_counts.argmax(), reduced_prediction_counts.shape)
    remaining_true_label = np.setdiff1d(np.arange(n_classes), np.array(list(labels_map.values())))[0]
    labels_map[remaining_predicted_label] = remaining_true_label

    relabeled = np.ones_like(predicted_labels) * 255
    for predicted_label, real_label in labels_map.items():
        relabeled[predicted_labels == predicted_label] = real_label
    return relabeled, labels_map
