import numpy as np
from sklearn.metrics import confusion_matrix


def stats(actual_labels=None, predicted_labels=None, cm=None):
    if cm is None:
        cm = confusion_matrix(actual_labels, predicted_labels)
    no_true_positives = np.sum(cm, axis=1) == 0
    n_classes = cm.shape[0]
    _recall = np.diag(cm) / np.sum(cm, axis=1)
    if np.sum(cm, axis=0).all() > 0:  # if there are some predictions for all classes
        _precision = np.diag(cm) / np.sum(cm, axis=0)
    else:  # some classes have no predictions
        _precision = np.zeros(n_classes)
        for label in range(n_classes):
            if np.sum(cm[:, label]) == 0:  # no predictions for this class
                _precision[label] = 1 if no_true_positives[label] is True else 0
                # if there are no true positives, then precision is 1, else 0
            else:  # there are predictions for this class
                _precision[label] = cm[label,label] / np.sum(cm[:, label])
    denominator = _precision + _recall
    if denominator.all() > 0:  # if there are some correct predictions for all classes
        _f1 = 2 * (_precision * _recall) / denominator
    else:  # some classes have no correct predictions
        _f1 = np.zeros(cm.shape[0])
        for label in range(n_classes):
            if denominator[label] > 0: # if there are some correct predictions for this class
                _f1[label] = 2 * (_precision[label] * _recall[label]) / denominator[label]
            else:  # if there are no correct predictions for this class
                _f1[label] = 1 if no_true_positives[label] is True else 0  # if there are no true positives, then 1, else 0
    _accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return cm, _recall, _precision, _f1, _accuracy

def ber(ground_truth: np.ndarray, predicted: np.ndarray):
    """Calculates the Bit Error Rate (BER) for the given ground truth and predicted bits.

    :param ground_truth: array of bits
    :param predicted: array of bits
    :return: BER
    """
    return np.sum(np.abs(ground_truth - predicted)) / len(ground_truth)

__all__ = ['stats', 'ber']
