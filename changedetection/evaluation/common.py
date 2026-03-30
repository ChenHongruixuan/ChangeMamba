import numpy as np


def safe_divide(numerator, denominator, eps=1e-7):
    return numerator / (denominator + eps)


class ConfusionMatrixEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def add_batch(self, labels, predictions):
        labels = np.asarray(labels)
        predictions = np.asarray(predictions)
        assert labels.shape == predictions.shape

        mask = (labels >= 0) & (labels < self.num_classes)
        encoded = self.num_classes * labels[mask].astype(np.int64) + predictions[mask].astype(np.int64)
        counts = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)
