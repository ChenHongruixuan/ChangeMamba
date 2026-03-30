from dataclasses import dataclass

import numpy as np

from .common import ConfusionMatrixEvaluator, safe_divide


@dataclass(frozen=True)
class DamageMetrics:
    per_class_f1: np.ndarray
    harmonic_mean_f1: float


class DamageClassificationEvaluator(ConfusionMatrixEvaluator):
    def compute(self):
        matrix = self.confusion_matrix
        true_positives = np.diag(matrix)[1:]
        false_negatives = np.sum(matrix, axis=1)[1:] - true_positives
        false_positives = np.sum(matrix, axis=0)[1:] - true_positives

        precisions = safe_divide(true_positives, true_positives + false_positives)
        recalls = safe_divide(true_positives, true_positives + false_negatives)
        f1_scores = safe_divide(2 * precisions * recalls, precisions + recalls)
        harmonic_mean = len(f1_scores) / np.sum(1.0 / np.clip(f1_scores, 1e-7, None))

        return DamageMetrics(
            per_class_f1=f1_scores.astype(np.float64),
            harmonic_mean_f1=float(harmonic_mean),
        )
