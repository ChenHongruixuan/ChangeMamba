from dataclasses import dataclass

import numpy as np

from .common import ConfusionMatrixEvaluator, safe_divide


@dataclass(frozen=True)
class MultiClassMetrics:
    oa: float
    iou_per_class: np.ndarray
    miou: float
    kappa: float


class MultiClassEvaluator(ConfusionMatrixEvaluator):
    def compute(self):
        matrix = self.confusion_matrix
        iou_per_class = safe_divide(
            np.diag(matrix),
            np.sum(matrix, axis=1) + np.sum(matrix, axis=0) - np.diag(matrix),
        )
        oa = safe_divide(np.trace(matrix), matrix.sum())
        miou = np.nanmean(iou_per_class)
        expected_accuracy = np.sum(
            safe_divide(np.sum(matrix, axis=0), matrix.sum()) * safe_divide(np.sum(matrix, axis=1), matrix.sum())
        )
        kappa = safe_divide(oa - expected_accuracy, 1 - expected_accuracy)
        return MultiClassMetrics(
            oa=float(oa),
            iou_per_class=iou_per_class.astype(np.float64),
            miou=float(miou),
            kappa=float(kappa),
        )
