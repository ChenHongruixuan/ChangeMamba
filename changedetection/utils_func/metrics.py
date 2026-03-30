import numpy as np

from changedetection.evaluation import BinaryChangeEvaluator, DamageClassificationEvaluator, MultiClassEvaluator


class Evaluator:
    def __init__(self, num_class):
        self.num_class = num_class
        if num_class == 2:
            self._evaluator = BinaryChangeEvaluator()
        else:
            self._evaluator = MultiClassEvaluator(num_class)
            self._damage_evaluator = DamageClassificationEvaluator(num_class)

    @property
    def confusion_matrix(self):
        return self._evaluator.confusion_matrix

    def Pixel_Accuracy(self):
        if self.num_class == 2:
            return self._evaluator.compute().oa
        return self._evaluator.compute().oa

    def Pixel_Accuracy_Class(self):
        matrix = self.confusion_matrix
        acc = np.diag(matrix) / (matrix.sum(axis=1) + 1e-7)
        return np.nanmean(acc), acc

    def Pixel_Precision_Rate(self):
        return self._evaluator.compute().precision

    def Pixel_Recall_Rate(self):
        return self._evaluator.compute().recall

    def Pixel_F1_score(self):
        return self._evaluator.compute().f1

    def Damage_F1_socore(self):
        return self._damage_evaluator.compute().per_class_f1

    def Damage_F1_score(self):
        return self.Damage_F1_socore()

    def Mean_Intersection_over_Union(self):
        if self.num_class == 2:
            return self._evaluator.compute().iou
        return self._evaluator.compute().miou

    def Intersection_over_Union(self):
        if self.num_class == 2:
            return self._evaluator.compute().iou
        return self._evaluator.compute().iou_per_class

    def Kappa_coefficient(self):
        return self._evaluator.compute().kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        matrix = self.confusion_matrix
        freq = np.sum(matrix, axis=1) / (np.sum(matrix) + 1e-7)
        iu = np.diag(matrix) / (np.sum(matrix, axis=1) + np.sum(matrix, axis=0) - np.diag(matrix) + 1e-7)
        return (freq[freq > 0] * iu[freq > 0]).sum()

    def add_batch(self, gt_image, pre_image):
        self._evaluator.add_batch(gt_image, pre_image)
        if self.num_class != 2:
            self._damage_evaluator.add_batch(gt_image, pre_image)

    def reset(self):
        self._evaluator.reset()
        if self.num_class != 2:
            self._damage_evaluator.reset()
