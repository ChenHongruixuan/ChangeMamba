import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from changedetection.utils_func import eval_segm as seg_acc


@dataclass(frozen=True)
class SemanticChangeMetrics:
    kappa: float
    fscd: float
    miou: float
    sek: float
    oa: float


class AverageMeter:
    def __init__(self):
        self.values = []

    @property
    def avg(self):
        if not self.values:
            return 0.0
        return float(np.mean(self.values))

    def update(self, value):
        self.values.append(float(value))


def accuracy(pred, label, ignore_zero=False):
    valid = label >= 0
    if ignore_zero:
        valid = label > 0
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        return 0.0
    po = np.diag(hist).sum() / hist.sum()
    pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
    if pe == 1:
        return 0.0
    return float((po - pe) / (1 - pe))


def SCDD_eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        label_array = np.array(label)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set(range(num_class))), "unrecognized label number"
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()

    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist) + 1e-10)
    iou_fg = iu[1]
    iou_mean = (iu[0] + iu[1]) / 2
    sek = (kappa_n0 * math.exp(iou_fg)) / math.e

    pixel_sum = hist.sum()
    change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    sc_tp = np.diag(hist[1:, 1:]).sum()
    sc_precision = sc_tp / (change_pred_sum + 1e-10)
    sc_recall = sc_tp / (change_label_sum + 1e-10)
    fscd = stats.hmean([sc_precision, sc_recall]) if sc_precision > 0 and sc_recall > 0 else 0.0
    return float(kappa_n0), float(fscd), float(iou_mean), float(sek)


def SCDD_eval(pred, label, num_class):
    return SCDD_eval_all([pred], [label], num_class)[1:]


def FWIoU(pred, label, bn_mode=False, ignore_zero=False):
    if bn_mode:
        pred = pred >= 0.5
        label = label >= 0.5
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    return seg_acc.frequency_weighted_IU(pred, label)


class SemanticChangeEvaluator:
    def __init__(self, num_class=37):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.preds = []
        self.labels = []
        self.acc_meter = AverageMeter()

    def add_batch(self, pred_scd, label_scd):
        acc, _ = accuracy(pred_scd, label_scd)
        self.preds.append(np.asarray(pred_scd))
        self.labels.append(np.asarray(label_scd))
        self.acc_meter.update(acc)

    def compute(self):
        kappa_n0, fscd, iou_mean, sek = SCDD_eval_all(self.preds, self.labels, self.num_class)
        return SemanticChangeMetrics(
            kappa=float(kappa_n0),
            fscd=float(fscd),
            miou=float(iou_mean),
            sek=float(sek),
            oa=float(self.acc_meter.avg),
        )
