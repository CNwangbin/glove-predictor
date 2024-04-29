import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_curve, auc, roc_curve


def compute_metrics(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    y_pred_hard = np.where(y_pred > 0.5, 1, 0)
    acc = accuracy_score(y_true, y_pred_hard)
    mcc = matthews_corrcoef(y_true, y_pred_hard)
    fpr, tpr, t = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # 从FPR和TPR计算特异度和敏感度
    specificity = 1 - fpr
    sensitivity = tpr
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    # 选取thresholds为0.5时的precision和recall，计算F1
    idx = np.argmin(np.abs(thresholds - 0.5))
    precision = precision[idx]
    recall = recall[idx]
    sensitivity = sensitivity[np.argmin(np.abs(t - 0.5))]
    specificity = specificity[np.argmin(np.abs(t - 0.5))]
    f1 = 2 * precision * recall / (precision + recall)
    metric_dict = {'acc': acc, 'mcc': mcc, 'roc_auc': roc_auc, 'aupr': aupr, 'f1':f1, 'spec': specificity, 'sens': sensitivity, 'p': precision, 'r': recall}
    return metric_dict


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)