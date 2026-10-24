from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, brier_score_loss, log_loss


def regression_metrics(y, pred) -> dict:
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    err = y - pred
    return {
        'mae': float(mean_absolute_error(y, pred)),
        'rmse': float(mean_squared_error(y, pred) ** 0.5),
        'medae': float(np.median(np.abs(err))),
        'p90_abs_error': float(np.quantile(np.abs(err), .90)),
        'p95_abs_error': float(np.quantile(np.abs(err), .95)),
        'r2': float(r2_score(y, pred)),
    }


def interval_metrics(y, lower, upper, alpha=.10) -> dict:
    y = np.asarray(y, float); lo = np.asarray(lower, float); hi = np.asarray(upper, float)
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    interval_score = width + (2/alpha) * (lo-y) * (y < lo) + (2/alpha) * (y-hi) * (y > hi)
    nominal = 1-alpha
    return {
        'coverage': float(np.mean(covered)),
        'coverage_error': float(abs(np.mean(covered) - nominal)),
        'mean_interval_width': float(np.mean(width)),
        'median_interval_width': float(np.median(width)),
        'interval_score': float(np.mean(interval_score)),
    }


def binary_metrics(y, p) -> dict:
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    out = {'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p))}
    if len(np.unique(y)) == 2:
        out['auroc'] = float(roc_auc_score(y, p))
    else:
        out['auroc'] = float('nan')
    return out
