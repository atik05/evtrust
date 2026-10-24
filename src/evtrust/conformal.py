from __future__ import annotations
import numpy as np


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float('nan')
    q = np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)
    q = min(1.0, q)
    return float(np.quantile(scores, q, method='higher'))


def split_conformal_calibrate(y_cal: np.ndarray, pred_cal: np.ndarray, alpha: float = 0.10) -> float:
    return finite_sample_quantile(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)), alpha)


def split_conformal_interval(pred: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    return pred - qhat, pred + qhat


def cqr_calibrate(y_cal: np.ndarray, lower_cal: np.ndarray, upper_cal: np.ndarray, alpha: float = 0.10) -> float:
    y = np.asarray(y_cal, dtype=float)
    lo = np.asarray(lower_cal, dtype=float)
    hi = np.asarray(upper_cal, dtype=float)
    scores = np.maximum(lo - y, y - hi)
    return finite_sample_quantile(scores, alpha)


def cqr_interval(lower: np.ndarray, upper: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(lower) - qhat, np.asarray(upper) + qhat


def mondrian_binary_calibrate(y_cal: np.ndarray, p_free_cal: np.ndarray, alpha: float = .10) -> dict[int, float]:
    y = np.asarray(y_cal, dtype=int)
    p = np.clip(np.asarray(p_free_cal, dtype=float), 0, 1)
    probs = np.column_stack([1-p, p])
    out = {}
    for cls in (0, 1):
        mask = y == cls
        scores = 1 - probs[mask, cls]
        out[cls] = finite_sample_quantile(scores, alpha)
    return out


def mondrian_prediction_set(p_free: np.ndarray, qhat: dict[int, float]) -> np.ndarray:
    p = np.clip(np.asarray(p_free, dtype=float), 0, 1)
    probs = np.column_stack([1-p, p])
    include0 = (1 - probs[:, 0]) <= qhat[0]
    include1 = (1 - probs[:, 1]) <= qhat[1]
    return np.column_stack([include0, include1])
