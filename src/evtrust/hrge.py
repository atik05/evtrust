"""History-Reliability Gated Ensemble (HRGE).

Motivation (measured in the confirmatory study):
  * user-history features carry most of the predictive signal under a random
    split, yet contribute nothing when the driver is new (grouped ablation);
  * accuracy collapses when the driver is unseen (user-disjoint protocol).

A single model trained on all features has to serve both regimes with one set
of weights. HRGE instead trains two learners on the same rows:

  * f_hist  uses every feature, including the driver's history;
  * f_free  uses every feature except the driver's history;

and blends them with a gate that depends only on how much history the driver
has at prediction time:

      y_hat = w(g) * f_hist(x) + (1 - w(g)) * f_free(x),   g = history group.

The gate weights are chosen on the VALIDATION block, one weight per history
group, by minimising validation absolute error over a fixed grid. Groups with
too few validation rows fall back to the pooled weight. The test block never
influences the gate, and calibration rows are reserved for conformal intervals.

The gate is deliberately piecewise constant and monotone-free: it is easy to
inspect, it has at most four parameters, and it cannot overfit the way a
learned neural gate could on a few thousand validation rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .improvements import USER_HISTORY_COLUMNS, history_group

GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


def history_free_columns(numeric: list[str]) -> list[str]:
    return [c for c in numeric if c not in USER_HISTORY_COLUMNS]


@dataclass
class HRGE:
    """Two learners and a per-history-group blending gate."""
    f_hist: object
    f_free: object
    weights: dict = field(default_factory=dict)
    pooled_weight: float = 0.5
    min_rows: int = 200

    def gate(self, frame: pd.DataFrame) -> np.ndarray:
        groups = history_group(frame).to_numpy()
        return np.array([self.weights.get(str(g), self.pooled_weight) for g in groups], dtype=float)

    def predict_parts(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.f_hist.predict(frame), dtype=float),
                np.asarray(self.f_free.predict(frame), dtype=float))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ph, pf = self.predict_parts(frame)
        w = self.gate(frame)
        return w * ph + (1.0 - w) * pf


def _best_weight(y: np.ndarray, ph: np.ndarray, pf: np.ndarray) -> float:
    errs = [np.mean(np.abs(y - (w * ph + (1 - w) * pf))) for w in GRID]
    return float(GRID[int(np.argmin(errs))])


def fit_gate(model: HRGE, validation: pd.DataFrame, y_val: np.ndarray) -> HRGE:
    """Choose one blending weight per history group on the validation block."""
    ok = np.isfinite(y_val)
    v = validation.loc[ok]
    y = np.asarray(y_val, dtype=float)[ok]
    ph, pf = model.predict_parts(v)
    model.pooled_weight = _best_weight(y, ph, pf)
    groups = history_group(v).to_numpy()
    model.weights = {}
    for g in pd.unique(groups):
        sel = groups == g
        if int(sel.sum()) >= model.min_rows:
            model.weights[str(g)] = _best_weight(y[sel], ph[sel], pf[sel])
    return model


def fit_hrge(train: pd.DataFrame, y_train: np.ndarray, validation: pd.DataFrame, y_val: np.ndarray,
             numeric: list[str], categorical: list[str], fit_fn, min_rows: int = 200) -> HRGE:
    """Fit both learners with the same learner factory, then the gate.

    ``fit_fn(train, y, validation, y_val, numeric, categorical)`` must return an
    object with ``predict``. Using one factory keeps the comparison with the
    single full-feature learner fair: same algorithm, same hyperparameters,
    same rows, only the feature set and the gate differ.
    """
    free_numeric = history_free_columns(numeric)
    f_hist = fit_fn(train, y_train, validation, y_val, numeric, categorical)
    f_free = fit_fn(train, y_train, validation, y_val, free_numeric, categorical)
    model = HRGE(f_hist=f_hist, f_free=f_free, min_rows=min_rows)
    return fit_gate(model, validation, y_val)
