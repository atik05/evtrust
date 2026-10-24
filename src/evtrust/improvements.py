"""Domain-aware extensions motivated by the measurements of the main study.

Three additions, each answering something the confirmatory run revealed:

1. **History dropout training.** The ablation shows that user-history features
   carry most of the signal under a random split and none of it when the driver
   is new. Masking those features for a fraction of training rows forces the
   model to keep a usable history-free pathway.
2. **History-conditional conformal calibration.** Conditional coverage is far
   below nominal for drivers with little history, so the calibration quantile is
   fitted per history group instead of once over the pool.
3. **Published-style features.** User aggregates computed over the whole dataset,
   the way several published pipelines appear to compute them, quantify how much
   of the reported accuracy in the literature comes from that choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import split_conformal_calibrate

USER_HISTORY_COLUMNS = ['user_prior_count', 'user_prior_mean_duration',
                        'user_prior_median_duration', 'user_prior_mean_energy']

HISTORY_BINS = [(-0.5, 0.5, '0'), (0.5, 2.5, '1-2'), (2.5, 9.5, '3-9'), (9.5, np.inf, '10+')]


def history_group(frame: pd.DataFrame) -> pd.Series:
    """Label each row by how much history its driver had at prediction time."""
    counts = pd.to_numeric(frame.get('user_prior_count'), errors='coerce').fillna(0).to_numpy(float)
    out = np.empty(len(counts), dtype=object)
    for lo, hi, name in HISTORY_BINS:
        out[(counts > lo) & (counts <= hi)] = name
    return pd.Series(out, index=frame.index, dtype=object)


def apply_history_dropout(train: pd.DataFrame, rate: float, seed: int,
                          columns: list[str] | None = None) -> pd.DataFrame:
    """Mask user-history features on a random share of TRAINING rows only.

    The masked value is NaN, which is what a genuinely new driver looks like at
    prediction time, so the model learns to operate without that pathway.
    """
    if rate <= 0:
        return train
    cols = [c for c in (columns or USER_HISTORY_COLUMNS) if c in train.columns]
    if not cols:
        return train
    out = train.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(out)) < rate
    idx = out.index[mask]
    for c in cols:
        out.loc[idx, c] = np.nan
    if 'user_prior_count' in cols:
        out.loc[idx, 'user_prior_count'] = 0.0
    return out


def mondrian_regression_calibrate(y_cal: np.ndarray, pred_cal: np.ndarray, groups_cal: pd.Series,
                                  alpha: float = .10, min_per_group: int = 50) -> dict:
    """One conformal quantile per history group, with a pooled fallback.

    A group with too few calibration points cannot support its own finite-sample
    quantile, so it falls back to the pooled quantile rather than producing an
    unreliable one.
    """
    pooled = split_conformal_calibrate(y_cal, pred_cal, alpha)
    table = {'__pooled__': float(pooled)}
    g = pd.Series(groups_cal).to_numpy()
    for name in pd.unique(g):
        sel = g == name
        if int(sel.sum()) >= min_per_group:
            table[str(name)] = float(split_conformal_calibrate(y_cal[sel], pred_cal[sel], alpha))
    return table


def mondrian_regression_interval(pred: np.ndarray, groups: pd.Series, table: dict
                                 ) -> tuple[np.ndarray, np.ndarray]:
    q = np.array([table.get(str(name), table['__pooled__']) for name in pd.Series(groups).to_numpy()],
                 dtype=float)
    return pred - q, pred + q


def coverage_by_group(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      groups: pd.Series) -> pd.DataFrame:
    covered = (y >= lower) & (y <= upper)
    z = pd.DataFrame({'group': pd.Series(groups).to_numpy(), 'covered': covered,
                      'width': upper - lower})
    out = z.groupby('group', dropna=False).agg(n=('covered', 'size'), coverage=('covered', 'mean'),
                                               mean_width=('width', 'mean')).reset_index()
    return out


def add_published_style_features(frame: pd.DataFrame) -> pd.DataFrame:
    """User aggregates computed over the WHOLE dataset, including the test rows.

    This reproduces a common published setup. It is deliberately leaky and exists
    only to quantify the size of that effect; it must never be used for a result
    that is reported as valid.
    """
    out = frame.copy()
    if 'user_id' not in out.columns:
        return out
    uid = out['user_id'].astype(str)
    known = uid.notna() & (uid != 'None') & (uid != '<NA>') & (uid.str.strip() != '')
    for col, src in [('user_prior_mean_duration', 'duration_hours'),
                     ('user_prior_median_duration', 'duration_hours'),
                     ('user_prior_mean_energy', 'kwh_delivered')]:
        if src not in out.columns:
            continue
        agg = 'median' if 'median' in col else 'mean'
        values = out.loc[known].groupby(uid[known])[src].transform(agg)
        out.loc[known, col] = values
    if 'user_prior_count' in out.columns:
        out.loc[known, 'user_prior_count'] = uid[known].map(uid[known].value_counts()).to_numpy()
    return out
