from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Split:
    train: np.ndarray
    calibration: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    name: str
    metadata: dict


def _partition_indices(idx: np.ndarray, val_frac: float, cal_frac: float, test_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generic four-way partition preserving the historical return order train/cal/val/test.

    Chronological protocols deliberately reorder the two middle blocks to
    train -> validation -> calibration -> test so model selection occurs
    before conformal calibration.
    """
    n = len(idx)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_cal = max(1, int(round(n * cal_frac)))
    n_train = max(1, n - n_test - n_val - n_cal)
    train = idx[:n_train]
    cal = idx[n_train:n_train+n_cal]
    val = idx[n_train+n_cal:n_train+n_cal+n_val]
    test = idx[n_train+n_cal+n_val:]
    return train, cal, val, test


def random_split(df: pd.DataFrame, seed: int, val_frac=.1, cal_frac=.1, test_frac=.2) -> Split:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(df.index.to_numpy())
    train, cal, val, test = _partition_indices(idx, val_frac, cal_frac, test_frac)
    return Split(train, cal, val, test, 'random', {'seed': seed})


def chronological_split(df: pd.DataFrame, val_frac=.1, cal_frac=.1, test_frac=.2,
                        embargo_hours: float = 4.0) -> Split:
    """Leakage-safe chronological split: train -> validation -> calibration -> test.

    Validation is used for model/hyperparameter selection. Calibration comes
    afterwards so conformal residuals are both selection-independent and as
    temporally close as possible to the untouched test block. Earlier blocks
    are purged so every regression label has matured before the next block,
    with an additional configured embargo.
    """
    ordered = df.sort_values('connection_time').index.to_numpy()
    # The generic partitioner returns the second block before the third, so the
    # fractions are passed in the order the chronological protocol consumes them:
    # the second block is validation and the third is calibration.
    train, middle1, middle2, test = _partition_indices(ordered, cal_frac, val_frac, test_frac)
    val, cal = middle1, middle2

    if min(len(train), len(val), len(cal), len(test)) == 0:
        return Split(train, cal, val, test, 'chronological', {'embargo_hours': embargo_hours})

    train_boundary = df.loc[val[0], 'connection_time']
    val_boundary = df.loc[cal[0], 'connection_time']
    cal_boundary = df.loc[test[0], 'connection_time']
    train = _purge_to_boundary(df, train, train_boundary, embargo_hours)
    val = _purge_to_boundary(df, val, val_boundary, embargo_hours)
    cal = _purge_to_boundary(df, cal, cal_boundary, embargo_hours)
    return Split(
        train, cal, val, test, 'chronological',
        {
            'embargo_hours': embargo_hours,
            'temporal_order': 'train->validation->calibration->test',
            'validation_start': str(train_boundary),
            'calibration_start': str(val_boundary),
            'test_start': str(cal_boundary),
        },
    )


def _identified_user_mask(df: pd.DataFrame) -> pd.Series:
    u = df['user_id'].astype('string')
    stripped = u.str.strip()
    bad = stripped.isna() | stripped.isin(['', '<NA>', 'nan', 'None', 'null', 'NULL'])
    return ~bad


def user_disjoint_split(df: pd.DataFrame, seed: int, val_frac=.1, cal_frac=.1, test_frac=.2) -> Split:
    """Group-disjoint split over identified users only.

    Rows without a stable user identifier are excluded from this protocol.
    Treating every anonymous session as one pseudo-user (or as one unique
    user) would give the protocol a misleading interpretation.
    """
    valid_mask = _identified_user_mask(df)
    eligible = df.loc[valid_mask]
    users = eligible['user_id'].astype('string').astype(str)
    unique = users.drop_duplicates().to_numpy()
    if len(unique) < 4:
        raise ValueError('Not enough identified users for user-disjoint splitting.')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_cal = max(1, int(round(n * cal_frac)))
    test_u = set(unique[:n_test])
    val_u = set(unique[n_test:n_test+n_val])
    cal_u = set(unique[n_test+n_val:n_test+n_val+n_cal])
    train_u = set(unique[n_test+n_val+n_cal:])
    return Split(
        eligible.index[users.isin(train_u)].to_numpy(),
        eligible.index[users.isin(cal_u)].to_numpy(),
        eligible.index[users.isin(val_u)].to_numpy(),
        eligible.index[users.isin(test_u)].to_numpy(),
        'user_disjoint',
        {
            'seed': seed,
            'n_users': n,
            'excluded_anonymous_rows': int((~valid_mask).sum()),
            'anonymous_policy': 'excluded_from_user_disjoint_protocol',
        },
    )


def site_disjoint_splits(df: pd.DataFrame, val_frac=.1, cal_frac=.1, embargo_hours: float = 4.0) -> list[Split]:
    """Pure cross-site transfer with leakage-safe source validation and target calibration.

    Source sites provide train/validation data. The unseen target site is
    chronologically divided into target calibration then untouched test.
    Calibration rows whose final labels are not mature before the target test
    boundary are purged. Source training is likewise purged at the source
    validation boundary.
    """
    out = []
    for target in sorted(df['site'].dropna().unique()):
        source = df[df['site'] != target].sort_values('connection_time')
        target_df = df[df['site'] == target].sort_values('connection_time')
        if len(source) < 20 or len(target_df) < 20:
            continue

        src_idx = source.index.to_numpy()
        n_val = max(1, int(round(len(src_idx) * val_frac)))
        train = src_idx[:-n_val]
        val = src_idx[-n_val:]
        if len(train) and len(val):
            source_val_boundary = df.loc[val[0], 'connection_time']
            train = _purge_to_boundary(df, train, source_val_boundary, embargo_hours)
        else:
            source_val_boundary = pd.NaT

        target_idx = target_df.index.to_numpy()
        n_cal = max(1, int(round(len(target_idx) * cal_frac)))
        cal = target_idx[:n_cal]
        test = target_idx[n_cal:]
        if len(cal) and len(test):
            target_test_boundary = df.loc[test[0], 'connection_time']
            cal = _purge_to_boundary(df, cal, target_test_boundary, embargo_hours)
        else:
            target_test_boundary = pd.NaT

        out.append(Split(
            train, cal, val, test, f'site_disjoint:{target}',
            {
                'target_site': target,
                'embargo_hours': embargo_hours,
                'source_validation_start': str(source_val_boundary),
                'target_test_start': str(target_test_boundary),
                'target_calibration_purged': True,
            },
        ))
    return out


def _purge_to_boundary(df: pd.DataFrame, idx: np.ndarray, boundary: pd.Timestamp, embargo_hours: float) -> np.ndarray:
    if len(idx) == 0 or pd.isna(boundary):
        return idx
    embargo = pd.Timedelta(hours=embargo_hours)
    allowed = df.loc[idx, 'disconnect_time'] < (boundary - embargo)
    return idx[allowed.to_numpy()]


def validate_split(df: pd.DataFrame, split: Split) -> dict:
    parts = {'train': set(split.train), 'calibration': set(split.calibration), 'validation': set(split.validation), 'test': set(split.test)}
    overlap = {}
    keys = list(parts)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            overlap[f'{a}__{b}'] = len(parts[a] & parts[b])
    user_overlap = None
    if split.name.startswith('user_disjoint'):
        train_users = set(df.loc[split.train, 'user_id'].dropna().astype(str))
        test_users = set(df.loc[split.test, 'user_id'].dropna().astype(str))
        user_overlap = len(train_users & test_users)
    return {
        'name': split.name,
        'sizes': {k: len(v) for k, v in parts.items()},
        'index_overlap': overlap,
        'user_overlap': user_overlap,
        'metadata': split.metadata,
    }
