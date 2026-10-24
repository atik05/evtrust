"""A restarted run must skip finished work and reproduce identical predictions."""
import os

import numpy as np
import pandas as pd

from evtrust.experiments import run_regression
from evtrust.features import build_features


def _frame(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp('2019-01-01', tz='UTC')
    conn = pd.DatetimeIndex([base + pd.Timedelta(minutes=int(m)) for m in np.sort(rng.integers(0, 400 * 24 * 60, n))])
    dur = np.clip(2 + 1.5 * np.sin(2 * np.pi * conn.hour / 24) + rng.normal(0, .8, n), .3, 12)
    site = rng.choice(['caltech', 'jpl'], n, p=[.6, .4])
    df = pd.DataFrame({
        'session_id': [f's{i}' for i in range(n)], 'site': site,
        'station_id': [f'{s}_st{rng.integers(1, 6)}' for s in site],
        'user_id': [f'u{rng.integers(1, 60)}' for _ in range(n)],
        'kwh_delivered': np.clip(2.4 * dur + rng.normal(0, 1.5, n), .1, 60),
        'connection_time': conn, 'disconnect_time': conn + pd.to_timedelta(dur, unit='h'),
        'done_charging_time': conn + pd.to_timedelta(dur * .8, unit='h'),
        'user_inputs': [[{'modifiedAt': t.isoformat(), 'kWhRequested': 12.0, 'minutesAvailable': 240.0}] for t in conn],
        'duration_hours': dur, 'charging_duration_hours': dur * .8,
    })
    return build_features(df)


def _cfg():
    return {
        'project': {'five_seeds': [11, 23], 'resume': True},
        'experiment': {'learned_models': ['ridge']},
        'splits': {'validation_fraction': .1, 'calibration_fraction': .1, 'test_fraction': .2, 'embargo_hours': 4},
        'features': {'numeric': ['arrival_hour_sin', 'arrival_hour_cos', 'user_prior_count',
                                 'user_prior_mean_duration', 'station_prior_mean_duration'],
                     'categorical': ['site', 'is_weekend']},
        'models': {'ridge': {'alpha': 10.0}},
        'conformal': {'alpha': .1, 'calibration_budgets': [0, 25], 'cqr': False},
        'tuning': {'enabled': False, 'canonical_seed': 11},
    }


def test_resume_skips_finished_units_and_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _frame()
    cfg = _cfg()
    run_regression(df, cfg)
    first = pd.read_parquet('results/predictions/regression_predictions.parquet')
    shards = sorted(p for p in os.listdir('results/predictions/shards') if p.startswith('pred__'))
    assert len(shards) >= 4

    removed = shards[:2]
    for name in removed:
        os.remove(os.path.join('results/predictions/shards', name))
    run_regression(df, cfg)
    second = pd.read_parquet('results/predictions/regression_predictions.parquet')

    key = ['seed', 'split_name', 'target', 'model', 'row_id']
    a = first.sort_values(key).reset_index(drop=True)
    b = second.sort_values(key).reset_index(drop=True)
    assert len(a) == len(b)
    assert np.allclose(a.y_pred.to_numpy(), b.y_pred.to_numpy())
    assert np.allclose(a.y_true.to_numpy(), b.y_true.to_numpy())
