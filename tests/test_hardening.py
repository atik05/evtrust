"""Invariants added during the second hardening pass."""
import numpy as np
import pandas as pd

from evtrust.data import clean_sessions
from evtrust.splits import chronological_split
from evtrust.stats import paired_day_block_bootstrap, block_metric_ci


def _sessions(n=400):
    t = pd.date_range('2020-01-01', periods=n, freq='6h', tz='UTC')
    return pd.DataFrame({'connection_time': t, 'disconnect_time': t + pd.Timedelta(hours=3),
                         'user_id': [f'u{i%30}' for i in range(n)], 'site': ['a'] * n,
                         'station_id': ['s'] * n, 'duration_hours': 3.0, 'kwh_delivered': 5.0})


def test_chronological_block_fractions_are_not_swapped():
    df = _sessions()
    sp = chronological_split(df, val_frac=.05, cal_frac=.20, test_frac=.20)
    # calibration asked for 20 percent, validation for 5 percent
    assert len(sp.calibration) > 3 * len(sp.validation)
    order = [df.loc[i, 'connection_time'].min() for i in (sp.train, sp.validation, sp.calibration, sp.test)]
    assert order == sorted(order)


def _paired_frame(seed=0, n=4000, n_days=120):
    rng = np.random.default_rng(seed)
    days = pd.date_range('2020-01-01', periods=n_days).strftime('%Y-%m-%d')
    base = pd.DataFrame({'date': rng.choice(days, n), 'row_id': np.arange(n),
                         'y_true': rng.normal(5, 2, n)})
    a = base.assign(model='a', y_pred=base.y_true + rng.normal(0, 1.0, n))
    b = base.assign(model='b', y_pred=base.y_true + rng.normal(.3, 1.4, n))
    return pd.concat([a, b], ignore_index=True)


def _reference_paired(df, n_boot, seed):
    """Original formulation: resample days, rebuild rows, recompute MAE."""
    a = df[df.model == 'a'][['date', 'row_id', 'y_true', 'y_pred']].rename(columns={'y_pred': 'pred_a'})
    b = df[df.model == 'b'][['date', 'row_id', 'y_pred']].rename(columns={'y_pred': 'pred_b'})
    m = a.merge(b, on=['date', 'row_id'])
    days = m['date'].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        z = pd.concat([m[m.date == d] for d in rng.choice(days, len(days), replace=True)], ignore_index=True)
        out.append(np.mean(np.abs(z.y_true - z.pred_a)) - np.mean(np.abs(z.y_true - z.pred_b)))
    return np.asarray(out)


def test_vectorised_bootstrap_matches_reference_distribution():
    df = _paired_frame()
    ref = _reference_paired(df, 400, 11)
    fast = paired_day_block_bootstrap(df, 'a', 'b', 'mae', n_boot=4000, seed=11)
    # Same sampling scheme, so the bootstrap distributions must agree closely.
    assert abs(fast['difference_a_minus_b'] - ref.mean()) < 0.02
    assert abs(fast['ci_low'] - np.quantile(ref, .025)) < 0.03
    assert abs(fast['ci_high'] - np.quantile(ref, .975)) < 0.03


def test_block_ci_brackets_the_point_estimate():
    df = _paired_frame()
    g = df[df.model == 'a']
    lo, hi = block_metric_ci(g, 'mae', n_boot=2000, seed=5)
    mae = float(np.mean(np.abs(g.y_true - g.y_pred)))
    assert lo < mae < hi


def test_cleaning_drops_zero_length_sessions_and_logs_flow(tmp_path):
    df = _sessions(50)
    df.loc[:4, 'disconnect_time'] = df.loc[:4, 'connection_time']  # connector artefacts
    df['duration_hours'] = (df.disconnect_time - df.connection_time).dt.total_seconds() / 3600
    df['session_id'] = [f's{i}' for i in range(len(df))]
    flow = tmp_path / 'data_flow.csv'
    out = clean_sessions(df, flow_path=flow)
    assert len(out) == 45
    assert (out['duration_hours'] > 0).all()
    recorded = pd.read_csv(flow)
    assert recorded.loc[recorded.stage == 'raw_sessions', 'sessions'].iloc[0] == 50
