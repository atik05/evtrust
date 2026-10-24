"""Invariants for the domain-aware extensions."""
import numpy as np
import pandas as pd

from evtrust.improvements import (add_published_style_features, apply_history_dropout,
                                  coverage_by_group, history_group, mondrian_regression_calibrate,
                                  mondrian_regression_interval)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'user_id': [f'u{i % 25}' for i in range(n)],
        'user_prior_count': rng.integers(0, 20, n).astype(float),
        'user_prior_mean_duration': rng.normal(4, 1, n),
        'user_prior_median_duration': rng.normal(4, 1, n),
        'user_prior_mean_energy': rng.normal(10, 3, n),
        'station_prior_mean_duration': rng.normal(4, 1, n),
        'duration_hours': rng.normal(4, 1.5, n),
        'kwh_delivered': rng.normal(10, 4, n),
    })


def test_history_dropout_masks_only_history_columns():
    df = _frame()
    out = apply_history_dropout(df, rate=0.4, seed=1)
    masked = out['user_prior_mean_duration'].isna()
    assert 0.25 < masked.mean() < 0.55                      # roughly the requested rate
    assert (out.loc[masked, 'user_prior_count'] == 0).all()  # a new driver has no history
    # other features and labels are untouched
    assert out['station_prior_mean_duration'].equals(df['station_prior_mean_duration'])
    assert out['duration_hours'].equals(df['duration_hours'])
    assert apply_history_dropout(df, rate=0.0, seed=1).equals(df)


def test_history_groups_follow_the_declared_bins():
    df = pd.DataFrame({'user_prior_count': [0, 1, 2, 3, 9, 10, 40]})
    assert list(history_group(df)) == ['0', '1-2', '1-2', '3-9', '3-9', '10+', '10+']


def test_conditional_calibration_fixes_a_group_with_wider_errors():
    rng = np.random.default_rng(3)
    n = 4000
    group_src = pd.DataFrame({'user_prior_count': rng.choice([0, 15], n)})
    g = history_group(group_src)
    scale = np.where(g.to_numpy() == '0', 4.0, 0.5)          # new drivers are much harder
    y = rng.normal(0, scale)
    pred = np.zeros(n)
    half = n // 2
    cal, test = slice(0, half), slice(half, n)
    from evtrust.conformal import split_conformal_calibrate, split_conformal_interval
    q = split_conformal_calibrate(y[cal], pred[cal], .10)
    lo_m, hi_m = split_conformal_interval(pred[test], q)
    table = mondrian_regression_calibrate(y[cal], pred[cal], g.iloc[cal], .10)
    lo_c, hi_c = mondrian_regression_interval(pred[test], g.iloc[test], table)
    marg = coverage_by_group(y[test], lo_m, hi_m, g.iloc[test]).set_index('group')
    cond = coverage_by_group(y[test], lo_c, hi_c, g.iloc[test]).set_index('group')
    # marginal calibration under-covers the hard group and over-covers the easy one
    assert marg.loc['0', 'coverage'] < 0.85
    assert marg.loc['10+', 'coverage'] > 0.95
    # conditional calibration brings both near nominal
    assert abs(cond.loc['0', 'coverage'] - 0.90) < 0.04
    assert abs(cond.loc['10+', 'coverage'] - 0.90) < 0.04


def test_small_groups_fall_back_to_the_pooled_quantile():
    y = np.random.default_rng(0).normal(0, 1, 300)
    pred = np.zeros(300)
    g = pd.Series(['10+'] * 295 + ['0'] * 5)
    table = mondrian_regression_calibrate(y, pred, g, .10, min_per_group=50)
    assert '0' not in table and '10+' in table               # 5 points cannot carry a quantile
    lo, hi = mondrian_regression_interval(pred[:5], g.iloc[295:], table)
    assert np.allclose(hi - lo, 2 * table['__pooled__'])


def test_published_style_features_use_future_sessions():
    df = _frame()
    out = add_published_style_features(df)
    # the aggregate for a user now equals that user's mean over the whole file
    expected = df.groupby('user_id')['duration_hours'].mean()
    got = out.groupby('user_id')['user_prior_mean_duration'].first()
    assert np.allclose(got.loc[expected.index].to_numpy(), expected.to_numpy())
    assert not np.allclose(out['user_prior_mean_duration'], df['user_prior_mean_duration'])
