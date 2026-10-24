"""Reporting fixes: finite-sample bootstrap p-values and out-of-sample skill."""
import numpy as np
import pandas as pd

from evtrust.experiments import _add_skill_scores
from evtrust.metrics import regression_metrics
from evtrust.stats import paired_day_block_bootstrap


def test_bootstrap_p_value_is_never_exactly_zero():
    rng = np.random.default_rng(0)
    n = 3000
    days = np.repeat(np.arange(300), 10).astype(str)
    y = rng.normal(10, 1, n)
    d = pd.concat([pd.DataFrame({'date': days, 'row_id': np.arange(n), 'y_true': y, 'model': 'a', 'y_pred': y + 0.01}),
                   pd.DataFrame({'date': days, 'row_id': np.arange(n), 'y_true': y, 'model': 'b', 'y_pred': y + 5.0})])
    r = paired_day_block_bootstrap(d, 'a', 'b', 'mae', n_boot=999, seed=1)
    assert r['p_a_better'] == 1.0                     # every draw favours a
    assert np.isclose(r['p_two_sided'], 2 / 1000)     # (0 + 1) / (B + 1), doubled
    assert r['p_two_sided'] > 0


def test_r2_oos_is_positive_when_model_beats_training_mean_under_shift():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(0, 1, n)
    y = 20 + 2 * x + rng.standard_t(2, n)             # heavy tails, test mean shifted to 20
    model = 20 + 2 * x - 2.0                          # informative but biased
    train_mean = np.full(n, 12.0)                     # historical mean learned before the shift
    sk_r2 = regression_metrics(y, model)['r2']
    base = dict(target='energy', protocol='user_disjoint', split_name='user_disjoint', target_site=np.nan, seed=1)
    p = pd.concat([pd.DataFrame({**base, 'model': 'catboost', 'row_id': np.arange(n), 'y_true': y, 'y_pred': model}),
                   pd.DataFrame({**base, 'model': 'historical_mean', 'row_id': np.arange(n), 'y_true': y, 'y_pred': train_mean})])
    metrics = pd.DataFrame([{**base, 'model': 'catboost'}, {**base, 'model': 'historical_mean'}])
    out = _add_skill_scores(metrics, p).set_index('model')
    assert out.loc['catboost', 'r2_oos'] > 0.5 and out.loc['catboost', 'mae_skill'] > 0.5
    assert out.loc['historical_mean', 'r2_oos'] == 0.0
    assert out.loc['catboost', 'r2_oos'] > sk_r2      # test-mean r2 understates skill under shift


def test_identical_predictions_are_never_significant():
    """Old sign-frequency p-value returned 0 for identical models; the corrected one returns 1."""
    rng = np.random.default_rng(0)
    n = 2000
    days = np.repeat(np.arange(200), 10).astype(str)
    y = rng.normal(10, 2, n)
    pred = y + rng.normal(0, 1, n)
    d = pd.concat([pd.DataFrame({'date': days, 'row_id': np.arange(n), 'y_true': y, 'model': 'a', 'y_pred': pred}),
                   pd.DataFrame({'date': days, 'row_id': np.arange(n), 'y_true': y, 'model': 'b', 'y_pred': pred})])
    r = paired_day_block_bootstrap(d, 'a', 'b', 'mae', n_boot=3000, seed=1)
    assert r['difference_a_minus_b'] == 0
    assert r['p_two_sided'] == 1.0
