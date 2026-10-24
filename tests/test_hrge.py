"""HRGE gate behaviour."""
import numpy as np
import pandas as pd

from evtrust.hrge import GRID, HRGE, fit_gate, history_free_columns


class _Const:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values[: len(frame)]


def _frame(counts):
    return pd.DataFrame({'user_prior_count': counts})


def test_gate_picks_history_model_where_it_is_right_and_free_model_elsewhere():
    rng = np.random.default_rng(0)
    n = 2000
    counts = np.where(np.arange(n) < n // 2, 0, 15)            # half new drivers, half regulars
    y = rng.normal(10, 1, n)
    hist = np.where(counts == 15, y, y + 6)                     # history model is only good for regulars
    free = np.where(counts == 0, y, y + 6)                      # free model is only good for new drivers
    m = HRGE(f_hist=_Const(hist), f_free=_Const(free), min_rows=100)
    fit_gate(m, _frame(counts), y)
    assert m.weights['10+'] == 1.0
    assert m.weights['0'] == 0.0
    blended = m.predict(_frame(counts))
    assert np.allclose(blended, y)


def test_gate_falls_back_to_pooled_weight_for_rare_groups():
    y = np.zeros(400)
    counts = np.r_[np.full(395, 15), np.full(5, 1)]
    m = HRGE(f_hist=_Const(np.zeros(400)), f_free=_Const(np.ones(400)), min_rows=50)
    fit_gate(m, _frame(counts), y)
    assert '1-2' not in m.weights
    assert m.gate(_frame([1]))[0] == m.pooled_weight


def test_gate_never_worse_than_either_component_on_validation():
    rng = np.random.default_rng(4)
    n = 3000
    counts = rng.choice([0, 1, 5, 20], n)
    y = rng.normal(0, 1, n)
    hist = y + rng.normal(0, 1.0, n)
    free = y + rng.normal(0, 1.5, n)
    m = HRGE(f_hist=_Const(hist), f_free=_Const(free), min_rows=50)
    fit_gate(m, _frame(counts), y)
    mae = lambda p: np.mean(np.abs(y - p))
    assert mae(m.predict(_frame(counts))) <= min(mae(hist), mae(free)) + 1e-12
    assert set(m.weights.values()) <= set(GRID.tolist())


def test_history_free_columns_drop_only_user_history():
    cols = ['arrival_hour_sin', 'user_prior_count', 'user_prior_mean_energy', 'station_prior_mean_duration']
    assert history_free_columns(cols) == ['arrival_hour_sin', 'station_prior_mean_duration']
