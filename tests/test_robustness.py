"""Unit tests for the robustness analyses and the ACN-Sim summaries (no data or GPU needed)."""
import numpy as np
import pytest

from robustness.acnsim import _first_cap
from robustness.common import alap
from robustness.decomp import target_encode


def test_unseen_driver_encoding():
    users_tr = np.array(['a', 'a', 'b', 'b', 'b'])
    y_tr = np.array([1.0, 3.0, 5.0, 5.0, 5.0])
    _, q_nan = target_encode(users_tr, y_tr, np.array(['a', 'z']), seed=0)
    _, q_mean = target_encode(users_tr, y_tr, np.array(['a', 'z']), seed=0, unseen='mean')
    assert np.isnan(q_nan[1]) and q_mean[1] == pytest.approx(y_tr.mean())
    assert q_nan[0] == pytest.approx(q_mean[0])          # a seen driver is unaffected


def test_proposition_1_union_bound():
    rng = np.random.default_rng(0)
    D = rng.uniform(0.5, 12, 20000); E = rng.uniform(0, 40, 20000)
    Dh = D * rng.uniform(0.6, 1.4, D.size); Eh = E * rng.uniform(0.6, 1.4, E.size)
    E_ach, U, F = alap(D, E, Dh, Eh, 6.6)
    short = U > 1e-12
    assert np.all((Dh > D) | (Eh < E_ach) | ~short)       # U > 0 only if a forecast is optimistic
    safe = (Dh <= D) & (Eh >= E_ach)
    assert np.all(U[safe] <= 1e-12)
    assert np.all(F[safe] <= D[safe] - E_ach[safe] / 6.6 + 1e-9)   # deferral within the oracle's


def test_capacity_needed_interpolation():
    caps = np.array([0.3, 0.4, 0.5, 0.6])
    share = np.array([10.0, 4.0, 1.0, 0.2])
    assert _first_cap(caps, share) == pytest.approx(0.4 + 0.1 * (4.0 - 2.0) / (4.0 - 1.0))
    assert _first_cap(caps, np.array([1.0, 0.5, 0.2, 0.1])) == pytest.approx(0.3)
    assert np.isnan(_first_cap(caps, np.array([9.0, 8.0, 7.0, 6.0])))


def test_proposition_2_state_conditional_bound():
    """Calibrating the start within each state holds P(U > 0) at gamma in every state, whatever the error scale."""
    import pandas as pd
    from robustness.infostate import information_state, outcome, start_shift
    rng = np.random.default_rng(1)
    n = 60000
    state = rng.choice(['anonymous', 'first visit', 'returning'], n, p=[0.3, 0.1, 0.6])
    scale = np.select([state == 'anonymous', state == 'first visit'], [2.5, 4.0], 1.0)
    D = rng.uniform(1, 12, n); E = rng.uniform(1, 40, n)
    Dp = D + rng.normal(0, scale); Ep = E + rng.normal(0, 3 * scale)
    groups = rng.integers(0, 5, n)
    taus = np.array([0.2, 0.1, 0.05, 0.02])
    S = start_shift(D, E, Dp, Ep, groups, state, 6.6, taus)
    _, U, _, _ = outcome(D, E, S, 6.6)
    for st in ('anonymous', 'first visit', 'returning'):
        m = state == st
        rate = (U[:, m] > 1e-9).mean(1)
        assert np.all(rate <= taus + 0.01) and np.all(rate >= taus - 0.02)
    # the observable state: first visit exactly where the deployment forecast equals the history-hidden forecast
    z = pd.DataFrame({'population': ['anonymous', 'cohort', 'cohort'], 'Dp_deployment': [1.0, 2.0, 3.0], 'Dp_cold': [1.0, 2.0, 3.5],
                      'Ep_deployment': [5.0, 6.0, 7.0], 'Ep_cold': [5.0, 6.0, 7.0]})
    assert list(information_state(z)) == ['anonymous', 'first visit', 'returning']


def test_shortfall_iff_start_after_latest_start():
    from robustness.infostate import latest_start, outcome
    rng = np.random.default_rng(2)
    D = rng.uniform(0.2, 12, 5000); E = rng.uniform(0, 60, 5000); S = rng.uniform(0, 14, 5000)
    _, U, F, Fo = outcome(D, E, S[None], 6.6)
    late = S > latest_start(D, E, 6.6) + 1e-9
    pos = np.minimum(E, 6.6 * D) > 1e-9
    assert np.all((U[0] > 1e-9) == (late & pos))
