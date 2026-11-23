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
