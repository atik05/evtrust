"""The conservative flexibility metric never exceeds the metric of the paper or the oracle deferral."""
import numpy as np

from robustness.common import curves


def test_conservative_deferral_bounded():
    rng = np.random.default_rng(0)
    n = 400
    D = rng.uniform(1, 10, n); E = rng.uniform(1, 40, n)
    Dp = D * rng.uniform(0.6, 1.4, n); Ep = E * rng.uniform(0.6, 1.4, n)
    g = rng.integers(0, 5, n)
    U, EA, F, Fo = curves(D, E, Dp, Ep, g, 6.6, taus=(0.5, 0.3, 0.1))
    Fs = np.minimum(F, Fo[None])
    assert (Fs <= F + 1e-12).all() and (Fs <= Fo[None] + 1e-12).all()
    # deferral beyond the oracle's always leaves energy undelivered
    over = F > Fo[None] + 1e-9
    assert (U[over] > 0).all()


def test_deferral_beyond_oracle_equals_undelivered_energy_over_rate():
    """Phi - min(Phi, Phi*) = U / P for every session and quantile level."""
    rng = np.random.default_rng(1)
    n, P = 500, 6.6
    D = rng.uniform(0.2, 12, n); E = rng.uniform(0.5, 60, n)       # includes sessions with E > P * D
    Dp = D * rng.uniform(0.4, 1.6, n); Ep = E * rng.uniform(0.4, 1.6, n)
    g = rng.integers(0, 4, n)
    U, EA, F, Fo = curves(D, E, Dp, Ep, g, P, taus=(0.5, 0.2, 0.05))
    Fs = np.minimum(F, Fo[None])
    np.testing.assert_allclose(F - Fs, U / P, atol=1e-9)
    # at the same sessions the conservative metric lowers every forecast by the same sum(U) / P
    assert np.isclose((F - Fs).sum(), U.sum() / P)


def test_boundary_sessions():
    """Zero achievable deferral (E >= P * D) and exact forecasts."""
    P = 6.6
    D = np.array([2.0, 3.0, 5.0, 4.0]); E = np.array([13.2, 30.0, 6.6, 10.0])   # E = P*D, E > P*D, slack, slack
    U, EA, F, Fo = curves(D, E, D.copy(), E.copy(), np.zeros(4, int), P, taus=(0.5,))
    assert np.allclose(Fo[:2], 0.0) and (Fo[2:] > 0).all()
    Fs = np.minimum(F, Fo[None])
    np.testing.assert_allclose(F - Fs, U / P, atol=1e-9)
    assert (U >= -1e-12).all()
