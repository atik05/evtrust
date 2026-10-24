"""Tests for the cold-start decomposition, feature-availability and impact analyses."""
import numpy as np
import pandas as pd

from evtrust import availability, coldstart, impact

NUM = ['arrival_hour_sin', 'user_prior_count', 'user_prior_mean_duration', 'user_prior_median_duration',
       'user_prior_mean_energy']


def _sessions(n_users=60, per_user=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp('2019-01-01', tz='UTC')
    for u in range(n_users):
        base_e = rng.uniform(5, 25)
        for i in range(per_user):
            dur = max(0.5, rng.normal(6, 1.5))
            rows.append({'session_id': f's{u}_{i}', 'user_id': f'u{u}', 'site': 'a' if u % 2 else 'b',
                         'station_id': f'st{rng.integers(0, 6)}',
                         'connection_time': t0 + pd.Timedelta(days=int(i * 7 + u % 5)),
                         'duration_hours': dur, 'kwh_delivered': base_e + rng.normal(0, 1),
                         'charging_duration_hours': dur * 0.6, 'arrival_hour_sin': rng.normal(),
                         'requested_duration_hours': np.nan, 'requested_energy_kwh': np.nan})
    d = pd.DataFrame(rows).sort_values('connection_time').reset_index(drop=True)
    g = d.groupby('user_id')
    d['user_prior_count'] = g.cumcount().astype(float)
    for col, src in [('user_prior_mean_energy', 'kwh_delivered'), ('user_prior_mean_duration', 'duration_hours'),
                     ('user_prior_median_duration', 'duration_hours')]:
        d[col] = g[src].transform(lambda s: s.shift(1).expanding().mean())
    return d


def test_simulate_oracle_has_no_unmet_energy_and_full_deferral():
    D = np.array([8.0, 4.0, 2.0]); E = np.array([10.0, 30.0, 5.0]); P = 6.6
    E_ach, unmet, flex = impact.simulate(D, E, D, np.minimum(E, P * D), P)
    assert np.allclose(unmet, 0)
    assert np.allclose(flex, D - E_ach / P)
    assert np.isclose(E_ach[1], P * 4.0)            # infeasible request capped by the charger


def test_simulate_penalises_late_departure_forecast():
    D = np.array([6.0]); E = np.array([20.0]); P = 6.6
    _, unmet_ok, _ = impact.simulate(D, E, D, E, P)
    _, unmet_late, _ = impact.simulate(D, E, D + 2.0, E, P)   # car leaves 2 h earlier than planned
    assert unmet_ok[0] == 0 and unmet_late[0] > 0


def test_k_bucket_edges():
    assert list(coldstart.k_bucket(np.array([0, 1, 2, 3, 9, 10, 25]))) == ['0', '1', '2', '3-4', '5-9', '10-19', '20+']


def test_cluster_ci_is_zero_for_identical_errors():
    z = pd.DataFrame({'user': ['a', 'a', 'b', 'c'], 'x': [1.0, 2, 3, 4]})
    z['y'] = z['x']
    est, lo, hi, _ = coldstart.cluster_ratio_ci(z, 'x', 'y', 200)
    assert est == 0 and lo == 0 and hi == 0


def test_run_unit_settings_are_paired_and_cold_start_hides_history():
    d = _sessions()
    out = coldstart.run_unit(d, 'kwh_delivered', 'hgb', 0, NUM, {}, None)
    assert out['session_key'].is_unique
    assert out.loc[~out['is_T'], 'pred_B'].isna().all() and out.loc[out['is_T'], 'pred_B'].notna().all()
    for s in 'ACDE':
        assert out[f'pred_{s}'].notna().all()
    # driver energy is strongly driver-specific here, so hiding history must hurt
    assert out['err_D'].mean() > out['err_C'].mean()
    dec, warm = coldstart.summarise(out, n_boot=200)
    comp = set(dec['component'])
    assert {'future-session leakage', 'unseen-driver cost', 'value of own history', 'total gap'} <= comp


def test_availability_post_plugin_information_lowers_error():
    d = _sessions()
    d['kwh_delivered'] = d['charging_duration_hours'] * 6.0 + np.random.default_rng(1).normal(0, 0.5, len(d))
    rows = availability.run(d, ['arrival_hour_sin'], 'hgb', [0], {})
    s = availability.summarise(rows, 200).set_index(['scope', 'feature_set'])
    assert s.loc[('all', 'plus_charging'), 'mae'] < s.loc[('all', 'plug_in'), 'mae']
