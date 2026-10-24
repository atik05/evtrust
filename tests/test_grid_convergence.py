"""The smart-charging results must not depend on the grid of quantile levels."""
import numpy as np
import pandas as pd

from evtrust import impact


def _rows(seed=0, n_drivers=150):
    rng = np.random.default_rng(seed)
    recs = []
    for u in range(n_drivers):
        b_d, b_e = rng.normal(0, 1.5), rng.normal(0, 4)
        for j in range(20):
            D = max(0.5, 8 + b_d + rng.gamma(2, 1.2) - 2.4)
            E = max(0.5, min(6.6 * D, 12 + b_e + rng.gamma(2, 3) - 6))
            recs.append({'session_key': f'{u}-{j}', 'user': f'u{u}', 'k': j, 'seed': 1,
                         'duration_true': D, 'energy_true': E, 'requested_duration': np.nan, 'requested_energy': np.nan,
                         'd_A': D - b_d * 0.2 + rng.normal(0, 1), 'd_C': D - b_d * 0.4 + rng.normal(0, 1.5), 'd_D': 8 + rng.normal(0, 0.5),
                         'e_A': E - b_e * 0.2 + rng.normal(0, 3), 'e_C': E - b_e * 0.4 + rng.normal(0, 4), 'e_D': 12 + rng.normal(0, 1)})
    r = pd.DataFrame(recs)
    du = r.assign(pred_A=r.d_A, pred_C=r.d_C, pred_D=r.d_D)
    en = r.assign(pred_A=r.e_A, pred_C=r.e_C, pred_D=r.e_D)
    return du, en


def test_grid_is_dense_and_starts_at_the_point_forecast():
    t = np.array(impact.TAUS)
    assert t[0] == 0.5 and len(t) >= 100 and t.min() <= 0.005 and np.all(np.diff(t) < 0)


def test_results_converge_when_the_grid_is_refined():
    du, en = _rows()
    finer = (0.5,) + tuple(round(0.4975 - 0.0025 * i, 4) for i in range(199))
    _, _, a, _ = impact.run(du, en, 6.6)
    _, _, b, _ = impact.run(du, en, 6.6, taus=finer)
    m = a.merge(b, on=['setting', 'unmet_target_pct'], suffixes=('', '_fine'))
    assert len(m) and (m['flex_vs_oracle_pct'] - m['flex_vs_oracle_pct_fine']).abs().max() < 0.5
