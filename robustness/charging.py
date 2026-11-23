"""Charging-physics sensitivity of the flexibility result (no retraining).

(i)  Uncensored need: sessions whose charging ended at least 15 minutes before departure, so the energy
     delivered is the energy needed rather than an amount cut short by departure.
(ii) Charging power: 3.3, 6.6 and 11 kW, and each session's own observed average rate (energy divided by
     the time charging took, bounded to 1 to 19.2 kW) instead of one rate for all sessions.
Calibration on all cohort sessions; evaluation restricted for (i). Driver-cluster bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Ctx, CurveSet, bootstrap_table, curves, stable_group

MARGIN_H = 0.25


def session_physics(ctx: Ctx) -> pd.DataFrame:
    s = ctx.sessions()[['session_id', 'connection_time', 'disconnect_time', 'done_charging_time',
                        'charging_duration_hours', 'kwh_delivered']].copy()
    s['session_key'] = s['session_id'].astype(str)
    t1 = pd.to_datetime(s['disconnect_time'], utc=True)
    tc = pd.to_datetime(s['done_charging_time'], utc=True)
    ch = s['charging_duration_hours'].astype(float)
    s['uncensored'] = tc.notna() & ((t1 - tc).dt.total_seconds() / 3600 >= MARGIN_H) & (ch > 0)
    rate = s['kwh_delivered'] / ch
    s['rate_kw'] = np.where((ch >= 5 / 60) & np.isfinite(rate), np.clip(rate, 1.0, 19.2), np.nan)
    return s.set_index('session_key')[['uncensored', 'rate_kw']]


def run(ctx: Ctx):
    phys = session_physics(ctx)
    rows = []
    for learner in ('catboost', 'ft'):
        js = ctx.joined(learner, 'ACD')
        drivers = np.unique(np.concatenate([j['user'].astype(str).to_numpy() for j in js.values()]))
        nd = len(drivers)
        scenarios = {'P=6.6 kW (paper)': 6.6, 'P=3.3 kW': 3.3, 'P=11 kW': 11.0, 'observed session rate': None}
        for scen, P0 in scenarios.items():
            sets = {}
            for seed, j in js.items():
                ph = phys.reindex(j.index.astype(str))
                users = j['user'].astype(str).to_numpy(); di = np.searchsorted(drivers, users)
                D, E = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
                P = np.where(np.isfinite(ph['rate_kw'].to_numpy(float)), ph['rate_kw'].to_numpy(float), ctx.power) if P0 is None else P0
                unc = ph['uncensored'].fillna(False).to_numpy(bool)
                subsets = {'ALL': np.ones(len(D), bool)}
                if P0 == 6.6:
                    subsets.update({'UNCENSORED': unc, 'CENSORED': ~unc})
                grp = stable_group(users)
                for s in 'ACD':
                    U, EA, F, Fo = curves(D, E, j[f'pred_{s}'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float), grp, P)
                    for name, m in subsets.items():
                        sets.setdefault(f'{s}_{name}', CurveSet(nd)).add(U, EA, F, Fo, di, m)
                if seed == min(js):
                    rows_share = {'uncensored_share': float(unc.mean()), 'rate_known_share': float(np.isfinite(ph['rate_kw']).mean()),
                                  'rate_median_kw': float(np.nanmedian(ph['rate_kw']))}
            contrasts = {}
            for name in {k.split('_', 1)[1] for k in sets}:
                contrasts[f'loss_pct_{name}: 1 - D/C'] = (lambda nm: (lambda v: 100 * (1 - v[f'D_{nm}'] / v[f'C_{nm}'])))(name)
                contrasts[f'loss_pp_{name}: C - D'] = (lambda nm: (lambda v: v[f'C_{nm}'] - v[f'D_{nm}']))(name)
            t = bootstrap_table(sets, nd, ctx.n_boot, contrasts=contrasts)
            t.insert(0, 'scenario', scen); t.insert(0, 'learner', learner)
            for k, v in rows_share.items():
                t[k] = v
            rows.append(t)
            ctx.log(f'2f {learner} {scen}: ' + ', '.join(f"{q} {e:.2f}" for q, e in zip(t['quantity'], t['estimate']) if q.endswith('_ALL')))
    pd.concat(rows, ignore_index=True).to_csv(ctx.p('charging_sensitivity.csv'), index=False)
