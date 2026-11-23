"""Genuine first identified sessions against the simulated cold start, and headline flexibility CIs.

Calibration uses all cohort sessions (other driver groups), exactly as in the paper; only the evaluation is
restricted: FV = first identified session (k = 0, where settings C and D coincide), RET = later sessions.
Cutoffs drop drivers whose first identified session falls within the first N days of the data, where a
'first' session may be an app registration rather than a first visit. Driver-cluster bootstrap.
Also reported: energy shifted per session (kWh that charging on arrival would have delivered
within the deferral, i.e. moved later), the population rule (median duration and energy of the other driver
groups, no model and no driver information, shifted in the same way), and the characteristics of first
visits against later sessions (arrival hour, energy, duration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Ctx, CurveSet, bootstrap_table, cross_q_matrix, curves, stable_group

CUTOFFS = (-1, 60, 180, 365)


def first_session_days(ctx: Ctx) -> pd.Series:
    s = ctx.sessions()
    t = pd.to_datetime(s['connection_time'], utc=True)
    ident = s['user_id'].notna() & (s['user_id'].astype(str).str.strip() != '')
    first = t[ident].groupby(s.loc[ident, 'user_id'].astype(str)).min()
    return (first - t.min()).dt.total_seconds() / 86400


def _kwh(U, EA, F, Fo, P):
    """Energy moved later by the deferral: what charging on arrival would have delivered during it."""
    return U, EA, np.minimum(EA[None], P * F), np.minimum(EA, P * Fo)


def run(ctx: Ctx):
    s_all = ctx.sessions()
    ct = pd.to_datetime(s_all['connection_time'])
    hour = pd.Series(ct.dt.hour.to_numpy() + ct.dt.minute.to_numpy() / 60, index=s_all['session_id'].astype(str))
    days = first_session_days(ctx)
    days.rename('days_after_data_start').to_csv(ctx.p('firstvisit_first_session_days.csv'))
    for learner in ('catboost', 'ft'):
        js = ctx.joined(learner, 'ACD')
        drivers = np.unique(np.concatenate([j['user'].astype(str).to_numpy() for j in js.values()]))
        nd = len(drivers)
        tabs, maes = [], []
        for cut in CUTOFFS:
            sel_users = set(days[days > cut].index) if cut >= 0 else set(drivers)
            sets = {}
            for seed, j in js.items():
                users = j['user'].astype(str).to_numpy()
                di = np.searchsorted(drivers, users)
                k = j['k'].to_numpy()
                keep = np.isin(users, list(sel_users))
                subsets = {'ALL': np.ones(len(k), bool), 'FV': (k == 0) & keep, 'RET': (k >= 1) & keep}
                D, E = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
                grp = stable_group(users)
                hr = hour.reindex(j.index.astype(str)).to_numpy(float)
                fc = {s: (j[f'pred_{s}'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float)) for s in 'ACD'}
                fc['POP'] = (cross_q_matrix(D, grp, [0.5])[0], cross_q_matrix(E, grp, [0.5])[0])   # population rule
                for s, (Dp, Ep) in fc.items():
                    if s == 'POP' and cut != -1:
                        continue
                    U, EA, F, Fo = curves(D, E, Dp, Ep, grp, ctx.power)
                    for name, m in subsets.items():
                        sets.setdefault(f'{s}_{name}', CurveSet(nd)).add(U, EA, F, Fo, di, m)
                        if name == 'ALL':
                            sets.setdefault(f'{s}_{name}_hours', CurveSet(nd)).add(U, EA, F, Fo, di, m)
                            sets.setdefault(f'{s}_{name}_kwh', CurveSet(nd)).add(*_kwh(U, EA, F, Fo, ctx.power), di, m)
                        if cut == -1:
                            maes.append({'seed': seed, 'setting': s, 'subset': name, 'n': int(m.sum()),
                                         'mae_energy': float(np.abs(E - Ep)[m].mean()),
                                         'mae_duration': float(np.abs(D - Dp)[m].mean()),
                                         'bias_energy': float((Ep - E)[m].mean()), 'bias_duration': float((Dp - D)[m].mean()),
                                         'mean_energy_kwh': float(E[m].mean()), 'mean_duration_h': float(D[m].mean()),
                                         'mean_arrival_hour': float(np.nanmean(hr[m])),
                                         'oracle_deferral_h': float(np.minimum(np.clip(D - np.minimum(E, ctx.power * D) / ctx.power, 0, None), D)[m].mean())})
            keep_names = ['A_ALL', 'C_ALL', 'D_ALL', 'POP_ALL', 'A_FV', 'D_FV', 'A_RET', 'C_RET', 'D_RET',
                          'A_ALL_hours', 'C_ALL_hours', 'D_ALL_hours', 'POP_ALL_hours',
                          'A_ALL_kwh', 'C_ALL_kwh', 'D_ALL_kwh', 'POP_ALL_kwh']
            sets = {k: v for k, v in sets.items() if k in keep_names}
            contrasts = {
                'loss_pct: 1 - D_ALL/C_ALL': lambda v: 100 * (1 - v['D_ALL'] / v['C_ALL']),
                'loss_pct: 1 - D_ALL/A_ALL': lambda v: 100 * (1 - v['D_ALL'] / v['A_ALL']),
                'loss_pp: C_ALL - D_ALL': lambda v: v['C_ALL'] - v['D_ALL'],
                'loss_hours: C_ALL - D_ALL': lambda v: v['C_ALL_hours'] - v['D_ALL_hours'],
                'loss_kwh_shifted: C_ALL - D_ALL': lambda v: v['C_ALL_kwh'] - v['D_ALL_kwh'],
                'observed_loss_pct: 1 - D_FV/C_RET': lambda v: 100 * (1 - v['D_FV'] / v['C_RET']),
                'D_FV - D_RET (pp)': lambda v: v['D_FV'] - v['D_RET'],
                'D_FV - D_ALL (pp)': lambda v: v['D_FV'] - v['D_ALL'],
            }
            # selected drivers are resampled among themselves, the others among themselves (as in Table S37)
            strata = np.isin(drivers, list(sel_users)).astype(int)
            if 'POP_ALL' in sets:
                contrasts['D_ALL - POP_ALL (pp)'] = lambda v: v['D_ALL'] - v['POP_ALL']
            t = bootstrap_table(sets, nd, ctx.n_boot, contrasts=contrasts, strata=strata,
                                hours_sets={k for k in sets if k.endswith('_hours') or k.endswith('_kwh')})
            t.loc[t['quantity'].str.endswith('_kwh'), 'kind'] = "kwh_shifted_per_session"
            t.insert(0, 'cutoff_days', cut)
            t.insert(1, 'drivers_selected', len(sel_users & set(drivers)))
            tabs.append(t)
            ctx.log(f'2a {learner} cutoff {cut}: D_FV {t.set_index("quantity").loc["D_FV", "estimate"]:.2f}, '
                    f'D_ALL {t.set_index("quantity").loc["D_ALL", "estimate"]:.2f}')
        out = pd.concat(tabs, ignore_index=True)
        out.insert(0, 'learner', learner)
        out.to_csv(ctx.p(f'firstvisit_{learner}.csv'), index=False)
        (pd.DataFrame(maes).groupby(['setting', 'subset']).mean(numeric_only=True).drop(columns='seed')
         .to_csv(ctx.p(f'firstvisit_mae_{learner}.csv')))
