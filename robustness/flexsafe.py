"""Conservative flexibility metric (no retraining).

The paper counts as flexibility the deferral Phi = min(theta, D) of the deadline-aware policy. Deferral beyond the
oracle's, Phi* = D - E_ach / P, cannot be delivered safely: when theta exceeds Phi*, part of the session's energy is
left undelivered. The comparison at matched undelivered energy (2 %) already penalises this, but the metric itself
still counts the excess. Here flexibility is recomputed with Phi_safe = min(theta, Phi*), which never exceeds the
oracle deferral, at the same quantile levels and the same 2 % operating point; settings A, C and D, both learners,
five seeds, driver-cluster bootstrap. Output: flexibility_conservative.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Ctx, CurveSet, bootstrap_table, curves, stable_group


def run(ctx: Ctx):
    rows = []
    for learner in ('catboost', 'ft'):
        js = ctx.joined(learner, 'ACD')
        if not js:
            continue
        drivers = np.unique(np.concatenate([j['user'].astype(str).to_numpy() for j in js.values()]))
        nd = len(drivers)
        sets = {}
        for seed, j in js.items():
            users = j['user'].astype(str).to_numpy(); di = np.searchsorted(drivers, users)
            D, E = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
            grp = stable_group(users)
            for s in 'ACD':
                U, EA, F, Fo = curves(D, E, j[f'pred_{s}'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float), grp, ctx.power)
                sets.setdefault(f'{s}_paper', CurveSet(nd)).add(U, EA, F, Fo, di)
                sets.setdefault(f'{s}_safe', CurveSet(nd)).add(U, EA, np.minimum(F, Fo[None]), Fo, di)
        contrasts = {}
        for m in ('paper', 'safe'):
            contrasts[f'loss_pct_{m}: 1 - D/C'] = (lambda mm: (lambda v: 100 * (1 - v[f'D_{mm}'] / v[f'C_{mm}'])))(m)
            contrasts[f'loss_pp_{m}: C - D'] = (lambda mm: (lambda v: v[f'C_{mm}'] - v[f'D_{mm}']))(m)
        contrasts['loss_pct safe minus paper (pp)'] = lambda v: 100 * (v['D_paper'] * v['C_safe'] - v['D_safe'] * v['C_paper']) / (v['C_paper'] * v['C_safe'])
        t = bootstrap_table(sets, nd, ctx.n_boot, contrasts=contrasts)
        t.insert(0, 'learner', learner)
        rows.append(t)
        ctx.log('flex_conservative ' + learner + ': ' + ', '.join(f"{q} {e:.2f}" for q, e in zip(t['quantity'], t['estimate'])))
    pd.concat(rows, ignore_index=True).to_csv(ctx.p('flexibility_conservative.csv'), index=False)
