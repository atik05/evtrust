"""Forecasting the scheduling quantity directly: laxity L = D - min(E, P D) / P (the latest start).

The paper forecasts departure and energy separately and starts charging at D_hat - E_hat / P. Here the
start time is forecast directly, (i) by a point model shifted down by cross-conformal residual quantiles
(the same 100-level grid) and (ii) by quantile regression at fixed levels. If the cold-start loss of
flexibility remained with a forecast of the decision variable itself, it is not an artefact of combining
two forecasts. Settings A (random split), C (unseen driver with own history) and D (history hidden), on
the cohort sessions with both labels; compared on the same sessions with the separate forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Ctx, CurveSet, bootstrap_table, cross_q_matrix, curves, interp_target, mask_history, stable_group
from .decomp import _folds
from evtrust.impact import TAUS


def lax_unit(ctx: Ctx, seed: int, quantile: bool):
    learner = ctx.cfg['laxity']['learner']
    path = ctx.p('shards', f'laxity_{learner}_{seed}.parquet')
    if path.exists():
        return pd.read_parquet(path)
    d = ctx.cohort('kwh_delivered')
    d = d[d['duration_hours'].notna()].reset_index(drop=True)
    P = ctx.power
    D, E = d['duration_hours'].to_numpy(float), d['kwh_delivered'].to_numpy(float)
    L = np.clip(D - np.minimum(E, P * D) / P, 0.0, None)
    cols = ctx.numeric + ['station_code']
    X, Xm = d[cols], mask_history(d[cols])
    kf, gf = _folds(d, seed)
    out = pd.DataFrame({'session_key': d['session_id'].astype(str).to_numpy(), 'user': d['user_id'].astype(str).to_numpy(),
                        'D': D, 'E': E, 'L': L, 'seed': seed})
    alphas = list(ctx.cfg['laxity']['quantile_alphas']) if quantile else []
    for name, loss in [('point', None)] + [(f'q{a}', f'Quantile:alpha={a}') for a in alphas]:
        fit = ctx.fitter(learner, seed, loss=loss)
        pa, pc, pd_ = (np.full(len(d), np.nan) for _ in range(3))
        for tr, te in kf:
            pa[te] = fit(X.iloc[tr], L[tr]).predict(X.iloc[te])
        for tr, te in gf:
            m = fit(X.iloc[tr], L[tr]); pc[te] = m.predict(X.iloc[te]); pd_[te] = m.predict(Xm.iloc[te])
        out[f'{name}_A'], out[f'{name}_C'], out[f'{name}_D'] = pa, pc, pd_
        ctx.log(f'2g seed {seed} {name} done')
    out.to_parquet(path, index=False)
    return out


def _sched(D, E, s, P):
    E_ach = np.minimum(E, P * D)
    delivered = np.minimum(E_ach, P * np.clip(D - s, 0.0, None))
    return E_ach - delivered, np.minimum(s, D)


def run(ctx: Ctx):
    nq = int(ctx.cfg['laxity'].get('quantile_seeds', len(ctx.seeds)))
    units = {seed: lax_unit(ctx, seed, quantile=i < nq) for i, seed in enumerate(ctx.seeds)}
    js = ctx.joined('catboost', 'ACD')
    drivers = np.unique(np.concatenate([u['user'].to_numpy() for u in units.values()]))
    nd = len(drivers); P = ctx.power
    sets, qrows = {}, []
    for seed, u in units.items():
        D, E, L = u['D'].to_numpy(), u['E'].to_numpy(), u['L'].to_numpy()
        users = u['user'].to_numpy(); di = np.searchsorted(drivers, users); grp = stable_group(users)
        E_ach = np.minimum(E, P * D); Fo = np.minimum(np.clip(D - E_ach / P, 0, None), D)
        taus = np.asarray(TAUS)
        for s in 'ACD':
            Lp = u[f'point_{s}'].to_numpy()
            q = cross_q_matrix(L - Lp, grp, taus); q[np.isclose(taus, 0.5)] = 0.0
            U, F = _sched(D, E, np.clip(Lp[None] + q, 0.0, None), P)
            sets.setdefault(f'laxity point+conformal {s}', CurveSet(nd)).add(U, E_ach, F, Fo, di, key=seed)
            # the paper's separate forecasts on the same sessions
            if seed in js:
                j = js[seed].reindex(u['session_key'])
                cv = curves(D, E, j[f'pred_{s}'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float), grp, P)
                sets.setdefault(f'separate D,E (paper) {s}', CurveSet(nd)).add(*cv, di, key=seed)
            qcols = sorted([c for c in u.columns if c.startswith('q') and c.endswith(f'_{s}')], key=lambda c: -float(c[1:-2]))
            if qcols:
                for method in ('quantile regression', 'conformalized quantile regression'):
                    um, fm = [], []
                    for c in qcols:
                        a_ = float(c[1:-2]); Lq = u[c].to_numpy()
                        if method.startswith('conformalized'):          # one-sided cross-conformal correction at level alpha
                            Lq = Lq + cross_q_matrix(L - Lq, grp, [a_])[0]
                        Uq, Fq = _sched(D, E, np.clip(Lq, 0.0, None), P)
                        um.append(100 * Uq.sum() / E_ach.sum()); fm.append(100 * Fq.sum() / Fo.sum())
                        qrows.append({'seed': seed, 'setting': s, 'method': method, 'alpha': a_, 'undelivered_share_pct': um[-1],
                                      'flex_pct_of_oracle': fm[-1], 'mae_laxity_point': float(np.abs(L - u[f'point_{s}']).mean())})
                    qrows.append({'seed': seed, 'setting': s, 'method': method, 'alpha': 'interpolated at 2 %', 'undelivered_share_pct': 2.0,
                                  'flex_pct_of_oracle': interp_target(np.array(um), np.array(fm))})
    contrasts = {f'loss_pct {m}: 1 - D/C': (lambda mm: (lambda v: 100 * (1 - v[f'{mm} D'] / v[f'{mm} C'])))(m)
                 for m in ('laxity point+conformal', 'separate D,E (paper)')}
    t = bootstrap_table(sets, nd, ctx.n_boot, contrasts=contrasts)
    t.to_csv(ctx.p('laxity_flex.csv'), index=False)
    if qrows:
        q = pd.DataFrame(qrows); q['alpha'] = q['alpha'].astype(str)
        q.groupby(['method', 'setting', 'alpha']).agg(undelivered_share_pct=('undelivered_share_pct', 'mean'),
                                           flex_pct_of_oracle=('flex_pct_of_oracle', 'mean'),
                                           seeds=('seed', 'nunique')).to_csv(ctx.p('laxity_quantile_models.csv'))
    ctx.log('2g: ' + ', '.join(f"{a}={b:.2f}" for a, b in zip(t['quantity'], t['estimate'])))
