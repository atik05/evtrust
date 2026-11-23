"""Learned cold-start baselines: can a cold-start prior recover what the driver's history gives?

In each driver-grouped fold the unseen-driver model of the paper (setting C) is retrained exactly as in the
cold-start decomposition, and the hidden history of a test session is replaced by a learned prior from the
training drivers instead of being left missing (setting D):
  D_knn  similarity prior: mean history of the 50 most similar training sessions in plug-in context
         (arrival time, weekday, month, station history, request), training sessions with history only
  D_reg  regression prior: each history feature predicted from the plug-in context by gradient boosting
         trained on the training drivers' sessions with history
D_rerun is setting D refitted in this run (a check against the manuscript's D). Accuracy on the paper's
evaluation sessions (each driver's last 20 %) and flexibility at 2 % undelivered energy on all cohort
sessions, with a driver-cluster bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

from evtrust.coldstart import USER_HISTORY

from .common import TARGETS, Ctx, CurveSet, bootstrap_table, cluster_ratio, curves, mask_history, stable_group
from .decomp import _base, _folds

CONTEXT = ['arrival_hour_sin', 'arrival_hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
           'station_prior_mean_duration', 'station_prior_mean_energy', 'requested_energy_kwh', 'requested_duration_hours']
K_NEIGHBOURS = 50


def _priors(Xtr: pd.DataFrame, Xq: pd.DataFrame, seed: int):
    """History features for query rows from the training sessions that have history."""
    hist = Xtr[Xtr['user_prior_count'].fillna(0) >= 1]
    ctx_cols = [c for c in CONTEXT if c in Xtr and hist[c].notna().any()]      # Palo Alto has no requests
    med = hist[ctx_cols].median()
    A = np.nan_to_num(hist[ctx_cols].fillna(med).to_numpy(float)); Q = np.nan_to_num(Xq[ctx_cols].fillna(med).to_numpy(float))
    mu, sd = A.mean(0), A.std(0) + 1e-9
    nn = NearestNeighbors(n_neighbors=min(K_NEIGHBOURS, len(A))).fit((A - mu) / sd)
    _, idx = nn.kneighbors((Q - mu) / sd)
    H = hist[USER_HISTORY].to_numpy(float)
    knn = Xq.copy(); reg = Xq.copy()
    knn[USER_HISTORY] = np.nanmean(H[idx], axis=1)
    for j, c in enumerate(USER_HISTORY):
        ok = np.isfinite(H[:, j])
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=seed).fit(hist[ctx_cols][ok], H[ok, j])
        reg[c] = m.predict(Xq[ctx_cols])
    return knn, reg


def unit(ctx: Ctx, learner: str, target: str, seed: int) -> pd.DataFrame:
    path = ctx.p('shards', f'csb_{learner}_{target}_{seed}.parquet')
    if path.exists():
        return pd.read_parquet(path)
    d = ctx.cohort(target)
    cols = ctx.numeric + ['station_code']
    X, y = d[cols], d[target].to_numpy(float)
    fit = ctx.fitter(learner, seed)
    _, gf = _folds(d, seed)
    pred = {k: np.full(len(d), np.nan) for k in ('C_rerun', 'D_rerun', 'D_knn', 'D_reg')}
    for tr, te in gf:
        m = fit(X.iloc[tr], y[tr])
        Xq = X.iloc[te]
        knn, reg = _priors(X.iloc[tr], Xq, seed)
        pred['C_rerun'][te] = m.predict(Xq)
        pred['D_rerun'][te] = m.predict(mask_history(Xq))
        pred['D_knn'][te] = m.predict(knn)
        pred['D_reg'][te] = m.predict(reg)
    out = _base(d, target, learner, seed)
    for k, v in pred.items():
        out[f'pred_{k}'] = v
    out.to_parquet(path, index=False)
    return out


def run(ctx: Ctx):
    learners = ctx.cfg.get('cs_baselines', {}).get('learners', ['catboost'])
    for learner in learners:
        for target in TARGETS:
            for seed in ctx.seeds:
                unit(ctx, learner, target, seed)
                ctx.log(f'cold-start baselines {learner} {target} seed {seed} done')
        summarise(ctx, learner)


def summarise(ctx: Ctx, learner: str):
    import glob
    rows = []
    new = {t: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ctx.p('shards') / f'csb_{learner}_{t}_*.parquet')))])
           for t in TARGETS}
    # accuracy on the evaluation sessions, seeds averaged per session as in the paper
    for target in TARGETS:
        sh = ctx.shards(learner, target)
        newcols = [c for c in new[target].columns if c.startswith('pred_') and c != 'pred_A']
        z = sh.merge(new[target][['seed', 'session_key'] + newcols], on=['seed', 'session_key'])
        names = ['C', 'D'] + [c[5:] for c in newcols]
        for c in names:
            z[f'err_{c}'] = np.abs(z['y'] - z[f'pred_{c}'])
        m = z.groupby(['user', 'session_index', 'k', 'is_T'], as_index=False)[[f'err_{c}' for c in names]].mean()
        Cr = 'err_C_rerun' if 'err_C_rerun' in m else 'err_C'
        # the priors are compared with the cold start refitted from the same models (D_rerun) and with C from them
        for subset, mm in (('evaluation sessions (last 20 %)', m[m['is_T']]), ('first identified sessions (k = 0)', m[m['k'] == 0])):
            for a, b in [('err_D', 'err_D_rerun'), ('err_D_rerun', 'err_D_knn'), ('err_D_rerun', 'err_D_reg'), (Cr, 'err_D_knn'), (Cr, 'err_D_reg'), (Cr, 'err_D_rerun')]:
                if len(mm) < 30:
                    continue
                est, lo, hi, p = cluster_ratio(mm[a].to_numpy(), mm[b].to_numpy(), mm['user'].to_numpy(), ctx.n_boot)
                rows.append({'kind': 'mae_change_pct', 'target': target, 'subset': subset, 'from': a[4:], 'to': b[4:],
                             'mae_from': mm[a].mean(), 'mae_to': mm[b].mean(), 'estimate': est, 'ci_low': lo, 'ci_high': hi,
                             'p_value': p, 'n_rows': len(mm)})
    # flexibility at 2 % undelivered energy on all cohort sessions
    js = ctx.joined(learner, 'CD')
    drivers = np.unique(np.concatenate([j['user'].astype(str).to_numpy() for j in js.values()]))
    nd = len(drivers); sets = {}
    nd_ = {t: new[t].set_index(['seed', 'session_key']) for t in TARGETS}
    for seed, j in js.items():
        users = j['user'].astype(str).to_numpy(); di = np.searchsorted(drivers, users); grp = stable_group(users)
        D, E = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
        keys = pd.MultiIndex.from_arrays([np.full(len(j), seed), j.index.astype(str)])
        fc = {'C': (j['pred_C'].to_numpy(float), j['pred_C_e'].to_numpy(float)),
              'D': (j['pred_D'].to_numpy(float), j['pred_D_e'].to_numpy(float))}
        for v in ('C_rerun', 'D_knn', 'D_reg', 'D_rerun'):
            if f'pred_{v}' in nd_['duration_hours']:
                fc[v] = (nd_['duration_hours'][f'pred_{v}'].reindex(keys).to_numpy(float), nd_['kwh_delivered'][f'pred_{v}'].reindex(keys).to_numpy(float))
        for v, (Dp, Ep) in fc.items():
            sets.setdefault(v, CurveSet(nd)).add(*curves(D, E, Dp, Ep, grp, ctx.power), di, key=seed)
    cref = 'C_rerun' if 'C_rerun' in sets else 'C'
    contrasts = {'D_knn - D_rerun (pp)': lambda v: v['D_knn'] - v['D_rerun'], 'D_reg - D_rerun (pp)': lambda v: v['D_reg'] - v['D_rerun'],
                 'share of the C-D loss recovered by D_knn (%)': lambda v: 100 * (v['D_knn'] - v['D_rerun']) / (v[cref] - v['D_rerun']),
                 'share of the C-D loss recovered by D_reg (%)': lambda v: 100 * (v['D_reg'] - v['D_rerun']) / (v[cref] - v['D_rerun']),
                 'check: D_rerun - D manuscript (pp)': lambda v: v['D_rerun'] - v['D']}
    t = bootstrap_table(sets, nd, ctx.n_boot, contrasts=contrasts)
    t['kind'] = np.where(t['kind'] == 'contrast', 'flex_at_2pct contrast', 'flex_at_2pct')
    out = pd.concat([pd.DataFrame(rows), t], ignore_index=True)
    out.insert(0, 'learner', learner)
    out.to_csv(ctx.p(f'coldstart_baselines_{learner}.csv'), index=False)
    ctx.log(f'cold-start baselines {learner}: ' + ', '.join(f"{q}={e:.2f}" for q, e in zip(t['quantity'], t['estimate'])))
