"""Error direction, interval coverage and site weighting, without retraining.

error_metrics  from the cold-start shards, for every setting (A to E) and learner: MAE, bias, the share of
               under-forecasts (which drives undelivered energy), pinball loss of the cross-conformal
               0.1 and 0.9 quantile forecasts, and coverage and width of 90 % cross-conformal intervals;
               on all cohort sessions, on the evaluation sessions (last 20 %) and on first identified
               sessions. Also the share of variance between drivers (ICC) for duration and energy, which
               bears on why the duration and energy gaps differ between the networks.
protocols      from the protocol predictions of the main study (regression_predictions.parquet), only if
               they reproduce the manuscript's protocol MAEs: session-weighted against site-averaged
               site-disjoint error, per-site MAE with day-block CIs, and bias, under-forecasts and interval
               coverage by protocol. If they do not reproduce the manuscript, nothing is computed and a note
               says why.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .common import TARGETS, Ctx, cross_q_matrix, stable_group, write_json


def _pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def error_metrics(ctx: Ctx):
    rows, icc = [], []
    for learner in ('catboost', 'ft'):
        for target in TARGETS:
            sh = ctx.shards(learner, target)
            for seed, g in sh.groupby('seed'):
                y = g['y'].to_numpy(float); grp = stable_group(g['user'].astype(str))
                subsets = {'all cohort sessions': np.ones(len(g), bool), 'evaluation sessions (last 20 %)': g['is_T'].to_numpy(),
                           'first identified sessions (k = 0)': g['k'].to_numpy() == 0}
                for s in 'ABCDE':
                    p = g[f'pred_{s}'].to_numpy(float)
                    ok = np.isfinite(p)
                    if ok.sum() < 30:
                        continue
                    r = y - p
                    q = cross_q_matrix(np.where(ok, r, np.nan), grp, [0.1, 0.9, 0.95])
                    half = cross_q_matrix(np.where(ok, np.abs(r), np.nan), grp, [0.9])[0]
                    for name, m in subsets.items():
                        m = m & ok
                        if m.sum() < 30:
                            continue
                        rows.append({'learner': learner, 'target': target, 'seed': seed, 'setting': s, 'subset': name, 'n': int(m.sum()),
                                     'mae': np.abs(r[m]).mean(), 'bias_pred_minus_true': -r[m].mean(),
                                     'under_forecast_pct': 100 * np.mean(p[m] < y[m]),
                                     'pinball_q0.1': _pinball(y[m], (p + q[0])[m], 0.1), 'pinball_q0.9': _pinball(y[m], (p + q[1])[m], 0.9),
                                     'coverage_90_pct': 100 * np.mean(np.abs(r[m]) <= half[m]), 'mean_width_90': 2 * half[m].mean()})
            if learner == 'catboost':
                g = sh[sh['seed'] == sh['seed'].min()]
                yy, uu = g['y'].to_numpy(float), g['user'].astype(str).to_numpy()
                df = pd.DataFrame({'y': yy, 'u': uu})
                k = df.groupby('u')['y'].agg(['mean', 'var', 'size'])
                n = len(df); a = len(k); n0 = (n - (k['size'] ** 2).sum() / n) / (a - 1)
                msb = (k['size'] * (k['mean'] - yy.mean()) ** 2).sum() / (a - 1)
                msw = ((k['size'] - 1) * k['var'].fillna(0)).sum() / (n - a)
                icc.append({'target': target, 'drivers': a, 'sessions': n, 'icc1_between_driver_share': (msb - msw) / (msb + (n0 - 1) * msw),
                            'within_driver_sd': float(np.sqrt(msw)), 'total_sd': float(yy.std())})
    out = pd.DataFrame(rows)
    out = out.groupby(['learner', 'target', 'setting', 'subset']).mean(numeric_only=True).drop(columns='seed').reset_index()
    out.to_csv(ctx.p('error_direction_by_setting.csv'), index=False)
    pd.DataFrame(icc).to_csv(ctx.p('driver_variance_share.csv'), index=False)
    ctx.log('error metrics: ' + '; '.join(f"{r.target} {r.setting} under-forecast {r.under_forecast_pct:.1f}%"
                                          for r in out[(out.learner == 'catboost') & (out.subset == 'all cohort sessions')].itertuples()))


def protocols(ctx: Ctx):
    path = ctx.ncfg.get('regression_predictions')
    if path and not os.path.isabs(path):
        from .common import PKG
        path = str(PKG / path)
    exp = ctx.ncfg.get('expected_protocol_mae', {})
    note = ctx.p('protocols_NOTE.txt')
    from pathlib import Path
    if not path or not Path(path).exists():
        note.write_text(f'protocol predictions not found: {path}\n'); ctx.log(note.read_text().strip()); return
    cols = ['target', 'protocol', 'split_name', 'target_site', 'seed', 'model', 'row_id', 'date', 'y_true', 'y_pred', 'lower', 'upper']
    p = pd.read_parquet(path, columns=cols)
    if ctx.smoke:
        p = p[p['seed'] == p['seed'].min()]
    cb = p[p['model'] == 'catboost']
    per = cb.assign(ae=np.abs(cb['y_true'] - cb['y_pred'])).groupby(['target', 'protocol', 'split_name', 'seed'])['ae'].mean()
    mae = per.groupby(['target', 'protocol']).mean()
    check = []
    for key, v in exp.items():
        tgt, prot = key.split(':')
        got = float(mae.get((tgt, prot), np.nan))
        check.append({'target': tgt, 'protocol': prot, 'manuscript': v, 'reproduced': got, 'ok': abs(got - v) <= 0.015})
    check = pd.DataFrame(check); check.to_csv(ctx.p('protocols_check.csv'), index=False)
    if not ctx.smoke and (check.empty or not check['ok'].all()):
        note.write_text('The protocol predictions do not reproduce the manuscript; no protocol tables were computed.\n'
                        + check.to_string(index=False) + '\n')
        ctx.log(note.read_text()); return
    p['err'] = p['y_true'] - p['y_pred']
    rows = []
    for (tgt, prot, model), g in p.groupby(['target', 'protocol', 'model']):
        cov = g[g['lower'].notna() & g['upper'].notna()]
        rows.append({'target': tgt, 'protocol': prot, 'model': model, 'n_per_seed': len(g) / g['seed'].nunique(),
                     'mae': g['err'].abs().mean(), 'bias_pred_minus_true': -g['err'].mean(),
                     'under_forecast_pct': 100 * (g['y_pred'] < g['y_true']).mean(),
                     'coverage_pct': 100 * ((cov['y_true'] >= cov['lower']) & (cov['y_true'] <= cov['upper'])).mean() if len(cov) else np.nan,
                     'mean_width': (cov['upper'] - cov['lower']).mean() if len(cov) else np.nan})
    pd.DataFrame(rows).to_csv(ctx.p('protocols_error_direction.csv'), index=False)
    # site-disjoint: per-site MAE (seed-averaged per session) with day-block CIs; session-weighted vs site average
    site = p[p['protocol'] == 'site_disjoint']
    srows, rng = [], np.random.default_rng(5)
    for (tgt, model, ts), g in site.groupby(['target', 'model', 'target_site']):
        s = g.assign(ae=g['err'].abs()).groupby(['row_id', 'date'], as_index=False)['ae'].mean()
        days, di = np.unique(s['date'], return_inverse=True)
        e_ = np.bincount(di, weights=s['ae']); n_ = np.bincount(di).astype(float)
        bs = [e_ @ w / (n_ @ w) for w in (np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(ctx.n_boot))]
        srows.append({'target': tgt, 'model': model, 'target_site': ts, 'sessions': len(s), 'mae': s['ae'].mean(),
                      'ci_low': np.percentile(bs, 2.5), 'ci_high': np.percentile(bs, 97.5)})
    sd = pd.DataFrame(srows)
    if not sd.empty:
        agg = sd.groupby(['target', 'model']).apply(lambda x: pd.Series({
            'site_average_mae': x['mae'].mean(), 'session_weighted_mae': np.average(x['mae'], weights=x['sessions']),
            'sites': len(x)})).reset_index()
        sd.to_csv(ctx.p('protocols_site_disjoint_by_site.csv'), index=False)
        agg.to_csv(ctx.p('protocols_site_weighting.csv'), index=False)
    ctx.log('protocols: tables written')


def run(ctx: Ctx):
    error_metrics(ctx)
    protocols(ctx)
