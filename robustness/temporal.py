"""Rolling-origin evaluation: is the new-driver gap stable over time (and across the Palo Alto fee)?

For each origin, a model is trained on every session that had ended (plus a 4-hour embargo) before the
origin and tested on the sessions that start in the following window. Test sessions are split into
returning drivers (seen in training), new identified drivers (not seen in training) and anonymous
sessions, and accuracy is compared between them with a day-block bootstrap. Features are the paper's
plug-in features; history is past-only by construction. Descriptive statistics by period are added.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import TARGETS, Ctx


def run(ctx: Ctx):
    s = ctx.sessions().copy()
    s['t0'] = pd.to_datetime(s['connection_time'], utc=True)
    s['t1'] = pd.to_datetime(s['disconnect_time'], utc=True)
    s['uid'] = s['user_id'].astype(str)
    s['ident'] = s['user_id'].notna() & (s['uid'].str.strip() != '') & (s['uid'] != '<NA>')
    s['station_code'] = s['station_id'].astype('category').cat.codes
    tz = 'UTC'
    origins = ctx.ncfg.get('temporal_origins', [])
    if ctx.smoke:
        origins = origins[-2:]
    horizon = int(ctx.ncfg.get('temporal_horizon_months', 3))
    emb = pd.Timedelta(hours=float(ctx.cfg['temporal'].get('embargo_hours', 4)))
    seeds = ctx.seeds[: int(ctx.cfg['temporal'].get('seeds', 1))]
    cols = ctx.numeric + ['station_code']
    rows, desc = [], []
    for o in origins:
        t_o = pd.Timestamp(o, tz=tz); t_e = t_o + pd.DateOffset(months=horizon)
        for target in TARGETS:
            tr = s[(s['t1'] < t_o - emb) & s[target].notna()]
            te = s[(s['t0'] >= t_o) & (s['t0'] < t_e) & s[target].notna()].copy()
            if len(tr) < 500 or len(te) < 100:
                ctx.log(f'2h origin {o} {target}: too few sessions (train {len(tr)}, test {len(te)}), skipped')
                continue
            seen = set(tr.loc[tr['ident'], 'uid'])
            te['group'] = np.where(~te['ident'], 'anonymous', np.where(te['uid'].isin(seen), 'returning driver', 'new identified driver'))
            preds = []
            for seed in seeds:
                m = ctx.fitter('catboost', seed)(tr[cols], tr[target].to_numpy(float))
                preds.append(m.predict(te[cols]))
            te['err'] = np.abs(te[target].to_numpy(float) - np.mean(preds, 0))
            te['day'] = te['t0'].dt.strftime('%Y-%m-%d')
            days = np.array(sorted(te['day'].unique())); di = np.searchsorted(days, te['day'])
            rng = np.random.default_rng(3)
            boots = [np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(ctx.n_boot)]
            sums = {}
            for g in ('returning driver', 'new identified driver', 'anonymous'):
                sel = (te['group'] == g).to_numpy()
                sums[g] = (np.bincount(di[sel], weights=te['err'].to_numpy()[sel], minlength=len(days)),
                           np.bincount(di[sel], minlength=len(days)).astype(float))
            for g, (e_, n_) in sums.items():
                if n_.sum() < 30:
                    continue
                r = {'origin': o, 'test_end': str(t_e.date()), 'target': target, 'group': g, 'n_test': int(n_.sum()),
                     'n_train': len(tr), 'mae': e_.sum() / n_.sum(), 'seeds': len(seeds)}
                ref_e, ref_n = sums['returning driver']
                if g != 'returning driver' and ref_n.sum() >= 30:
                    ratio = lambda w: (e_ @ w / (n_ @ w)) / (ref_e @ w / (ref_n @ w)) - 1
                    x = np.array([ratio(w) for w in boots])
                    r.update(gap_vs_returning_pct=100 * ratio(np.ones(len(days))),
                             ci_low=100 * np.nanpercentile(x, 2.5), ci_high=100 * np.nanpercentile(x, 97.5))
                rows.append(r)
            ctx.log(f'2h origin {o} {target}: ' + ', '.join(f"{r['group']} {r['mae']:.3f}" for r in rows if r['origin'] == o and r['target'] == target))
        w = s[(s['t0'] >= t_o) & (s['t0'] < t_e)]
        desc.append({'origin': o, 'test_end': str(t_e.date()), 'sessions': len(w), 'anonymous_share': float((~w['ident']).mean()),
                     'mean_duration_h': w['duration_hours'].mean(), 'median_duration_h': w['duration_hours'].median(),
                     'mean_energy_kwh': w['kwh_delivered'].mean(), 'drivers': int(w.loc[w['ident'], 'uid'].nunique())})
    pd.DataFrame(rows).to_csv(ctx.p('temporal_rolling_origin.csv'), index=False)
    pd.DataFrame(desc).to_csv(ctx.p('temporal_descriptive.csv'), index=False)
    fee = ctx.ncfg.get('fee_date')
    if fee:
        t_f = pd.Timestamp(fee, tz=tz)
        z = s[(s['t0'] >= t_f - pd.DateOffset(years=1)) & (s['t0'] < t_f + pd.DateOffset(years=1))]
        z = z.assign(period=np.where(z['t0'] < t_f, 'year before fees', 'year after fees'))
        z.groupby('period').agg(sessions=('session_id', 'size'), mean_duration_h=('duration_hours', 'mean'),
                                median_duration_h=('duration_hours', 'median'), mean_energy_kwh=('kwh_delivered', 'mean'),
                                anonymous_share=('ident', lambda v: 1 - v.mean())).to_csv(ctx.p('temporal_fee_periods.csv'))
