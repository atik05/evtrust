"""Robustness of the smart-charging results to how the conformal shift is calibrated.

    python scripts/calibration_robustness.py --results runs/acn/results --sessions runs/acn/data/processed/sessions.parquet \
        --network ACN --out runs/acn/results/paper/calibration_robustness.csv

The main analysis (evtrust.impact) shifts each forecast by a quantile of residuals of the other driver groups from
all periods (cross-conformal), which assumes that residuals are exchangeable over time. The prospective alternative
uses only residuals of other driver groups from sessions completed before the month of the session's arrival, as an
operator could have collected them. Both are scored on the same sessions: those with at least MIN_POOL earlier
residuals of other driver groups (the first months are left out of both).
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from evtrust.impact import simulate, _cross_quantile, TAUS   # the grid of the main analysis
TARGETS = (1.0, 2.0, 5.0)
MIN_POOL = 200
SETTINGS = {'A': 'seen in training', 'C': 'own history', 'D': 'cold start'}


def prospective_quantile(resid, groups, q, arrival_month, done_month):
    out = np.full(len(resid), np.nan)
    for m in np.unique(arrival_month):
        before = (done_month < m) & np.isfinite(resid)
        for g in np.unique(groups):
            sel = (arrival_month == m) & (groups == g)
            pool = resid[before & (groups != g)]
            if sel.any() and len(pool) >= MIN_POOL:
                out[sel] = np.quantile(pool, q)
    return out


def flex_at(curve: pd.DataFrame, target: float) -> float:
    """As in the main analysis: interpolated when the target lies within the curve, otherwise not reported."""
    c = curve.sort_values('unmet_pct')
    if not (c['unmet_pct'].min() <= target <= c['unmet_pct'].max()):
        return float('nan')
    return float(np.interp(target, c['unmet_pct'], c['flex_pct']))


def session_times(sessions: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Arrival time per cold-start row: by session_id when the sessions carry it, else by driver, duration and
    energy (sessions sharing all three are left without a time)."""
    if 'session_id' in sessions.columns:
        s = sessions.drop_duplicates('session_id').set_index(sessions['session_id'].astype(str))['connection_time']
        return rows['session_key'].astype(str).map(s)
    key = lambda u, d, e: u.astype(str) + '|' + d.round(6).astype(str) + '|' + e.round(6).astype(str)
    s = sessions.dropna(subset=['user_id']).assign(k=lambda x: key(x['user_id'], x['duration_hours'], x['kwh_delivered']))
    s = s[~s['k'].duplicated(keep=False)].set_index('k')['connection_time']
    return key(rows['user'], rows['duration_true'], rows['energy_true']).map(s)


def run(results: Path, sessions: pd.DataFrame, network: str, power: float = 6.6) -> pd.DataFrame:
    shards = results / 'paper' / 'shards'
    out = []
    for learner in ('catboost', 'ft'):
        seeds = sorted(int(f.stem.rsplit('_', 1)[1]) for f in shards.glob(f'cs_{learner}_duration_hours_*.parquet'))
        for seed in seeds:
            du = pd.read_parquet(shards / f'cs_{learner}_duration_hours_{seed}.parquet')
            en = pd.read_parquet(shards / f'cs_{learner}_kwh_delivered_{seed}.parquet')
            j = du.merge(en[['session_key', 'pred_A', 'pred_C', 'pred_D']], on='session_key', suffixes=('_d', '_e'))
            j = j[np.isfinite(j['duration_true']) & np.isfinite(j['energy_true'])].reset_index(drop=True)
            t = pd.to_datetime(session_times(sessions, j), utc=True)
            timed = t.notna().to_numpy()
            j, t = j[timed].reset_index(drop=True), t[timed].reset_index(drop=True)
            end = t + pd.to_timedelta(j['duration_true'], unit='h')
            am = (t.dt.year * 12 + t.dt.month).to_numpy(); dm = (end.dt.year * 12 + end.dt.month).to_numpy()
            groups = np.array([zlib.crc32(str(u).encode()) % 5 for u in j['user']])
            common = np.isfinite(prospective_quantile(np.zeros(len(j)), groups, 0.5, am, dm))
            D, E = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
            for s, label in SETTINGS.items():
                Dp, Ep = j[f'pred_{s}_d'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float)
                for method in ('all periods (main analysis)', 'prospective'):
                    rows = []
                    for tau in TAUS:
                        if tau == 0.5:                     # the raw point forecast, as in the main analysis
                            qd = qe = np.zeros(int(common.sum()))
                        elif method.startswith('all'):
                            qd = _cross_quantile((D - Dp)[common], groups[common], tau)
                            qe = _cross_quantile((E - Ep)[common], groups[common], 1 - tau)
                        else:
                            qd = prospective_quantile(D - Dp, groups, tau, am, dm)[common]
                            qe = prospective_quantile(E - Ep, groups, 1 - tau, am, dm)[common]
                        Dg = np.maximum(Dp[common] + qd, 0.0); Eg = np.maximum(Ep[common] + qe, 0.0)
                        Ea, un, fl = simulate(D[common], E[common], Dg, Eg, power)
                        _, _, fo = simulate(D[common], E[common], D[common], np.minimum(E[common], power * D[common]), power)
                        rows.append({'unmet_pct': 100 * un.sum() / Ea.sum(), 'flex_pct': 100 * fl.sum() / fo.sum()})
                    c = pd.DataFrame(rows)
                    out.append({'network': network, 'learner': learner, 'seed': seed, 'setting': s, 'setting_label': label,
                                'calibration': method, 'sessions_timed': int(timed.sum()), 'sessions_scored': int(common.sum()),
                                **{f'flex_at_{x:g}pct': flex_at(c, x) for x in TARGETS}})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--sessions', required=True)
    ap.add_argument('--network', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    s = pd.read_parquet(a.sessions)
    r = run(Path(a.results), s, a.network)
    r.to_csv(a.out, index=False)
    m = r.groupby(['learner', 'calibration', 'setting'])['flex_at_2pct'].mean().unstack('setting')
    m['cold-start loss %'] = 100 * (1 - m['D'] / m['C'])
    print(f"{a.network}: {r['sessions_scored'].iloc[0]:,} of {r['sessions_timed'].iloc[0]:,} timed sessions scored")
    print(m.round(2).to_string())


if __name__ == '__main__':
    main()
