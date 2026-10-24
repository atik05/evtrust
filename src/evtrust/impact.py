"""Energy-system impact of prediction error under deadline-aware smart charging.

Each session is scheduled as late as possible: charging starts at
    s = predicted_departure - predicted_energy / P
so that the predicted energy is delivered just before the predicted departure,
which leaves the most room for the grid operator. Once started, the vehicle
charges at P until it is full or leaves.

    achievable energy   E_ach = min(E, P * D)
    delivered           min(E_ach, P * max(0, D - s))
    unmet energy        E_ach - delivered
    flexibility         hours of deferral min(s, D)

An oracle that knows D and E defers by D - E_ach / P and leaves nothing unmet.
Shifting the departure forecast down and the energy forecast up by conformal
quantiles trades flexibility for reliability; the trade-off curve is the
energy-system counterpart of the coverage results.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .coldstart import k_bucket


def simulate(D, E, D_hat, E_hat, power_kw):
    D = np.asarray(D, float); E = np.asarray(E, float)
    E_ach = np.minimum(E, power_kw * D)
    s = np.clip(np.asarray(D_hat, float) - np.asarray(E_hat, float) / power_kw, 0.0, None)
    delivered = np.minimum(E_ach, power_kw * np.clip(D - s, 0.0, None))
    return E_ach, E_ach - delivered, np.minimum(s, D)


def _cross_quantile(resid: np.ndarray, groups: np.ndarray, q: float) -> np.ndarray:
    """Quantile of residuals from OTHER driver groups (cross-conformal, no self-use)."""
    out = np.empty(len(resid))
    for g in np.unique(groups):
        sel = groups == g
        other = resid[~sel & np.isfinite(resid)]
        out[sel] = np.quantile(other, q) if len(other) else 0.0
    return out


# Quantile levels that trace each trade-off curve: 0.5 is the raw point forecast, then 0.495 down to 0.005 in steps
# of 0.005. Flexibility at a reliability target is interpolated between neighbouring levels, so the grid must be fine
# enough that the result no longer depends on it: halving the step changes no value by more than 0.05 percentage
# points on ACN, whereas the earlier 11-level grid understated the cold-start flexibility where its curve is concave.
TAUS = (0.5,) + tuple(round(0.495 - 0.005 * i, 3) for i in range(99))


def run(dur: pd.DataFrame, eng: pd.DataFrame, power_kw: float = 6.6,
        taus=TAUS, settings=('A', 'C', 'D')) -> tuple:
    """dur, eng: cold-start row outputs for duration and energy (same learner)."""
    rows, by_k = [], []
    for seed in sorted(set(dur['seed']) & set(eng['seed'])):
        a = dur[dur['seed'] == seed].set_index('session_key')
        b = eng[eng['seed'] == seed].set_index('session_key')
        j = a.join(b[[f'pred_{s}' for s in settings]], rsuffix='_e', how='inner')
        D, E = j['duration_true'].to_numpy(), j['energy_true'].to_numpy()
        users = j['user'].astype(str).to_numpy()
        grp = np.array([zlib.crc32(u.encode()) % 5 for u in users])   # stable across runs (hash() is salted)
        kb = k_bucket(j['k'].to_numpy())
        E_ach_o, unmet_o, flex_o = simulate(D, E, D, np.minimum(E, power_kw * D), power_kw)
        cands = {}
        for s in settings:
            Dp, Ep = j[f'pred_{s}'].to_numpy(), j[f'pred_{s}_e'].to_numpy()
            for tau in taus:
                qd = _cross_quantile(D - Dp, grp, tau)          # lower quantile shifts departure earlier
                qe = _cross_quantile(E - Ep, grp, 1 - tau)      # upper quantile shifts energy up
                if tau == 0.5:
                    qd = np.zeros_like(qd); qe = np.zeros_like(qe)   # tau 0.5 = raw point forecast
                cands[(s, tau)] = (np.maximum(Dp + qd, 0), np.maximum(Ep + qe, 0))
        rq = j[['requested_duration', 'requested_energy']].to_numpy()
        has_req = np.isfinite(rq).all(1)
        if has_req.any():
            cands[('driver_request', 0.5)] = (np.where(has_req, rq[:, 0], np.nan), np.where(has_req, rq[:, 1], np.nan))
        for (s, tau), (Dh, Eh) in cands.items():
            ok = np.isfinite(Dh) & np.isfinite(Eh)
            E_ach, unmet, flex = simulate(D[ok], E[ok], Dh[ok], Eh[ok], power_kw)
            rows.append({'seed': seed, 'setting': s, 'tau': tau, 'n': int(ok.sum()),
                         'unmet_share_pct': 100 * unmet.sum() / E_ach.sum(),
                         'sessions_unmet_over_1kwh_pct': 100 * (unmet > 1).mean(),
                         'flex_hours_mean': flex.mean(),
                         'flex_vs_oracle_pct': 100 * flex.sum() / flex_o[ok].sum()})
            if tau in (0.5, 0.1):
                for kk in pd.unique(kb[ok]):
                    sel = kb[ok] == kk
                    if sel.sum() >= 30:
                        by_k.append({'seed': seed, 'setting': s, 'tau': tau, 'k_bucket': kk, 'n': int(sel.sum()),
                                     'unmet_share_pct': 100 * unmet[sel].sum() / E_ach[sel].sum(),
                                     'flex_vs_oracle_pct': 100 * flex[sel].sum() / flex_o[ok][sel].sum()})
    res = pd.DataFrame(rows)
    summ = (res.groupby(['setting', 'tau']).agg(unmet_share_pct=('unmet_share_pct', 'mean'),
                                                 unmet_sd=('unmet_share_pct', 'std'),
                                                 sessions_unmet_over_1kwh_pct=('sessions_unmet_over_1kwh_pct', 'mean'),
                                                 flex_hours_mean=('flex_hours_mean', 'mean'),
                                                 flex_vs_oracle_pct=('flex_vs_oracle_pct', 'mean'),
                                                 n=('n', 'mean'), seeds=('seed', 'nunique')).reset_index())
    # flexibility retained at matched reliability, interpolated along each seed's trade-off curve
    matched = []
    for (seed, setting), g in res[res['setting'] != 'driver_request'].groupby(['seed', 'setting']):
        g = g.sort_values('unmet_share_pct')
        for target in (1.0, 2.0, 5.0):
            if g['unmet_share_pct'].min() <= target <= g['unmet_share_pct'].max():
                f = float(np.interp(target, g['unmet_share_pct'], g['flex_vs_oracle_pct']))
                matched.append({'seed': seed, 'setting': setting, 'unmet_target_pct': target, 'flex_vs_oracle_pct': f})
    matched = pd.DataFrame(matched)
    if not matched.empty:
        matched = matched.groupby(['setting', 'unmet_target_pct']).agg(
            flex_vs_oracle_pct=('flex_vs_oracle_pct', 'mean'), flex_sd=('flex_vs_oracle_pct', 'std'),
            seeds=('seed', 'nunique')).reset_index()
    bk = pd.DataFrame(by_k)
    if not bk.empty:
        bk = bk.groupby(['setting', 'tau', 'k_bucket']).agg(unmet_share_pct=('unmet_share_pct', 'mean'),
                                                             flex_vs_oracle_pct=('flex_vs_oracle_pct', 'mean'),
                                                             n=('n', 'mean')).reset_index()
    return summ, bk, matched, res
