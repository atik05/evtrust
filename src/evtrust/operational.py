"""Operational analyses that reuse saved out-of-sample predictions (no retraining).

1. request_matched  Forecasts against the driver's own request on the identical sessions where a
                    request exists, at matched reliability, by driver history.
2. site_level       Site-level coordinated charging under a capacity limit: how much grid capacity
                    each forecast needs to keep undelivered energy small.
3. bias_correction  Target-site bias correction of point forecasts at an unseen site from the first
                    m completed sessions there.

All quantile shifts are cross-conformal over driver groups formed with a stable hash, so results are
identical across runs.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


STEP_H = 0.25                                   # 15-minute scheduling step
from .impact import TAUS   # the same quantile levels as the smart-charging evaluation


def stable_group(values, n: int = 5) -> np.ndarray:
    """Deterministic group id per driver (Python's hash() is salted per process)."""
    return np.array([zlib.crc32(str(v).encode()) % n for v in values])


def cross_quantile(resid: np.ndarray, groups: np.ndarray, q: float) -> np.ndarray:
    out = np.empty(len(resid))
    for g in np.unique(groups):
        sel = groups == g
        other = resid[~sel & np.isfinite(resid)]
        out[sel] = np.quantile(other, q) if len(other) else 0.0
    return out


def alap(D, E, D_hat, E_hat, power):
    E_ach = np.minimum(E, power * D)
    s = np.clip(D_hat - E_hat / power, 0.0, None)
    delivered = np.minimum(E_ach, power * np.clip(D - s, 0.0, None))
    return E_ach, E_ach - delivered, np.minimum(s, D)


def join_targets(rows: pd.DataFrame, settings=('A', 'C', 'D')) -> pd.DataFrame:
    """One row per (seed, session) with duration and energy forecasts side by side."""
    dur = rows[rows['target'] == 'duration_hours']
    eng = rows[rows['target'] == 'kwh_delivered']
    keep = ['seed', 'session_key'] + [f'pred_{s}' for s in settings]
    j = dur.merge(eng[keep], on=['seed', 'session_key'], suffixes=('', '_e'), how='inner')
    return j


# ----------------------------------------------------------------------------- 1. request
def _driver_sums(j: pd.DataFrame, settings, taus, power):
    """Per-driver sums of (unmet, E_ach, flex, oracle flex) for every setting and tau, averaged over seeds."""
    users = j['user'].astype(str).to_numpy()
    uniq, uid = np.unique(users, return_inverse=True)
    out = np.zeros((len(settings), len(taus), len(uniq), 4))
    seeds = sorted(j['seed'].unique())
    for seed in seeds:
        z = j['seed'].to_numpy() == seed
        D, E = j['duration_true'].to_numpy()[z], j['energy_true'].to_numpy()[z]
        grp = stable_group(users[z])
        _, _, flex_o = alap(D, E, D, np.minimum(E, power * D), power)
        for si, s in enumerate(settings):
            if s == 'request':
                Dp, Ep = j['requested_duration'].to_numpy()[z], j['requested_energy'].to_numpy()[z]
            else:
                Dp, Ep = j[f'pred_{s}'].to_numpy()[z], j[f'pred_{s}_e'].to_numpy()[z]
            for ti, tau in enumerate(taus):
                qd = np.zeros(len(D)) if tau == 0.5 else cross_quantile(D - Dp, grp, tau)
                qe = np.zeros(len(D)) if tau == 0.5 else cross_quantile(E - Ep, grp, 1 - tau)
                E_ach, unmet, flex = alap(D, E, np.maximum(Dp + qd, 0), np.maximum(Ep + qe, 0), power)
                for c, v in enumerate((unmet, E_ach, flex, flex_o)):
                    out[si, ti, :, c] += np.bincount(uid[z], weights=v, minlength=len(uniq)) / len(seeds)
    return out


def _matched(sums: np.ndarray, w: np.ndarray, targets):
    """Flexibility (% of oracle) at matched undelivered-energy shares, for weights w over drivers."""
    agg = np.tensordot(sums, w, axes=([2], [0]))          # settings x taus x 4
    unmet = 100 * agg[..., 0] / agg[..., 1]
    flex = 100 * agg[..., 2] / agg[..., 3]
    res = np.full((sums.shape[0], len(targets)), np.nan)
    for si in range(sums.shape[0]):
        o = np.argsort(unmet[si])
        for k, t in enumerate(targets):
            if unmet[si][o][0] <= t <= unmet[si][o][-1]:
                res[si, k] = np.interp(t, unmet[si][o], flex[si][o])
    return res, unmet, flex


def request_matched(rows: pd.DataFrame, power: float = 6.6, taus=TAUS, targets=(1.0, 2.0, 5.0),
                    n_boot: int = 2000, seed: int = 0) -> dict[str, pd.DataFrame]:
    settings = ('A', 'C', 'D', 'request')
    j = join_targets(rows)
    has = (np.isfinite(j['requested_duration']) & np.isfinite(j['requested_energy'])
           & (j['requested_duration'] > 0) & (j['requested_energy'] > 0))
    j = j[has].reset_index(drop=True)
    kb = np.where(j['k'].to_numpy() == 0, 'first visit',
                  np.where(j['k'].to_numpy() <= 4, '1-4 sessions', '5+ sessions'))
    rng = np.random.default_rng(seed)
    curves, matched, diffs = [], [], []
    for group in ['all', 'first visit', '1-4 sessions', '5+ sessions']:
        g = j if group == 'all' else j[kb == group]
        if g['user'].nunique() < 20:
            continue
        sums = _driver_sums(g, settings, taus, power)
        n_drv = sums.shape[2]
        point, unmet, flex = _matched(sums, np.ones(n_drv), targets)
        for si, s in enumerate(settings):
            for ti, tau in enumerate(taus):
                curves.append({'group': group, 'setting': s, 'tau': tau, 'unmet_share_pct': unmet[si, ti],
                               'flex_vs_oracle_pct': flex[si, ti]})
        boot = np.empty((n_boot,) + point.shape)
        for b in range(n_boot):
            boot[b] = _matched(sums, rng.multinomial(n_drv, np.full(n_drv, 1 / n_drv)).astype(float), targets)[0]
        for si, s in enumerate(settings):
            for k, t in enumerate(targets):
                lo, hi = np.nanpercentile(boot[:, si, k], [2.5, 97.5]) if np.isfinite(boot[:, si, k]).sum() > 100 else (np.nan, np.nan)
                matched.append({'group': group, 'setting': s, 'unmet_target_pct': t, 'flex_vs_oracle_pct': point[si, k],
                                'ci_low': lo, 'ci_high': hi, 'n_sessions_per_seed': len(g) // g['seed'].nunique(),
                                'n_drivers': n_drv})
        ri = settings.index('request')
        for si, s in enumerate(settings[:-1]):
            for k, t in enumerate(targets):
                d = boot[:, ri, k] - boot[:, si, k]
                ok = np.isfinite(d)
                if ok.sum() < 100:
                    continue
                p = min(1.0, 2 * (min((d[ok] <= 0).sum(), (d[ok] >= 0).sum()) + 1) / (ok.sum() + 1))
                diffs.append({'group': group, 'comparison': f'request - {s}', 'unmet_target_pct': t,
                              'difference_pct_points': point[ri, k] - point[si, k],
                              'ci_low': np.percentile(d[ok], 2.5), 'ci_high': np.percentile(d[ok], 97.5), 'p_value': p})
    return {'request_curves': pd.DataFrame(curves), 'request_matched': pd.DataFrame(matched),
            'request_differences': pd.DataFrame(diffs)}


# -------------------------------------------------------------------------- 2. site level
def _simulate_site(a, d, E, base, cap, power, D_hat_steps=None, E_hat=None, order=None):
    """Least-laxity-first charging of controlled sessions under cap minus the uncontrolled base load.

    a, d       arrival and departure steps (departure exclusive), E true energy need (kWh)
    D_hat_steps, E_hat   forecasts used for priority; None means first-come-first-served (order)
    Returns undelivered energy per session and the controlled load per step.
    """
    n_steps = len(base)
    remaining = E.astype(float).copy()
    delivered_pred = np.zeros(len(E))
    load = np.zeros(n_steps)
    max_step = power * STEP_H
    starts = np.argsort(a, kind='stable')
    ptr, active = 0, np.empty(0, dtype=int)
    for t in range(n_steps):
        new = []
        while ptr < len(starts) and a[starts[ptr]] <= t:
            new.append(starts[ptr]); ptr += 1
        if new:
            active = np.concatenate([active, np.array(new, dtype=int)])
        if active.size:
            active = active[(d[active] > t) & (remaining[active] > 1e-9)]   # drop departed or full sessions
        if active.size == 0:
            continue
        room = max(cap - base[t], 0.0) * STEP_H
        if room <= 0:
            continue
        if D_hat_steps is None:
            prio = order[active].astype(float)
        else:
            rem_pred = np.maximum(E_hat[active] - delivered_pred[active], 0.0) / max_step
            prio = (a[active] + D_hat_steps[active] - t) - rem_pred
            prio = np.where(rem_pred > 0, prio, 1e12 + prio)     # forecast says full: lowest priority
        for i in active[np.argsort(prio, kind='stable')]:
            give = min(max_step, remaining[i], room)
            if give <= 0:
                break
            remaining[i] -= give; delivered_pred[i] += give; room -= give; load[t] += give / STEP_H
    return remaining, load


def site_level(rows: pd.DataFrame, sessions: pd.DataFrame, power: float = 6.6,
               fractions=tuple(np.round(np.arange(0.10, 0.6001, 0.025), 3)), n_boot: int = 1000,
               seed: int = 0, sites=None) -> dict[str, pd.DataFrame]:
    j = join_targets(rows)
    fc = (j.groupby('session_key')[['pred_A', 'pred_C', 'pred_D', 'pred_A_e', 'pred_C_e', 'pred_D_e']].mean()
          .join(j.groupby('session_key')[['requested_duration', 'requested_energy']].first()))
    s = sessions.copy()
    s['session_key'] = s['session_id'].astype(str)
    s['t0'] = pd.to_datetime(s['connection_time'], utc=True)
    s['t1'] = pd.to_datetime(s['disconnect_time'], utc=True)
    s = s[s['kwh_delivered'].notna() & (s['t1'] > s['t0'])]
    s = s.join(fc, on='session_key')
    s['controlled'] = s['pred_A'].notna()
    rows_out, needed = [], []
    rng = np.random.default_rng(seed)
    for site, g in s.groupby('site'):
        if sites and site not in sites:
            continue
        ctl = g[g['controlled']]
        if len(ctl) < 200:
            continue
        lo, hi = ctl['t0'].min(), ctl['t0'].max()
        g = g[(g['t0'] >= lo) & (g['t0'] <= hi)].sort_values('t0').reset_index(drop=True)
        origin = lo.floor('D')
        a = np.floor((g['t0'] - origin).dt.total_seconds() / 3600 / STEP_H).astype(int).to_numpy()
        d = np.maximum(np.ceil((g['t1'] - origin).dt.total_seconds() / 3600 / STEP_H).astype(int).to_numpy(), a + 1)
        E = g['kwh_delivered'].to_numpy(float)
        Dh = (g['t1'] - g['t0']).dt.total_seconds().to_numpy() / 3600
        E_ach = np.minimum(E, power * (d - a) * STEP_H)
        n_steps = int(d.max()) + 1
        unc = ~g['controlled'].to_numpy()
        base = np.zeros(n_steps)
        for i in np.where(unc)[0]:                                    # uncontrolled: as fast as possible
            left = E[i]
            for t in range(a[i], d[i]):
                give = min(power * STEP_H, left)
                if give <= 0:
                    break
                base[t] += give / STEP_H; left -= give
        full = base.copy()
        for i in np.where(~unc)[0]:
            left = E[i]
            for t in range(a[i], d[i]):
                give = min(power * STEP_H, left)
                if give <= 0:
                    break
                full[t] += give / STEP_H; left -= give
        peak = full.max()
        c = ~unc
        ac, dc, Ec = a[c], d[c], E[c]
        day = (ac * STEP_H // 24).astype(int)
        ndays = day.max() + 1
        policies = {'first come first served': None, 'oracle': (Dh[c], np.minimum(E, power * Dh)[c])}
        for st in ['A', 'C', 'D']:
            policies[st] = (g[f'pred_{st}'].to_numpy()[c], g[f'pred_{st}_e'].to_numpy()[c])
        rq_d, rq_e = g['requested_duration'].to_numpy()[c], g['requested_energy'].to_numpy()[c]
        has = np.isfinite(rq_d) & np.isfinite(rq_e) & (rq_d > 0) & (rq_e > 0)
        policies['request (D otherwise)'] = (np.where(has, rq_d, policies['D'][0]), np.where(has, rq_e, policies['D'][1]))
        daily = {}
        for name, fcst in policies.items():
            for f in fractions:
                cap = f * peak
                if fcst is None:
                    rem, load = _simulate_site(ac, dc, Ec, base, cap, power, order=np.arange(len(ac)))
                else:
                    Dhs = np.maximum(fcst[0], 0) / STEP_H
                    rem, load = _simulate_site(ac, dc, Ec, base, cap, power, Dhs, np.maximum(fcst[1], 0))
                und = np.clip(rem - (Ec - E_ach[c]), 0, None)        # energy that could have been delivered but was not
                rows_out.append({'site': site, 'policy': name, 'cap_fraction': f, 'cap_kw': cap, 'uncontrolled_peak_kw': peak,
                                 'undelivered_share_pct': 100 * und.sum() / E_ach[c].sum(),
                                 'sessions_short_over_1kwh_pct': 100 * (und > 1).mean(),
                                 'base_exceeds_cap_steps_pct': 100 * (base > cap).mean(),
                                 'n_controlled': int(c.sum()), 'n_uncontrolled': int(unc.sum())})
                daily[(name, f)] = (np.bincount(day, weights=und, minlength=ndays),
                                    np.bincount(day, weights=E_ach[c], minlength=ndays))
        # capacity needed to keep undelivered energy at or below 1 and 2 percent, with a day-block bootstrap
        names = list(policies)
        fr = np.array(fractions)
        boots = rng.integers(0, ndays, size=(n_boot, ndays))
        for target in (1.0, 2.0):
            point, draws = {}, {}
            for name in names:
                U = np.stack([daily[(name, f)][0] for f in fr]); A = np.stack([daily[(name, f)][1] for f in fr])
                share = 100 * U.sum(1) / A.sum(1)
                point[name] = _first_cap(fr, share, target)
                bs = 100 * U[:, boots].sum(2) / A[:, boots].sum(2)          # fractions x n_boot
                draws[name] = np.array([_first_cap(fr, bs[:, b], target) for b in range(n_boot)])
            for name in names:
                dd = draws[name]
                needed.append({'site': site, 'policy': name, 'undelivered_target_pct': target,
                               'capacity_needed_fraction': point[name],
                               'ci_low': np.nanpercentile(dd, 2.5) if np.isfinite(dd).any() else np.nan,
                               'ci_high': np.nanpercentile(dd, 97.5) if np.isfinite(dd).any() else np.nan})
            for x, y in [('D', 'A'), ('D', 'oracle'), ('request (D otherwise)', 'D'), ('first come first served', 'A')]:
                diff = draws[x] - draws[y]
                ok = np.isfinite(diff)
                needed.append({'site': site, 'policy': f'{x} minus {y}', 'undelivered_target_pct': target,
                               'capacity_needed_fraction': point[x] - point[y],
                               'ci_low': np.percentile(diff[ok], 2.5) if ok.sum() > 100 else np.nan,
                               'ci_high': np.percentile(diff[ok], 97.5) if ok.sum() > 100 else np.nan})
    return {'site_level': pd.DataFrame(rows_out), 'site_capacity_needed': pd.DataFrame(needed)}


def _first_cap(fr: np.ndarray, share: np.ndarray, target: float) -> float:
    """Smallest capacity fraction with undelivered share <= target, linearly interpolated."""
    ok = share <= target
    if not ok.any():
        return np.nan
    k = int(np.argmax(ok))
    if k == 0:
        return float(fr[0])
    x0, x1, y0, y1 = fr[k - 1], fr[k], share[k - 1], share[k]
    return float(x0 + (x1 - x0) * (y0 - target) / (y0 - y1)) if y0 != y1 else float(fr[k])


# ---------------------------------------------------------------------- 3. bias correction
def bias_correction(pred: pd.DataFrame, sessions: pd.DataFrame, budgets=(0, 10, 25, 50, 100, 250),
                    models=('catboost', 'torch_ft', 'ridge', 'user_history', 'historical_mean')) -> pd.DataFrame:
    """Shift unseen-site forecasts by the residual of the first m completed target-site sessions.

    The median residual corrects absolute error and the mean residual corrects squared error. All
    budgets are evaluated on the same sessions: those that begin after every adaptation session has
    ended, so no label is used before it exists.
    """
    disc = pd.to_datetime(sessions['disconnect_time'], utc=True)
    out = []
    for (seed, split, target), g in pred.groupby(['seed', 'split_name', 'target']):
        hm = g[g['model'] == 'historical_mean']
        if hm.empty:
            continue
        train_mean = float(hm['y_pred'].iloc[0])
        for model in models:
            z = g[g['model'] == model].copy()
            if len(z) < 200:
                continue
            z['t0'] = pd.to_datetime(z['connection_time'], utc=True)
            z['t1'] = disc.reindex(z['row_id'].to_numpy()).to_numpy()
            z = z.sort_values('t0').reset_index(drop=True)
            m_max = min(max(budgets), len(z) // 4)
            ends = z['t1'].iloc[:m_max].max()
            ev = z[z['t0'] > ends]
            if len(ev) < 100:
                continue
            y, p0 = ev['y_true'].to_numpy(float), ev['y_pred'].to_numpy(float)
            sst = ((y - y.mean()) ** 2).sum(); sso = ((y - train_mean) ** 2).sum()
            for m in [b for b in budgets if b <= m_max]:
                r = (z['y_true'] - z['y_pred']).to_numpy(float)[:m]
                p_med = p0 + (np.median(r) if m else 0.0)
                p_mean = p0 + (np.mean(r) if m else 0.0)
                out.append({'seed': seed, 'split_name': split, 'target_site': split.split(':')[-1], 'target': target,
                            'model': model, 'budget': m, 'n_eval': len(ev),
                            'mae': np.abs(y - p_med).mean(), 'rmse': np.sqrt(((y - p_mean) ** 2).mean()),
                            'r2': 1 - ((y - p_mean) ** 2).sum() / sst, 'r2_oos': 1 - ((y - p_mean) ** 2).sum() / sso})
    return pd.DataFrame(out)
