"""Results read at the operating point of the whole population, the reliability check, and protocol errors by period.

subsets        Subset results read at the operating point of the whole population: the quantile level is chosen on
               all cohort sessions (2 % undelivered energy) and every subset (first identified sessions, later
               sessions, sessions with observed and censored energy need) is evaluated at that level, reporting both
               its flexibility and its own undelivered energy. An operator sets one level for the site and cannot
               choose it per subset. The subset-specific reading (each subset at its own 2 % level) is kept alongside.
reliability    Empirical check of P(D_hat > D) <= gamma, P(E_hat < E_ach) <= gamma and hence P(U > 0) <= 2 gamma
               (Proposition 1 of the paper) for the cross-conformal shifts (five driver groups), per setting.
periods        Protocol errors by period with day-block confidence intervals: before and after 1 March 2020 (ACN);
               before the Palo Alto fees (1 August 2017), with fees before 1 March 2020, and after 1 March 2020
               (Palo Alto). The optimism of each protocol is compared with random splitting on test sessions of
               the same period.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .common import (TARGET_UNMET, TARGETS, Ctx, CurveSet, alap, boot_weights, curves, mask_history, shifted,
                     stable_group)


# ============================================================================================ subsets
def _arrays(cs: CurveSet, key, w):
    Uc, EAc, Fc, Foc, _ = cs.items[key]
    ww = np.ones(cs.nc) if w is None else w
    return 100 * (Uc @ ww) / (EAc @ ww), 100 * (Fc @ ww) / (Foc @ ww)


def _ref_position(unmet_ref: np.ndarray, target: float = TARGET_UNMET) -> float:
    """Fractional position on the quantile grid at which the reference reaches the target undelivered share."""
    if not (np.nanmin(unmet_ref) <= target <= np.nanmax(unmet_ref)):
        return np.nan
    o = np.argsort(unmet_ref, kind='stable')
    return float(np.interp(target, unmet_ref[o], np.arange(len(unmet_ref))[o]))


def fixed_level_values(sets: dict, w=None) -> dict:
    """For every curve set 'S_SUB': flexibility and undelivered share at the level where 'S_ALL' reaches 2 %,
    and flexibility at the subset's own 2 % level; averaged over seeds."""
    out = {}
    for name, cs in sets.items():
        s, sub = name.split('_', 1)
        ref = sets[f'{s}_ALL']
        flex_fix, unmet_fix = [], []
        for key in cs.items:
            u_r, _ = _arrays(ref, key, w)
            u_s, f_s = _arrays(cs, key, w)
            pos = _ref_position(u_r)
            idx = np.arange(len(u_s))
            flex_fix.append(np.interp(pos, idx, f_s) if np.isfinite(pos) else np.nan)
            unmet_fix.append(np.interp(pos, idx, u_s) if np.isfinite(pos) else np.nan)
        out[f'{name}|flex_fixed'] = float(np.nanmean(flex_fix))
        out[f'{name}|unmet_fixed'] = float(np.nanmean(unmet_fix))
        out[f'{name}|flex_own'] = cs.value(w)
    return out


def _boot(sets: dict, nd: int, B: int, contrasts: dict, seed: int = 20261123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    point = fixed_level_values(sets)
    cpt = {k: f(point) for k, f in contrasts.items()}
    draws = {k: [] for k in point}; cdraws = {k: [] for k in contrasts}
    for _ in range(B):
        v = fixed_level_values(sets, boot_weights(rng, nd))
        for k in point:
            draws[k].append(v[k])
        for k, f in contrasts.items():
            cdraws[k].append(f(v))
    kinds = {'flex_fixed': 'flex_pct_of_oracle at population level', 'unmet_fixed': 'undelivered_pct at population level',
             'flex_own': 'flex_pct_of_oracle at subset level (first run)'}
    rows = []
    for k, est in point.items():
        x = np.asarray(draws[k], float); x = x[np.isfinite(x)]
        name, what = k.split('|')
        rows.append({'quantity': name, 'kind': kinds[what], 'estimate': est,
                     'ci_low': np.percentile(x, 2.5) if len(x) else np.nan, 'ci_high': np.percentile(x, 97.5) if len(x) else np.nan,
                     'p_boot': np.nan})
    for k, est in cpt.items():
        x = np.asarray(cdraws[k], float); x = x[np.isfinite(x)]
        p = max(2 * min(np.mean(x <= 0), np.mean(x >= 0)), 1 / (B + 1)) if len(x) else np.nan
        rows.append({'quantity': k, 'kind': 'contrast at population level', 'estimate': est,
                     'ci_low': np.percentile(x, 2.5) if len(x) else np.nan, 'ci_high': np.percentile(x, 97.5) if len(x) else np.nan,
                     'p_boot': p})
    return pd.DataFrame(rows)


def run_subsets(ctx: Ctx):
    from .charging import session_physics
    phys = session_physics(ctx)
    F = lambda v, k: v[f'{k}|flex_fixed']                                                    # noqa: E731
    U = lambda v, k: v[f'{k}|unmet_fixed']                                                   # noqa: E731
    tabs = []
    for learner in ('catboost', 'ft'):
        js = ctx.joined(learner, 'ACD')
        drivers = np.unique(np.concatenate([j['user'].astype(str).to_numpy() for j in js.values()]))
        nd = len(drivers)
        fv, ch = {}, {}
        for seed, j in js.items():
            users = j['user'].astype(str).to_numpy(); di = np.searchsorted(drivers, users); grp = stable_group(users)
            D, E = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
            k = j['k'].to_numpy()
            unc = phys.reindex(j.index.astype(str))['uncensored'].fillna(False).to_numpy(bool)
            for s in 'ACD':
                c = curves(D, E, j[f'pred_{s}'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float), grp, ctx.power)
                for name, m in {'ALL': None, 'FV': k == 0, 'RET': k >= 1}.items():
                    fv.setdefault(f'{s}_{name}', CurveSet(nd)).add(*c, di, m, key=seed)
                for name, m in {'ALL': None, 'UNCENSORED': unc, 'CENSORED': ~unc}.items():
                    ch.setdefault(f'{s}_{name}', CurveSet(nd)).add(*c, di, m, key=seed)
        c_fv = {'first visit: D_FV - D_RET (pp)': lambda v: F(v, 'D_FV') - F(v, 'D_RET'),
                'first visit: D_FV - D_ALL (pp)': lambda v: F(v, 'D_FV') - F(v, 'D_ALL'),
                'first visit: undelivered D_FV - 2 % (pp)': lambda v: U(v, 'D_FV') - TARGET_UNMET,
                'first visit: observed loss 1 - D_FV/C_RET (%)': lambda v: 100 * (1 - F(v, 'D_FV') / F(v, 'C_RET'))}
        c_ch = {}
        for sub in ('UNCENSORED', 'CENSORED'):
            c_ch[f'{sub.lower()}: loss 1 - D/C (%)'] = (lambda sb: (lambda v: 100 * (1 - F(v, f'D_{sb}') / F(v, f'C_{sb}'))))(sub)
            c_ch[f'{sub.lower()}: loss C - D (pp)'] = (lambda sb: (lambda v: F(v, f'C_{sb}') - F(v, f'D_{sb}')))(sub)
            c_ch[f'{sub.lower()}: undelivered D - C (pp)'] = (lambda sb: (lambda v: U(v, f'D_{sb}') - U(v, f'C_{sb}')))(sub)
        for label, sets, con in (('first identified sessions', fv, c_fv), ('censoring of the energy need', ch, c_ch)):
            t = _boot(sets, nd, ctx.n_boot, con)
            t.insert(0, 'analysis', label); t.insert(0, 'learner', learner)
            tabs.append(t)
            ctx.log(f'subsets {learner} {label}: ' + ', '.join(f"{q} {e:.2f}" for q, e in zip(t['quantity'], t['estimate'])
                                                              if q in con))
    out = pd.concat(tabs, ignore_index=True)
    out.to_csv(ctx.p('subsets_population_level.csv'), index=False)


# ======================================================================================== reliability
GAMMAS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2)


def run_reliability(ctx: Ctx):
    rows = []
    for learner in ('catboost', 'ft'):
        for seed, j in ctx.joined(learner, 'ACD').items():
            users = j['user'].astype(str).to_numpy(); grp = stable_group(users)
            D, E = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
            E_ach = np.minimum(E, ctx.power * D)
            for s in 'ACD':
                Dh, Eh = shifted(D, E, j[f'pred_{s}'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float), grp, GAMMAS)
                _, Uu, _ = alap(D, E, Dh, Eh, ctx.power)
                for i, g in enumerate(GAMMAS):
                    rows.append({'learner': learner, 'seed': seed, 'setting': s, 'gamma': g,
                                 'P(D_hat>D) %': 100 * np.mean(Dh[i] > D), 'P(E_hat<E_ach) %': 100 * np.mean(Eh[i] < E_ach - 1e-9),
                                 'P(U>0) %': 100 * np.mean(Uu[i] > 1e-6), 'P(U>0.1 kWh) %': 100 * np.mean(Uu[i] > 0.1),
                                 'bound gamma %': 100 * g, 'bound 2 gamma %': 200 * g,
                                 'undelivered share %': 100 * Uu[i].sum() / E_ach.sum()})
    r = pd.DataFrame(rows)
    keys = ['learner', 'setting', 'gamma']
    agg = r.drop(columns='seed').groupby(keys).mean()
    agg['P(U>0) % sd over seeds'] = r.groupby(keys)['P(U>0) %'].std()
    agg['seeds'] = r.groupby(keys)['seed'].nunique()
    agg = agg.reset_index()
    agg['P(U>0) within 2 gamma'] = agg['P(U>0) %'] <= agg['bound 2 gamma %']
    agg.to_csv(ctx.p('reliability_two_gamma.csv'), index=False)
    ctx.log('reliability: ' + ', '.join(f"{x.setting} g{x.gamma}: P(U>0) {x['P(U>0) %']:.1f}% (<= {x['bound 2 gamma %']:.1f}%)"
                                        for _, x in agg[agg['learner'] == 'catboost'].iterrows() if x['gamma'] in (0.01, 0.05)))


# ============================================================================================ periods
def _period_breaks(ctx: Ctx):
    if 'period_breaks' in ctx.ncfg:
        return [pd.Timestamp(b) for b in ctx.ncfg['period_breaks']]
    if ctx.net == 'paloalto':
        return [pd.Timestamp(ctx.ncfg.get('fee_date', '2017-08-01')), pd.Timestamp('2020-03-01')]
    return [pd.Timestamp('2020-03-01')]


def _period_labels(breaks):
    labels = [f'before {breaks[0]:%d %b %Y}']
    for a, b in zip(breaks[:-1], breaks[1:]):
        labels.append(f'{a:%d %b %Y} to {b - pd.Timedelta(days=1):%d %b %Y}')
    labels.append(f'from {breaks[-1]:%d %b %Y}')
    return labels


def _day_boot(dates: np.ndarray, ae: np.ndarray, B: int, rng):
    days, di = np.unique(dates, return_inverse=True)
    e_ = np.bincount(di, weights=ae); n_ = np.bincount(di).astype(float)
    W = np.stack([np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(B)])
    return (W @ e_) / (W @ n_)


def run_periods(ctx: Ctx):
    from pathlib import Path
    path = ctx.ncfg.get('regression_predictions')
    if path and not os.path.isabs(path):
        from .common import PKG
        path = str(PKG / path)
    chk = ctx.p('protocols_check.csv')
    if not path or not Path(path).exists():
        ctx.log(f'periods: protocol predictions not found ({path}); skipped'); return
    if chk.exists() and not ctx.smoke:
        c = pd.read_csv(chk)
        if c.empty or not c['ok'].all():
            ctx.log('periods: the protocol predictions do not reproduce the manuscript; skipped'); return
    cols = ['target', 'protocol', 'split_name', 'target_site', 'seed', 'model', 'row_id', 'date', 'y_true', 'y_pred']
    p = pd.read_parquet(path, columns=cols)
    p = p[p['model'].isin(['catboost', 'torch_ft'])]
    if ctx.smoke:
        p = p[p['seed'] == p['seed'].min()]
    p['ae'] = (p['y_true'] - p['y_pred']).abs()
    p['site'] = np.where(p['protocol'] == 'site_disjoint', p['target_site'].astype(str), 'all')
    # one row per session: errors averaged over seeds, as in the paper
    s = p.groupby(['target', 'protocol', 'site', 'model', 'row_id', 'date'], as_index=False)['ae'].mean()
    s['day'] = pd.to_datetime(s['date']).dt.tz_localize(None).dt.normalize()
    breaks = _period_breaks(ctx); labels = _period_labels(breaks)
    s['period'] = pd.cut(s['day'], bins=[pd.Timestamp.min] + breaks + [pd.Timestamp.max], right=False, labels=labels)
    rng = np.random.default_rng(23)
    rows, draws = [], {}
    for (tgt, prot, site, model, per), g in s.groupby(['target', 'protocol', 'site', 'model', 'period'], observed=True):
        if len(g) < 30:
            continue
        bs = _day_boot(g['day'].to_numpy(), g['ae'].to_numpy(), ctx.n_boot, rng)
        draws[(tgt, prot, site, model, per)] = bs
        rows.append({'target': tgt, 'protocol': prot, 'site': site, 'model': model, 'period': per, 'sessions': len(g),
                     'days': g['day'].nunique(), 'mae': g['ae'].mean(), 'ci_low': np.percentile(bs, 2.5), 'ci_high': np.percentile(bs, 97.5)})
    r = pd.DataFrame(rows)
    gap, lo, hi = [], [], []
    for x in r.itertuples():
        base = r[(r['target'] == x.target) & (r['protocol'] == 'random') & (r['model'] == x.model) & (r['period'] == x.period)]
        if x.protocol == 'random' or base.empty:
            gap.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        b = base.iloc[0]
        ratio = draws[(x.target, x.protocol, x.site, x.model, x.period)] / draws[(b.target, 'random', 'all', b.model, b.period)]
        gap.append(100 * (x.mae / b.mae - 1)); lo.append(100 * (np.percentile(ratio, 2.5) - 1)); hi.append(100 * (np.percentile(ratio, 97.5) - 1))
    r['optimism_vs_random_same_period_pct'] = gap; r['opt_ci_low'] = lo; r['opt_ci_high'] = hi
    r.to_csv(ctx.p('protocols_by_period.csv'), index=False)
    ctx.log('periods: ' + '; '.join(f"{x.target} {x.protocol} {x.period}: {x.mae:.2f}" for x in r[(r['model'] == 'catboost') & (r['site'] == 'all')].itertuples()))
