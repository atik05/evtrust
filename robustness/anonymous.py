"""Anonymous sessions and the network-level deployment mix.

The decomposition cohort contains identified drivers with at least five sessions. The remaining sessions of
the same window are anonymous (no driver identifier: a permanent cold start) or belong to rarely seen
identified drivers. A model trained on all cohort sessions predicts them; no driver of these sessions is in
training. Outputs:
  anonymous_mae.csv           accuracy by population against the cohort settings
  anonymous_flex.csv          flexibility at 2 % undelivered energy by population and for the whole network
                              (benchmark: cohort random split; deployment: cohort unseen-driver with own history;
                              cold: all history hidden), with a stratified cluster bootstrap (drivers, days)
  anonymous_transfer.csv      reliability when anonymous sessions are calibrated with identified drivers' residuals
  deploy_forecasts_*.parquet  forecasts for every session of the window, used by the ACN-Sim study
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (TARGETS, Ctx, CurveSet, alap, bootstrap_table, curves, mask_history, stable_group)


def _ident(s: pd.Series) -> np.ndarray:
    return (s.notna() & (s.astype(str).str.strip() != '') & (s.astype(str) != '<NA>')).to_numpy()


def fill_unit(ctx: Ctx, learner: str, target: str, seed: int) -> pd.DataFrame:
    path = ctx.p('shards', f'fill_{learner}_{target}_{seed}.parquet')
    if path.exists():
        return pd.read_parquet(path)
    coh = ctx.cohort(target)
    win = ctx.window(target)
    rest = win[~win['session_id'].astype(str).isin(set(coh['session_id'].astype(str)))].copy()
    codes = dict(zip(coh['station_id'].astype(str), coh['station_code']))
    rest['station_code'] = rest['station_id'].astype(str).map(codes).fillna(-1).astype(int)
    cols = ctx.numeric + ['station_code']
    model = ctx.fitter(learner, seed)(coh[cols], coh[target].to_numpy(float))
    ident = _ident(rest['user_id'])
    out = pd.DataFrame({
        'session_key': rest['session_id'].astype(str).to_numpy(),
        'user': np.where(ident, rest['user_id'].astype(str), ''),
        'population': np.where(ident, 'rare_identified', 'anonymous'),
        'site': rest['site'].to_numpy(), 'day': rest['t'].dt.strftime('%Y-%m-%d').to_numpy(),
        'week': rest['t'].dt.strftime('%G-W%V').to_numpy(),
        'arrival_hour': (pd.to_datetime(rest['connection_time']).dt.hour + pd.to_datetime(rest['connection_time']).dt.minute / 60).to_numpy(float),
        'k': rest['user_prior_count'].fillna(0).to_numpy(float), 'y': rest[target].to_numpy(float),
        'duration_true': rest['duration_hours'].to_numpy(float), 'energy_true': rest['kwh_delivered'].to_numpy(float),
        'requested_duration': rest['requested_duration_hours'].to_numpy(float),
        'requested_energy': rest['requested_energy_kwh'].to_numpy(float),
        'pred': model.predict(rest[cols]), 'pred_cold': model.predict(mask_history(rest[cols])),
    })
    out['learner'], out['target'], out['seed'] = learner, target, seed
    out.to_parquet(path, index=False)
    return out


def _fill_joined(ctx, learner, seed):
    a = fill_unit(ctx, learner, 'duration_hours', seed).set_index('session_key')
    b = fill_unit(ctx, learner, 'kwh_delivered', seed).set_index('session_key')
    return a.join(b[['pred', 'pred_cold']], rsuffix='_e', how='inner')


def run(ctx: Ctx):
    for learner in ctx.cfg['anonymous']['learners']:
        for seed in ctx.seeds:
            for target in TARGETS:
                fill_unit(ctx, learner, target, seed)
            ctx.log(f'2b fill {learner} seed {seed} done')
        summarise(ctx, learner)


def summarise(ctx: Ctx, learner: str):
    js = ctx.joined(learner, 'ACD')
    s_all = ctx.sessions(); ct = pd.to_datetime(s_all['connection_time'])
    hour = pd.Series((ct.dt.hour + ct.dt.minute / 60).to_numpy(float), index=s_all['session_id'].astype(str))
    fills = {seed: _fill_joined(ctx, learner, seed) for seed in js}
    # cluster universe: drivers (identified) and days (anonymous); stratified resampling
    keys = set()
    for seed in js:
        keys |= set('u:' + js[seed]['user'].astype(str))
        f = fills[seed]
        keys |= set(np.where(f['population'] == 'anonymous', 'd:' + f['day'], 'u:' + f['user']))
    keys = np.array(sorted(keys)); nc = len(keys)
    strata = np.array([k.startswith('d:') for k in keys]).astype(int)
    cid_of = lambda arr: np.searchsorted(keys, arr)
    sets, mae, transfer, deploy = {}, [], [], []
    for seed, j in js.items():
        f = fills[seed]
        # ---- cohort
        Dc, Ec = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
        uc = j['user'].astype(str).to_numpy(); gc = stable_group(uc); cc = cid_of('u:' + uc)
        cohort_curves = {}
        for s in 'ACD':
            cohort_curves[s] = curves(Dc, Ec, j[f'pred_{s}'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float), gc, ctx.power)
            sets.setdefault(f'cohort_{s}', CurveSet(nc)).add(*cohort_curves[s], cc, key=seed)
            mae.append({'seed': seed, 'population': 'cohort', 'forecast': s, 'n': len(Dc),
                        'mae_energy': float(np.abs(Ec - j[f'pred_{s}_e']).mean()), 'mae_duration': float(np.abs(Dc - j[f'pred_{s}']).mean()),
                        'bias_energy': float((j[f'pred_{s}_e'] - Ec).mean()), 'bias_duration': float((j[f'pred_{s}'] - Dc).mean()),
                        'mean_energy_kwh': float(Ec.mean()), 'mean_duration_h': float(Dc.mean()),
                        'mean_arrival_hour': float(np.nanmean(hour.reindex(j.index.astype(str)).to_numpy(float)))})
        k0 = j['k'].to_numpy() == 0
        mae.append({'seed': seed, 'population': 'cohort first identified session', 'forecast': 'D', 'n': int(k0.sum()),
                    'mae_energy': float(np.abs(Ec - j['pred_D_e'])[k0].mean()), 'mae_duration': float(np.abs(Dc - j['pred_D'])[k0].mean())})
        # ---- other sessions, calibrated within their own population
        fill_curves = {}
        for pop in ('anonymous', 'rare_identified'):
            z = f[f['population'] == pop]
            if len(z) < 50:
                continue
            D, E = z['duration_true'].to_numpy(float), z['energy_true'].to_numpy(float)
            g = stable_group(z['week'] if pop == 'anonymous' else z['user'])
            cid = cid_of(('d:' + z['day']) if pop == 'anonymous' else ('u:' + z['user']))
            for var in ('pred', 'pred_cold'):
                cv = curves(D, E, z[var].to_numpy(float), z[f'{var}_e'].to_numpy(float), g, ctx.power)
                fill_curves[(pop, var)] = (cv, cid)
                sets.setdefault(f'{pop}_{var}', CurveSet(nc)).add(*cv, cid, key=seed)
                mae.append({'seed': seed, 'population': pop, 'forecast': var, 'n': len(z),
                            'mae_energy': float(np.abs(E - z[f'{var}_e']).mean()), 'mae_duration': float(np.abs(D - z[var]).mean()),
                            'bias_energy': float((z[f'{var}_e'] - E).mean()), 'bias_duration': float((z[var] - D).mean()),
                            'mean_energy_kwh': float(E.mean()), 'mean_duration_h': float(D.mean()),
                            'mean_arrival_hour': float(np.nanmean(z['arrival_hour']))})
            # transfer: anonymous calibrated with the identified drivers' cold-start residuals (setting D)
            if pop == 'anonymous':
                rd = Dc - j['pred_D'].to_numpy(float); re_ = Ec - j['pred_D_e'].to_numpy(float)
                Eac = np.minimum(E, ctx.power * D)
                for gam in (0.1, 0.05, 0.025, 0.01, 0.005):
                    for calib, (qd, qe) in {'identified drivers (transfer)': (np.quantile(rd, gam), np.quantile(re_, 1 - gam))}.items():
                        Dh = np.maximum(z['pred'].to_numpy(float) + qd, 0); Eh = np.maximum(z['pred_e'].to_numpy(float) + qe, 0)
                        Ea, U, F = alap(D, E, Dh, Eh, ctx.power)
                        transfer.append({'seed': seed, 'gamma': gam, 'calibration': calib, 'n': len(z),
                                         'P(D_hat>D)_pct': 100 * np.mean(Dh > D), 'P(E_hat<E_ach)_pct': 100 * np.mean(Eh < Eac),
                                         'P(U>0)_pct': 100 * np.mean(U > 1e-9), 'bound_2gamma_pct': 200 * gam,
                                         'undelivered_share_pct': 100 * U.sum() / Ea.sum(),
                                         'flex_pct_of_oracle': 100 * F.sum() / np.minimum(np.clip(D - Eac / ctx.power, 0, None), D).sum()})
        # ---- network mix: cohort forecast + other sessions, each population calibrated on its own
        mix = {'benchmark (cohort random split)': ('A', 'pred'), 'deployment (unseen driver, own history)': ('C', 'pred'),
               'cold (history hidden everywhere)': ('D', 'pred_cold')}
        for name, (s, var) in mix.items():
            cs = sets.setdefault(f'network: {name}', CurveSet(nc))
            cs.add(*cohort_curves[s], cc, key=seed)
            for pop in ('anonymous', 'rare_identified'):
                if (pop, var) in fill_curves:
                    cv, cid = fill_curves[(pop, var)]
                    cs.add(*cv, cid, key=seed)
        # ---- forecasts for the ACN-Sim study (every window session, one seed)
        if learner == ctx.cfg['acnsim']['learner'] and seed == int(ctx.cfg['acnsim']['seed']):
            c = pd.DataFrame({'session_key': j.index.astype(str), 'population': 'cohort', 'calib_group': gc,
                              'D': Dc, 'E': Ec, 'requested_duration': j['requested_duration'].to_numpy(float),
                              'requested_energy': j['requested_energy'].to_numpy(float)})
            for name, (s, var) in mix.items():
                tag = name.split(' ')[0]
                c[f'Dp_{tag}'] = j[f'pred_{s}'].to_numpy(float); c[f'Ep_{tag}'] = j[f'pred_{s}_e'].to_numpy(float)
            o = pd.DataFrame({'session_key': f.index.astype(str), 'population': f['population'].to_numpy(),
                              'calib_group': np.where(f['population'] == 'anonymous', stable_group(f['week']), stable_group(f['user'])),
                              'D': f['duration_true'].to_numpy(float), 'E': f['energy_true'].to_numpy(float),
                              'requested_duration': f['requested_duration'].to_numpy(float),
                              'requested_energy': f['requested_energy'].to_numpy(float)})
            for name, (s, var) in mix.items():
                tag = name.split(' ')[0]
                o[f'Dp_{tag}'] = f[var].to_numpy(float); o[f'Ep_{tag}'] = f[f'{var}_e'].to_numpy(float)
            pd.concat([c, o], ignore_index=True).to_parquet(ctx.p(f'deploy_forecasts_{learner}_seed{seed}.parquet'), index=False)
    m = pd.DataFrame(mae).groupby(['population', 'forecast']).mean(numeric_only=True).drop(columns='seed')
    m.to_csv(ctx.p(f'anonymous_mae_{learner}.csv'))
    if transfer:
        pd.DataFrame(transfer).groupby(['calibration', 'gamma']).mean(numeric_only=True).drop(columns='seed') \
            .to_csv(ctx.p(f'anonymous_transfer_{learner}.csv'))
    nb = 'network: benchmark (cohort random split)'; nd_ = 'network: deployment (unseen driver, own history)'
    nco = 'network: cold (history hidden everywhere)'
    contrasts = {'network loss_pct: 1 - deployment/benchmark': lambda v: 100 * (1 - v[nd_] / v[nb]),
                 'network loss_pct: 1 - cold/deployment': lambda v: 100 * (1 - v[nco] / v[nd_]),
                 'cohort loss_pct: 1 - D/C': lambda v: 100 * (1 - v['cohort_D'] / v['cohort_C'])}
    if 'anonymous_pred' in sets:
        contrasts['anonymous minus cohort D (pp)'] = lambda v: v['anonymous_pred'] - v['cohort_D']
    t = bootstrap_table(sets, nc, ctx.n_boot, contrasts=contrasts, strata=strata)
    shares = {}
    for seed in js:
        f = fills[seed]; n_c = len(js[seed])
        tot = n_c + len(f)
        shares = {'share_cohort': n_c / tot, 'share_anonymous': float((f['population'] == 'anonymous').sum() / tot),
                  'share_rare_identified': float((f['population'] == 'rare_identified').sum() / tot), 'sessions_window': tot}
        break
    for k, v in shares.items():
        t[k] = v
    t.insert(0, 'learner', learner)
    t.to_csv(ctx.p(f'anonymous_flex_{learner}.csv'), index=False)
    ctx.log(f'2b {learner}: ' + ', '.join(f"{q}={e:.2f}" for q, e in zip(t['quantity'], t['estimate'])))
