"""Experiment E3 by week: energy cost, demand charge, peak and undelivered energy of every optimisation run.

acnsim.summarise reports E3 over the whole simulated period under the configured tariff. This stage splits each E3
run (adaptive scheduling algorithm, acnsim.py) into its simulated weeks, so that runs can be compared on any subset of
weeks (for example the eight weeks of an earlier release), paired differences between forecasts can be resampled by
week, and the same comparison can be made under two tariffs. The tariff is the one the run directory's configuration
names (acnsim.tariff), which is also the tariff the optimiser minimised.
Outputs
  acnsim_e3_weeks.csv    one row per run and week: energy, time-of-use energy cost, peak, undelivered and
                         achievable energy, sessions
  acnsim_e3_months.csv   one row per run and calendar month: peak and the tariff's demand charge rate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .acnsim import PMAX, _prices, load_inputs
from .common import Ctx


def run(ctx: Ctx):
    sc = ctx.cfg['acnsim']
    rdir = ctx.p('acnsim', 'runs', 'x').parent
    info = json.loads(ctx.p('acnsim', 'site_info.json').read_text())
    z = load_inputs(ctx).set_index('session_key')
    weeks, months = [], []
    for f in sorted(rdir.glob('*__E3__*.parquet')):
        rid = f.name[:-len('.parquet')]
        if rid.endswith('.tmp'):
            continue
        site, _, sch, cap, fc, tau = rid.split('__')
        period = int(sc.get('asa_period_minutes', 15))
        origin = pd.Timestamp(info[site]['origin'])
        r = pd.read_parquet(f).set_index('session_key')
        g = z.loc[r.index]
        E = g['E'].to_numpy(float)
        E_ach = np.minimum(E, PMAX * (g['t1'] - g['t0']).dt.total_seconds().to_numpy() / 3600)
        unmet = np.clip(E_ach - r['delivered_kwh'].to_numpy(float), 0, None)
        wk = ((g['t0'] - origin).dt.days // 7).to_numpy()
        sess = pd.DataFrame({'week': wk, 'unmet': unmet, 'E_ach': E_ach}).groupby('week').agg(
            unmet_kwh=('unmet', 'sum'), achievable_kwh=('E_ach', 'sum'), sessions=('unmet', 'size'))
        pw = np.load(rdir / f'{rid}.power.npz')
        n = max(int(o) + len(pw[f'p{int(o)}']) for o in pw['offsets'])
        load = np.zeros(n)
        for o in pw['offsets']:
            p = pw[f'p{int(o)}']; load[int(o): int(o) + len(p)] += p
        ts, price, dem = _prices(origin.to_pydatetime(), n, period, sc['tariff'])
        e = load * period / 60
        steps_week = 7 * 24 * 60 // period
        w_of = np.arange(n) // steps_week
        d = pd.DataFrame({'week': w_of, 'energy_kwh': e, 'cost': e * price, 'load': load})
        pw_week = d.groupby('week').agg(energy_kwh=('energy_kwh', 'sum'), tou_energy_cost_usd=('cost', 'sum'), peak_kw=('load', 'max'))
        simulated = sorted(int(o) // steps_week for o in pw['offsets'])
        tab = pw_week.join(sess, how='outer').reindex(simulated)
        for w, row in tab.iterrows():
            weeks.append({'run_id': rid, 'site': site, 'scheduler': sch, 'forecast': fc, 'tau': tau.replace('tau', ''),
                          'tariff': sc['tariff'], 'week': int(w), 'week_start': str((origin + pd.Timedelta(weeks=int(w))).date()),
                          **{k: (0.0 if pd.isna(v) else float(v)) for k, v in row.items()}})
        m = pd.DataFrame({'month': ts.to_period('M').astype(str), 'load': load, 'dem': dem}).groupby('month').agg(
            peak_kw=('load', 'max'), demand_rate_usd_per_kw=('dem', 'max'))
        for mo, row in m[m['peak_kw'] > 0].iterrows():
            months.append({'run_id': rid, 'site': site, 'scheduler': sch, 'forecast': fc, 'tau': tau.replace('tau', ''),
                           'tariff': sc['tariff'], 'month': mo, 'peak_kw': float(row['peak_kw']),
                           'demand_rate_usd_per_kw': float(row['demand_rate_usd_per_kw'])})
    if not weeks:
        raise RuntimeError('no E3 runs found')
    pd.DataFrame(weeks).to_csv(ctx.p('acnsim_e3_weeks.csv'), index=False)
    pd.DataFrame(months).to_csv(ctx.p('acnsim_e3_months.csv'), index=False)
    ctx.log(f'E3 by week written: {len({w["run_id"] for w in weeks})} runs, tariff {sc["tariff"]}')
