"""Summarise the sensitivity analyses of version 1.0.10 over the five forecast seeds.

    python scripts/summarise_sensitivity.py [--results results]

Reads the per-seed outputs under results/robustness/acn/{seeds,tariffs,hosting_timeseries} and writes:

  tariffs/e3_tariffs_summary.csv            E3 under SCE TOU-EV-4 and TOU-EV-8, every week: undelivered energy and energy
                                            cost per delivered kWh against uncontrolled charging, mean and range over seeds
  tariffs/e3_tariffs_paired.csv             cold start against own history, energy cost per delivered kWh, per seed
  hosting_timeseries/hosting_timeseries_summary.csv     hubs hosted over the year by forecast and hub profiles,
                                            with the deterministic planning value of the same seeds and orders
  hosting_timeseries/hosting_timeseries_contrasts.csv   own history against cold start, per seed and over seeds
  hosting_timeseries/consistency_check.csv  hubs drawing the deployment limit on weekdays 07:00-19:00, time series
                                            against the deterministic screen, per seed, site and placement order

Energy cost is compared per delivered kWh because the policies deliver slightly different amounts of energy.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [11, 23, 37, 51, 79]
TARIFFS = ['sce_tou_ev_4_march_2019', 'sce_tou_ev_8_june_2019']
SB = '1-MV-comm--0-sw'
POLICY = {('uncontrolled', 'oracle', 'na'): 'uncontrolled', ('asa', 'oracle', 'na'): 'perfect information',
          ('asa', 'request', 'na'): 'driver request, else own history',
          ('asa', 'benchmark', '0.5'): 'driver seen, gamma 0.5', ('asa', 'benchmark', '0.1'): 'driver seen, gamma 0.1',
          ('asa', 'deployment', '0.5'): 'unseen driver, own history, gamma 0.5',
          ('asa', 'deployment', '0.1'): 'unseen driver, own history, gamma 0.1',
          ('asa', 'cold', '0.5'): 'cold start, gamma 0.5', ('asa', 'cold', '0.1'): 'cold start, gamma 0.1'}


def tau_str(v):
    return 'na' if pd.isna(v) else f'{float(v):g}'


def tariffs(acn: Path):
    rows = []
    for t in TARIFFS:
        for s in SEEDS:
            r = pd.read_csv(acn / 'tariffs' / t / f'seed{s}' / 'acnsim_runs.csv')
            r = r[r['experiment'] == 'E3'].copy()
            r['tariff'], r['seed'] = t, s
            rows.append(r)
    r = pd.concat(rows, ignore_index=True)
    r['policy'] = [POLICY[(a, b, tau_str(c))] for a, b, c in zip(r['scheduler'], r['forecast'], r['tau'])]
    key = ['tariff', 'seed', 'site']
    unc = r[r['policy'] == 'uncontrolled'].set_index(key)
    orc = r[r['policy'] == 'perfect information'].set_index(key)
    idx = list(zip(r['tariff'], r['seed'], r['site']))
    r['cost_per_kwh_vs_uncontrolled_pct'] = 100 * (r['tou_cost_per_kwh'].to_numpy() / unc.loc[idx, 'tou_cost_per_kwh'].to_numpy() - 1)
    r['cost_vs_uncontrolled_pct'] = 100 * (r['tou_energy_cost_usd'].to_numpy() / unc.loc[idx, 'tou_energy_cost_usd'].to_numpy() - 1)
    save_orc = 1 - orc.loc[idx, 'tou_cost_per_kwh'].to_numpy() / unc.loc[idx, 'tou_cost_per_kwh'].to_numpy()
    save = 1 - r['tou_cost_per_kwh'].to_numpy() / unc.loc[idx, 'tou_cost_per_kwh'].to_numpy()
    r['share_of_perfect_information_saving_pct'] = np.where(save_orc > 0, 100 * save / save_orc, np.nan)
    agg = {}
    for c in ['undelivered_share_pct', 'cost_per_kwh_vs_uncontrolled_pct', 'cost_vs_uncontrolled_pct',
              'share_of_perfect_information_saving_pct', 'tou_cost_per_kwh', 'demand_charge_usd', 'peak_kw']:
        agg[c + '_mean'] = (c, 'mean'); agg[c + '_min'] = (c, 'min'); agg[c + '_max'] = (c, 'max')
    agg['seeds'] = ('seed', 'nunique'); agg['chunk_errors'] = ('chunk_errors', 'sum'); agg['sessions'] = ('sessions', 'first')
    out = r.groupby(['tariff', 'site', 'policy'], sort=False).agg(**agg).reset_index()
    out.to_csv(acn / 'tariffs' / 'e3_tariffs_summary.csv', index=False)
    pr = []
    for (t, site, s), g in r.groupby(['tariff', 'site', 'seed']):
        g = g.set_index('policy')
        for gam in ['0.5', '0.1']:
            c, d = g.loc[f'cold start, gamma {gam}', 'tou_cost_per_kwh'], g.loc[f'unseen driver, own history, gamma {gam}', 'tou_cost_per_kwh']
            pr.append({'tariff': t, 'site': site, 'seed': s, 'gamma': gam, 'cold_over_own_history_cost_per_kwh_pct': 100 * (c / d - 1)})
    pd.DataFrame(pr).to_csv(acn / 'tariffs' / 'e3_tariffs_paired.csv', index=False)
    return out


def hosting(acn: Path):
    by, det, chk, con = [], [], [], []
    for s in SEEDS:
        d = acn / 'hosting_timeseries' / f'seed{s}'
        a = pd.read_csv(d / 'feeder_ts_by_order.csv'); a['seed'] = s; by.append(a)
        h = pd.read_csv(d / 'feeder_ts_hosting.csv'); h['seed'] = s; con.append(h)
        b = pd.read_csv(acn / 'seeds' / f'seed{s}' / 'feeder_hosting_by_order.csv'); b = b[b['feeder'] == SB].copy(); b['seed'] = s; det.append(b)
    by, det, hs = pd.concat(by), pd.concat(det), pd.concat(con)
    det = det[det['order'].isin(by['order'].unique())]
    # consistency: limit drawn through weekday 07:00-19:00 against the deterministic screen of the same limit
    c = by[by['forecast'].str.contains('check')].merge(
        det[det['forecast'] == 'unseen driver, own history'][['seed', 'site', 'order', 'hubs']].rename(columns={'hubs': 'hubs_deterministic'}),
        on=['seed', 'site', 'order'])
    c = c.rename(columns={'hubs': 'hubs_timeseries'})
    c['difference_hubs'] = c['hubs_timeseries'] - c['hubs_deterministic']
    c['difference_pct'] = 100 * c['difference_hubs'] / c['hubs_deterministic']
    c[['seed', 'site', 'order', 'hubs_timeseries', 'hubs_deterministic', 'difference_hubs', 'difference_pct', 'binding']].to_csv(
        acn / 'hosting_timeseries' / 'consistency_check.csv', index=False)
    # per seed means over orders
    ts = by[~by['forecast'].str.contains('check')].groupby(['hub_profiles', 'site', 'forecast', 'seed'])['hubs'].mean().rename('hubs').reset_index()
    dm = det.groupby(['site', 'forecast', 'seed'])['hubs'].mean().rename('hubs_deterministic').reset_index()
    ts = ts.merge(dm, on=['site', 'forecast', 'seed'], how='left')
    prof = hs[~hs['forecast'].str.contains('check')][['hub_profiles', 'site', 'forecast', 'seed', 'limit_kw', 'hub_peak_kw', 'coincidence_factor',
                                                      'binding_voltage_pct', 'mw_contracted_mean']]
    ts = ts.merge(prof, on=['hub_profiles', 'site', 'forecast', 'seed'])
    out = ts.groupby(['hub_profiles', 'site', 'forecast']).agg(
        hubs_mean=('hubs', 'mean'), hubs_min=('hubs', 'min'), hubs_max=('hubs', 'max'),
        hubs_deterministic_mean=('hubs_deterministic', 'mean'), limit_kw_mean=('limit_kw', 'mean'),
        hub_peak_kw_mean=('hub_peak_kw', 'mean'), coincidence_factor_mean=('coincidence_factor', 'mean'),
        binding_voltage_pct_mean=('binding_voltage_pct', 'mean'), mw_contracted_mean=('mw_contracted_mean', 'mean'),
        seeds=('seed', 'nunique')).reset_index()
    out.to_csv(acn / 'hosting_timeseries' / 'hosting_timeseries_summary.csv', index=False)
    p = ts.pivot_table(index=['hub_profiles', 'site', 'seed'], columns='forecast', values='hubs').reset_index()
    own, cold, fcfs, orc = 'unseen driver, own history', 'cold start', 'no forecast (FCFS)', 'perfect information'
    p['own_history_over_cold_start_pct'] = 100 * (p[own] / p[cold] - 1)
    p['cold_start_below_own_history_pct'] = 100 * (1 - p[cold] / p[own])
    p['own_history_over_fcfs_pct'] = 100 * (p[own] / p[fcfs] - 1)
    p['perfect_information_over_fcfs_pct'] = 100 * (p[orc] / p[fcfs] - 1)
    p.columns.name = None
    p.to_csv(acn / 'hosting_timeseries' / 'hosting_timeseries_contrasts.csv', index=False)
    return out, p, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    acn = Path(ap.parse_args().results) / 'robustness' / 'acn'
    e3 = tariffs(acn)
    print(e3[['tariff', 'site', 'policy', 'undelivered_share_pct_mean', 'cost_per_kwh_vs_uncontrolled_pct_mean',
              'cost_per_kwh_vs_uncontrolled_pct_min', 'cost_per_kwh_vs_uncontrolled_pct_max', 'share_of_perfect_information_saving_pct_mean']].round(2).to_string(index=False))
    out, p, c = hosting(acn)
    print(out.round(2).to_string(index=False))
    print(p.groupby(['hub_profiles', 'site'])['cold_start_below_own_history_pct'].agg(['mean', 'min', 'max']).round(2))
    print('consistency: largest |time series - deterministic| %.4f hubs (%.4f%%) over %d placements'
          % (c['difference_hubs'].abs().max(), c['difference_pct'].abs().max(), len(c)))


if __name__ == '__main__':
    main()
