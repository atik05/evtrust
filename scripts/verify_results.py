"""Recompute the sensitivity results quoted in the paper from the per-seed result tables and compare them with the
values printed in the paper.

    python scripts/verify_results.py [--results results]

Every value is recomputed from the per-seed CSVs (not from the summary tables of scripts/summarise_sensitivity.py) and
rounded as in the paper. Exit status 1 if any value differs. With the published tables in results/ every check passes;
on a rerun (runs/sensitivity/results) differences of the last digit can arise from training on a GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [11, 23, 37, 51, 79]
RATED = {'caltech': 150.0, 'jpl': 195.0}
SB = '1-MV-comm--0-sw'
# value printed in the paper: (text, format)
PAPER = {
    'cold start minus own history, C2% (kW), Caltech, min over seeds': ('2.3', '{:.1f}'),
    'cold start minus own history, C2% (kW), Caltech, max over seeds': ('2.8', '{:.1f}'),
    'cold start minus own history, C2% (kW), JPL, min over seeds': ('7.1', '{:.1f}'),
    'cold start minus own history, C2% (kW), JPL, max over seeds': ('9.9', '{:.1f}'),
    'cold start, capacity value captured at gamma 0.1 (%), Caltech, max over seeds': ('3.0', '{:.1f}'),
    'cold start, capacity value captured at gamma 0.1 (%), JPL, max over seeds': ('11.9', '{:.1f}'),
    'E5 firm limit, cold start minus own history (kW), Caltech, seed 11': ('10', '{:.0f}'),
    'E5 firm limit, cold start minus own history (kW), JPL, seed 11': ('22', '{:.0f}'),
    'E5 firm limit, cold start minus own history (kW), Caltech, min over seeds': ('6', '{:.0f}'),
    'E5 firm limit, cold start minus own history (kW), Caltech, max over seeds': ('16', '{:.0f}'),
    'E5 firm limit, cold start minus own history (kW), JPL, min over seeds': ('18', '{:.0f}'),
    'E5 firm limit, cold start minus own history (kW), JPL, max over seeds': ('24', '{:.0f}'),
    'tuning: largest |change| of protocol MAE, 8 to 40 trials (%)': ('3.4', '{:.1f}'),
    'E3 TOU-EV-4, gamma 0.1: smallest extra cost per kWh of unseen-driver forecasts over uncontrolled (%)': ('0.8', '{:.1f}'),
    'E3 TOU-EV-4, gamma 0.1: largest extra cost per kWh of unseen-driver forecasts over uncontrolled (%)': ('3.2', '{:.1f}'),
    'E3 TOU-EV-8, gamma 0.5: share of the perfect-information saving, own history, Caltech (%)': ('80', '{:.0f}'),
    'E3 TOU-EV-8, gamma 0.5: share of the perfect-information saving, own history, JPL (%)': ('92', '{:.0f}'),
    'E3 TOU-EV-8, gamma 0.5: share of the perfect-information saving, cold start, Caltech (%)': ('64', '{:.0f}'),
    'E3 TOU-EV-8, gamma 0.5: share of the perfect-information saving, cold start, JPL (%)': ('87', '{:.0f}'),
    'time series, diverse hubs: hubs lost to the cold start, Caltech (%)': ('8.2', '{:.1f}'),
    'time series, diverse hubs: hubs lost to the cold start, JPL (%)': ('4.5', '{:.1f}'),
    'time series, coincident hubs: hubs lost to the cold start, Caltech (%)': ('-5.5', '{:.1f}'),
    'time series, coincident hubs: hubs lost to the cold start, JPL (%)': ('2.1', '{:.1f}'),
    'time series over deterministic hosting, smallest gain (%)': ('10', '{:.0f}'),
    'time series over deterministic hosting, largest gain (%)': ('31', '{:.0f}'),
    'check: largest |time series - deterministic| (hubs)': ('0.14', '{:.2f}'),
}


def firm(s, target=95.0):
    d = sorted((float(k), float(v)) for k, v in (kv.split(':') for kv in s.split(';')))
    if not d[0][1] < target <= d[-1][1]:
        raise ValueError('firm limit outside the simulated grid')
    for (x0, y0), (x1, y1) in zip(d, d[1:]):
        if y1 >= target:
            return x0 + (x1 - x0) * (target - y0) / (y1 - y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    R = Path(ap.parse_args().results)
    A = R / 'robustness' / 'acn'
    got = {}
    # five forecast seeds
    for site, S in [('caltech', 'Caltech'), ('jpl', 'JPL')]:
        dl, kc, gap = [], [], {}
        for s in SEEDS:
            c = pd.read_csv(A / 'seeds' / f'seed{s}' / 'acnsim_capacity_contrasts.csv')
            dl.append(float(c[(c.site == site) & (c.tau == 0.5) & (c.contrast == 'cold - deployment')].estimate_kw.iloc[0]))
            kc.append(float(c[(c.site == site) & (c.tau == 0.1) & (c.contrast == 'share of the perfect-information capacity value captured by cold (%)')].estimate_fraction.iloc[0]))
            e = pd.read_csv(A / 'seeds' / f'seed{s}' / 'acnsim_event_dependable_flexibility.csv')
            e = e[(e.experiment == 'E5') & (e.site == site)]; e['tau'] = e['tau'].astype(str)
            f = {fc: e[(e.forecast == fc) & (e.tau == '0.5')].iloc[0] for fc in ('cold', 'deployment')}
            gap[s] = RATED[site] * (firm(f['cold'].share_of_event_days_within_2pct_by_limit) - firm(f['deployment'].share_of_event_days_within_2pct_by_limit))
        got[f'cold start minus own history, C2% (kW), {S}, min over seeds'] = min(dl)
        got[f'cold start minus own history, C2% (kW), {S}, max over seeds'] = max(dl)
        got[f'cold start, capacity value captured at gamma 0.1 (%), {S}, max over seeds'] = max(kc)
        got[f'E5 firm limit, cold start minus own history (kW), {S}, seed 11'] = gap[11]
        got[f'E5 firm limit, cold start minus own history (kW), {S}, min over seeds'] = min(gap.values())
        got[f'E5 firm limit, cold start minus own history (kW), {S}, max over seeds'] = max(gap.values())
    pub = A / 'acnsim_capacity_needed.csv'
    if pub.exists():
        m = pd.read_csv(pub).merge(pd.read_csv(A / 'seeds' / 'seed11' / 'acnsim_capacity_needed.csv'), on=['site', 'scheduler', 'forecast', 'tau'])
        print(f'seed 11 against the published capacities: {len(m)} rows, largest difference {float((m.capacity_needed_kw_x - m.capacity_needed_kw_y).abs().max()):.4f} kW')
    # tuning budget
    r = pd.read_csv(R / 'tuning' / 'tuning_budget_runs.csv')
    p = r.groupby(['learner', 'target', 'protocol', 'budget', 'seed']).test_mae.mean().groupby(['learner', 'target', 'protocol', 'budget']).mean().unstack('budget')
    got['tuning: largest |change| of protocol MAE, 8 to 40 trials (%)'] = (100 * (p['40 trials'] / p['8 trials'] - 1)).abs().max()
    better = {b: (lambda q: (q['catboost'] < q['torch_ft']))(p[b].unstack('learner')) for b in ('8 trials', '40 trials')}
    flips = sorted({i[1] for i in better['8 trials'].index[better['8 trials'] != better['40 trials']]})
    print('protocols in which the learner with the lower MAE changes with the budget:', flips)
    # E3 under two tariffs, cost per delivered kWh recomputed from cost and energy
    x = []
    for t in ('sce_tou_ev_4_march_2019', 'sce_tou_ev_8_june_2019'):
        for s in SEEDS:
            z = pd.read_csv(A / 'tariffs' / t / f'seed{s}' / 'acnsim_runs.csv'); z = z[z.experiment == 'E3'].copy()
            z['tariff'], z['seed'], z['cpk'] = t, s, z.tou_energy_cost_usd / z.energy_kwh; x.append(z)
    x = pd.concat(x); x['tau'] = x['tau'].map(lambda v: 'na' if pd.isna(v) else f'{float(v):g}')

    def rel(t, site, fc, tau):
        v = []
        for s in SEEDS:
            z = x[(x.tariff == t) & (x.site == site) & (x.seed == s)]
            u = z[z.scheduler == 'uncontrolled'].cpk.iloc[0]; o = z[(z.scheduler == 'asa') & (z.forecast == 'oracle')].cpk.iloc[0]
            f = z[(z.scheduler == 'asa') & (z.forecast == fc) & (z.tau == tau)].cpk.iloc[0]
            v.append((100 * (f / u - 1), 100 * (1 - f / u) / (1 - o / u)))
        return np.array(v).mean(axis=0)
    four = [rel('sce_tou_ev_4_march_2019', s, f, '0.1')[0] for s in ('caltech', 'jpl') for f in ('deployment', 'cold')]
    got['E3 TOU-EV-4, gamma 0.1: smallest extra cost per kWh of unseen-driver forecasts over uncontrolled (%)'] = min(four)
    got['E3 TOU-EV-4, gamma 0.1: largest extra cost per kWh of unseen-driver forecasts over uncontrolled (%)'] = max(four)
    for site, S in [('caltech', 'Caltech'), ('jpl', 'JPL')]:
        for fc, F in [('deployment', 'own history'), ('cold', 'cold start')]:
            got[f'E3 TOU-EV-8, gamma 0.5: share of the perfect-information saving, {F}, {S} (%)'] = rel('sce_tou_ev_8_june_2019', site, fc, '0.5')[1]
    # time-series hosting from the per-order files, paired with the deterministic orders of the same seeds
    by, det = [], []
    for s in SEEDS:
        a = pd.read_csv(A / 'hosting_timeseries' / f'seed{s}' / 'feeder_ts_by_order.csv'); a['seed'] = s; by.append(a)
        d = pd.read_csv(A / 'seeds' / f'seed{s}' / 'feeder_hosting_by_order.csv'); d = d[d.feeder == SB].copy(); d['seed'] = s; det.append(d)
    by, det = pd.concat(by), pd.concat(det); det = det[det.order.isin(by.order.unique())]
    mt = by.groupby(['hub_profiles', 'site', 'forecast', 'seed']).hubs.mean(); md = det.groupby(['site', 'forecast', 'seed']).hubs.mean()
    gains = []
    for site, S in [('caltech', 'Caltech'), ('jpl', 'JPL')]:
        for prof in ('coincident', 'diverse'):
            got[f'time series, {prof} hubs: hubs lost to the cold start, {S} (%)'] = np.mean(
                [100 * (1 - mt[(prof, site, 'cold start', s)] / mt[(prof, site, 'unseen driver, own history', s)]) for s in SEEDS])
        for fc in ('cold start', 'unseen driver, own history', 'no forecast (FCFS)', 'perfect information', 'driver seen'):
            gains.append(100 * (np.mean([mt[('coincident', site, fc, s)] for s in SEEDS]) / np.mean([md[(site, fc, s)] for s in SEEDS]) - 1))
    got['time series over deterministic hosting, smallest gain (%)'] = min(gains)
    got['time series over deterministic hosting, largest gain (%)'] = max(gains)
    c = by[by.forecast.str.contains('check')].merge(det[det.forecast == 'unseen driver, own history'][['seed', 'site', 'order', 'hubs']],
                                                    on=['seed', 'site', 'order'], suffixes=('_ts', '_det'))
    got['check: largest |time series - deterministic| (hubs)'] = (c.hubs_ts - c.hubs_det).abs().max()

    bad = 0
    for k, (txt, fmt) in PAPER.items():
        v = fmt.format(got[k]); ok = v == txt; bad += not ok
        print(f'{"ok  " if ok else "DIFF"} {k}: paper {txt}, recomputed {v}')
    print('all values agree with the paper' if not bad else f'{bad} values differ from the paper')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
