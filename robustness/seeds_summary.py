"""Summarise the site-level analyses over forecast seeds.

    python -m robustness.seeds_summary --root RUNS_ROOT --seeds 11 23 37 51 79 --out OUT_DIR

RUNS_ROOT/seed<s>/full/acn holds, for each seed, the outputs of the stages acnsim, infostate, infostate_sim and
feeder run on the deploy forecasts of that seed. Writes seeds_summary.csv (one row per quantity and site, with the
value for every seed and the mean, minimum and maximum over seeds) and seeds_summary.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RATED = {'caltech': 150.0, 'jpl': 195.0}


def firm(s, target=95.0):
    """Event limit (fraction of the rating) at which the share of event days within 2 % undelivered energy reaches
    95 %, by linear interpolation between the simulated limits, as in the manuscript's Table III. The flag is '<=' or
    '>' when the limit lies outside the simulated range."""
    d = sorted((float(k), float(v)) for k, v in (kv.split(':') for kv in s.split(';')))
    if d[0][1] >= target:
        return '<=', d[0][0]
    for (x0, y0), (x1, y1) in zip(d, d[1:]):
        if y1 >= target:
            return '', x0 + (x1 - x0) * (target - y0) / (y1 - y0)
    return '>', d[-1][0]


def _tau(v):
    return 'na' if pd.isna(v) or str(v) == 'na' else f'{float(v):g}'


def one_seed(d: Path) -> list[dict]:
    rows = []

    def put(q, site, v):
        rows.append({'quantity': q, 'site': site, 'value': float(v) if v is not None and np.isfinite(v) else np.nan})

    need = pd.read_csv(d / 'acnsim_capacity_needed.csv')
    need['tau_s'] = need['tau'].map(_tau)
    for site, g in need.groupby('site'):
        for sch, fc, tau, lab in [('llf', 'oracle', 'na', 'perfect information'), ('llf', 'benchmark', '0.5', 'driver seen'),
                                  ('llf', 'deployment', '0.5', 'unseen driver, own history'), ('llf', 'cold', '0.5', 'cold start'),
                                  ('fcfs', 'none', 'na', 'no forecast (FCFS)')]:
            r = g[(g.scheduler == sch) & (g.forecast == fc) & (g.tau_s == tau)]
            put(f'C2% (kW), {lab}', site, r.capacity_needed_kw.iloc[0] if len(r) else np.nan)
    con = pd.read_csv(d / 'acnsim_capacity_contrasts.csv')
    con['tau_s'] = con['tau'].map(_tau)
    for site, g in con.groupby('site'):
        r = g[(g.tau_s == '0.5') & (g.contrast == 'cold - deployment')]
        put('C2% cold start minus own history (kW)', site, r.estimate_kw.iloc[0] if len(r) else np.nan)
        for tau in ('0.5', '0.1'):
            for fc, lab in [('deployment', 'own history'), ('cold', 'cold start')]:
                r = g[(g.tau_s == tau) & (g.contrast == f'share of the perfect-information capacity value captured by {fc} (%)')]
                put(f'capacity value captured (%), {lab}, gamma {tau}', site, r.estimate_fraction.iloc[0] if len(r) else np.nan)
    ev = pd.read_csv(d / 'acnsim_event_dependable_flexibility.csv')
    ev['tau_s'] = ev['tau'].map(_tau)
    for site, g in ev[ev.experiment == 'E5'].groupby('site'):
        fk = {}
        for sch, fc, tau, lab in [('llf', 'oracle', 'na', 'perfect information'), ('llf', 'deployment', '0.5', 'unseen driver, own history'),
                                  ('llf', 'cold', '0.5', 'cold start'), ('fcfs', 'none', 'na', 'no forecast (FCFS)')]:
            r = g[(g.scheduler == sch) & (g.forecast == fc) & (g.tau_s == tau)]
            if not len(r):
                continue
            flag, frac = firm(r.share_of_event_days_within_2pct_by_limit.iloc[0])
            fk[lab] = frac * RATED[site]
            put(f'E5 firm event limit (kW, interpolated), {lab}', site, fk[lab])
            put(f'E5 firm event limit at a grid edge (1 = yes), {lab}', site, float(flag != ''))
        if {'cold start', 'unseen driver, own history'} <= set(fk):
            put('E5 firm limit cold start minus own history (kW)', site, fk['cold start'] - fk['unseen driver, own history'])
    runs = pd.read_csv(d / 'acnsim_runs.csv')
    runs['tau_s'] = runs['tau'].map(_tau)
    for site, g in runs[runs.experiment == 'E1'].groupby('site'):
        for fc, lab in [('deployment', 'own history'), ('cold', 'cold start')]:
            r = g[(g.scheduler == 'llf') & (g.forecast == fc) & (g.tau_s == '0.5')]
            put(f'E1 undelivered energy (%), {lab}', site, r.undelivered_share_pct.iloc[0] if len(r) else np.nan)
    f = d / 'infostate_sim.csv'
    if f.exists():
        s = pd.read_csv(f)
        for site, g in s.groupby('site'):
            for cal, lab in [('state, start', 'information-state rule'), ('population, D and E', 'Proposition 1 calibration')]:
                r = g[g.calibration == cal]
                cols = [c for c in r.columns if c.endswith('undelivered_pct') and c != 'undelivered_pct']
                put(f'ACN-Sim E1 largest class undelivered (%), {lab}', site, float(r[cols].max().max()) if len(r) else np.nan)
    f = d / 'infostate_flex.csv'
    if f.exists():
        s = pd.read_csv(f)
        for cal, lab in [('state, start', 'information-state rule'), ('population, D and E', 'Proposition 1 calibration')]:
            r = s[(s.forecast == 'deployment') & (s.calibration == cal) & (s.quantity == 'network flex_pct')]
            put(f'network flexibility (% of oracle), {lab}', 'acn', r.estimate.iloc[0] if len(r) else np.nan)
    f = d / 'feeder_hosting_capacity.csv'
    if f.exists():
        h = pd.read_csv(f)
        for (fd, site), g in h.groupby(['feeder', 'site']):
            z = g.set_index('forecast')['hubs_mean']
            if {'unseen driver, own history', 'cold start'} <= set(z.index):
                dep, cold = z['unseen driver, own history'], z['cold start']
                put(f'hubs lost to the cold start (%), {fd}', site, 100 * (dep - cold) / dep)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for s in a.seeds:
        d = Path(a.root) / f'seed{s}' / 'full' / 'acn'
        if not (d / 'acnsim_capacity_needed.csv').exists():
            print(f'seed {s}: no ACN-Sim results in {d}, skipped'); continue
        frames.append(pd.DataFrame(one_seed(d)).assign(seed=s))
    if not frames:
        raise SystemExit('no seed results found')
    long = pd.concat(frames)
    wide = long.pivot_table(index=['quantity', 'site'], columns='seed', values='value', aggfunc='first')
    wide.columns = [f'seed {c}' for c in wide.columns]
    vals = wide.to_numpy(float)
    wide['mean'] = np.nanmean(vals, 1); wide['min'] = np.nanmin(vals, 1); wide['max'] = np.nanmax(vals, 1)
    wide['seeds'] = np.isfinite(vals).sum(1)
    wide = wide.reset_index()
    wide.to_csv(out / 'seeds_summary.csv', index=False)
    with open(out / 'seeds_summary.md', 'w') as fh:
        fh.write('| quantity | site | mean | min | max | seeds |\n|---|---|---|---|---|---|\n')
        for _, r in wide.iterrows():
            fh.write(f"| {r['quantity']} | {r['site']} | {r['mean']:.2f} | {r['min']:.2f} | {r['max']:.2f} | {int(r['seeds'])} |\n")
    print(f'written {out / "seeds_summary.csv"} ({len(wide)} rows, {len(frames)} seeds)')


if __name__ == '__main__':
    main()
