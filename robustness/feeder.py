"""Feeder hosting capacity for workplace charging hubs under the connection limits each forecast needs.

A hub is a replica of the Caltech or JPL network. Its contracted connection limit is the transformer capacity that
the site needs for at most 2 % undelivered energy with least-laxity-first scheduling under a given forecast
(acnsim_capacity_needed.csv; first come first served without a forecast; raw forecasts). A distribution system operator
that connects hubs at their contracted limits admits hubs until a voltage or thermal limit is reached
(deterministic hosting capacity, hubs drawing their limit at unity power factor, the planning view of a firm connection):

  IEEE 33-bus feeder (Baran and Wu), 12.66 kV, substation voltage regulated to 1.05 pu, 0.95 pu lower limit, peak
      load of the test case; the case carries no line ratings, so the limit is voltage
  SimBench 1-MV-comm--0-sw, 20 kV commercial feeder with its own 2016 load and photovoltaic profiles, voltage band
      and line and transformer ratings of the benchmark; hubs draw their limit on weekdays from 07:00 to 19:00 and the
      critical periods are the steps of that window with the highest net load

Hubs are added one at a time to load buses in random order (several hubs may share a bus); the hosting capacity of an
order is the largest number of hubs with no violation, found by bisection, plus the largest fraction of the next hub
that still fits, and the same orders are used for every forecast, so differences between forecasts are paired. The
interval of the cold-start contrast evaluates the cold start at the deployment limit plus the bounds of the paired
day-block bootstrap difference in capacity needed (acnsim_capacity_contrasts.csv).
Outputs
  feeder_hosting_capacity.csv   hubs and MW hosted per feeder, hub type and forecast (mean, median and 5-95 % over
                                orders), and for the cold start at the bounds of the paired capacity contrast
  feeder_contrasts.csv          paired differences in hubs hosted and the share of the forecast's hosting value
                                (deployment against no forecast) that a cold start forfeits
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .common import Ctx

FORECASTS = [('fcfs', 'none', 'na', 'no forecast (FCFS)'), ('llf', 'oracle', 'na', 'perfect information'),
             ('llf', 'benchmark', '0.5', 'driver seen'), ('llf', 'deployment', '0.5', 'unseen driver, own history'),
             ('llf', 'request', 'na', 'driver request'), ('llf', 'cold', '0.5', 'cold start')]


def contracted_limits(ctx: Ctx) -> pd.DataFrame:
    n = pd.read_csv(ctx.p('acnsim_capacity_needed.csv'))
    n['tau_s'] = n['tau'].map(lambda v: 'na' if pd.isna(v) or str(v) == 'na' else f'{float(v):g}')
    rows = []
    for site, g in n.groupby('site'):
        rated = float((g['capacity_needed_kw'] / g['capacity_needed_fraction']).dropna().iloc[0])
        for sch, fc, tau, label in FORECASTS:
            r = g[(g['scheduler'] == sch) & (g['forecast'] == fc) & (g['tau_s'] == tau)]
            if len(r):
                r = r.iloc[0]
                rows.append({'site': site, 'forecast': label, 'limit_kw': float(r['capacity_needed_kw']),
                             'limit_kw_low': float(r['ci_low']) * rated, 'limit_kw_high': float(r['ci_high']) * rated})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- feeders
class Feeder:
    def __init__(self, name: str, cfg: dict):
        import pandapower as pp
        warnings.filterwarnings('ignore')
        self.pp, self.name = pp, name
        if name == 'ieee33':
            import pandapower.networks as pn
            net = pn.case33bw()
            net.ext_grid['vm_pu'] = float(cfg.get('ieee33_slack_pu', 1.05))
            self.vmin, self.vmax = float(cfg.get('ieee33_vmin_pu', 0.95)), 1.10
            self.thermal = False
            self.steps = [None]
            self.base_p, self.base_q = net.load['p_mw'].to_numpy(), net.load['q_mvar'].to_numpy()
        else:
            import simbench as sb
            net = sb.get_simbench_net(name)
            prof = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
            lp, lq, sg = prof[('load', 'p_mw')], prof[('load', 'q_mvar')], prof[('sgen', 'p_mw')]
            t = pd.date_range('2016-01-01', periods=len(lp), freq='15min')
            win = (t.weekday < 5) & (t.hour >= 7) & (t.hour < 19)
            net_load = lp.sum(1).to_numpy() - sg.sum(1).to_numpy()
            k = int(cfg.get('simbench_critical_steps', 24))
            cand = np.flatnonzero(win)
            self.steps = list(cand[np.argsort(net_load[cand])[::-1][:k]])
            self.lp, self.lq, self.sg = lp.to_numpy(), lq.to_numpy(), sg.to_numpy()
            mv = net.bus.index[net.bus['vn_kv'] < 100]
            self.vmin = float(net.bus.loc[mv, 'min_vm_pu'].max()); self.vmax = float(net.bus.loc[mv, 'max_vm_pu'].min())
            self.thermal = True
        self.net = net
        self.cand = np.array(sorted(set(net.load['bus']) - set(net.ext_grid['bus'])))
        self.hub_idx = np.array([pp.create_load(net, int(b), p_mw=0.0, q_mvar=0.0, name=f'hub {b}') for b in self.cand])
        self.mv = net.bus.index[net.bus['vn_kv'] < 100]

    def ok(self, hubs_per_bus: np.ndarray, kw: float) -> tuple[bool, str]:
        net, pp = self.net, self.pp
        net.load.loc[self.hub_idx, 'p_mw'] = hubs_per_bus * kw / 1000.0
        base = ~net.load.index.isin(self.hub_idx)
        for s in self.steps:
            if s is not None:
                net.load.loc[base, 'p_mw'] = self.lp[s]; net.load.loc[base, 'q_mvar'] = self.lq[s]
                net.sgen['p_mw'] = self.sg[s]
            try:
                pp.runpp(net, numba=True, init='auto')
            except Exception:                                       # noqa: BLE001  non-convergence counts as a violation
                return False, 'power flow did not converge'
            v = net.res_bus.loc[self.mv, 'vm_pu']
            if v.min() < self.vmin or v.max() > self.vmax:
                return False, 'voltage'
            if self.thermal and (net.res_line['loading_percent'].max() > 100 or net.res_trafo['loading_percent'].max() > 100):
                return False, 'thermal'
        return True, ''

    def hosting(self, order: np.ndarray, kw: float, nmax: int) -> tuple[float, str]:
        """Hosting capacity of one placement order, in hubs: the largest n such that the first n hubs cause no
        violation (monotone, bisection), plus the largest fraction of hub n + 1 that still fits, so that the result
        varies continuously with the limit per hub."""
        def counts(n, frac=0.0):
            c = np.bincount(order[:n], minlength=len(self.cand)).astype(float)
            if frac > 0:
                c[order[n]] += frac
            return c
        good, _ = self.ok(counts(0), kw)
        if not good:
            return 0.0, 'base case'
        lo, hi = 0, nmax
        okhi, why = self.ok(counts(hi), kw)
        if okhi:
            return float(hi), 'not reached'
        while hi - lo > 1:
            mid = (lo + hi) // 2
            g, w = self.ok(counts(mid), kw)
            if g:
                lo = mid
            else:
                hi, why = mid, w
        a, b = 0.0, 1.0
        for _ in range(7):
            m = (a + b) / 2
            if self.ok(counts(lo, m), kw)[0]:
                a = m
            else:
                b = m
        return lo + a, why

def run(ctx: Ctx):
    cfg = ctx.cfg.get('feeder', {})
    feeders = list(cfg.get('feeders', ['ieee33', '1-MV-comm--0-sw']))
    n_orders = 5 if ctx.smoke else int(cfg.get('orders', 40))
    lim = contracted_limits(ctx)
    lim.to_csv(ctx.p('feeder_contracted_limits.csv'), index=False)
    rng = np.random.default_rng(20261124)
    rows, per_order = [], []
    for fname in feeders:
        F = Feeder(fname, cfg)
        nmax = int(cfg.get('nmax', 1000))
        orders = [rng.integers(0, len(F.cand), nmax) for _ in range(n_orders)]
        for site, g in lim.groupby('site'):
            kws = {r['forecast']: float(r['limit_kw']) for _, r in g.iterrows()}
            # paired interval of the cold-start contrast: the deployment limit plus the bounds of the paired
            # bootstrap difference in capacity needed (acnsim_capacity_contrasts.csv)
            extra = {}
            cc = ctx.p('acnsim_capacity_contrasts.csv')
            if cc.exists() and 'unseen driver, own history' in kws:
                c = pd.read_csv(cc)
                c = c[(c['site'] == site) & (c['contrast'] == 'cold - deployment') & (c['tau'].astype(str).isin(['0.5', '0.50']))]
                if len(c):
                    rated = float(c['estimate_kw'].iloc[0] / c['estimate_fraction'].iloc[0])
                    extra = {'cold start (contrast low)': kws['unseen driver, own history'] + float(c['ci_low'].iloc[0]) * rated,
                             'cold start (contrast high)': kws['unseen driver, own history'] + float(c['ci_high'].iloc[0]) * rated}
            for label, kw in {**kws, **extra}.items():
                res = [F.hosting(o, kw, nmax) for o in orders]
                hubs = np.array([h for h, _ in res])
                per_order += [{'feeder': fname, 'site': site, 'forecast': label, 'order': i, 'hubs': h, 'binding': w}
                              for i, (h, w) in enumerate(res)]
                rows.append({'feeder': fname, 'site': site, 'forecast': label, 'kw_per_hub': kw,
                             'hubs_mean': float(hubs.mean()), 'hubs_median': float(np.median(hubs)),
                             'hubs_p5': float(np.percentile(hubs, 5)), 'hubs_p95': float(np.percentile(hubs, 95)),
                             'mw_mean': float(hubs.mean() * kw / 1000),
                             'binding_voltage_pct': 100 * np.mean([w == 'voltage' for _, w in res]),
                             'binding_thermal_pct': 100 * np.mean([w == 'thermal' for _, w in res]),
                             'orders': n_orders, 'vmin_pu': F.vmin, 'vmax_pu': F.vmax, 'critical_steps': len(F.steps)})
            ctx.log(f'feeder {fname} {site}: done')
    out = pd.DataFrame(rows); out.to_csv(ctx.p('feeder_hosting_capacity.csv'), index=False)
    po = pd.DataFrame(per_order); po.to_csv(ctx.p('feeder_hosting_by_order.csv'), index=False)
    # paired contrasts over orders
    con = []
    for (fname, site), g in po.groupby(['feeder', 'site']):
        h = g.pivot(index='order', columns='forecast', values='hubs')
        def d(a, b):
            x = (h[a] - h[b]).to_numpy(float)
            return float(np.median(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95)), float(np.mean(x))
        for a, b in [('unseen driver, own history', 'cold start'), ('perfect information', 'no forecast (FCFS)'),
                     ('unseen driver, own history', 'no forecast (FCFS)'), ('cold start', 'no forecast (FCFS)')]:
            if a in h and b in h:
                med, lo, hi, mean = d(a, b)
                con.append({'feeder': fname, 'site': site, 'contrast': f'{a} minus {b} (hubs)', 'median': med, 'p5': lo, 'p95': hi,
                            'mean': mean, 'relative_pct_of_b': 100 * mean / float(h[b].mean())})
        for bound in ('low', 'high'):
            col = f'cold start (contrast {bound})'
            if col in h and 'unseen driver, own history' in h:
                x = (h['unseen driver, own history'] - h[col]).mean()
                con.append({'feeder': fname, 'site': site, 'contrast': f'unseen driver, own history minus cold start (hubs), {bound} bound of capacity contrast',
                            'mean': float(x), 'relative_pct_of_b': 100 * x / float(h[col].mean())})
        if {'unseen driver, own history', 'cold start', 'no forecast (FCFS)'} <= set(h.columns):
            gain = (h['unseen driver, own history'] - h['no forecast (FCFS)']).mean()
            lost = (h['unseen driver, own history'] - h['cold start']).mean()
            con.append({'feeder': fname, 'site': site, 'contrast': 'share of the forecast hosting gain forfeited by a cold start (%)',
                        'mean': 100 * lost / gain if gain else np.nan})
    pd.DataFrame(con).to_csv(ctx.p('feeder_contrasts.csv'), index=False)
    ctx.log('feeder hosting capacity written')
