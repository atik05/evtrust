"""Time-series hosting capacity: hubs that draw their simulated load, not their contracted limit.

feeder.py connects hubs at their contracted limits (the capacity each forecast needs for at most 2 % undelivered
energy) and assumes that every hub draws its full limit through the critical periods of the feeder, the planning view
of a firm connection. Here every hub instead draws the load that ACN-Sim simulates for the site connected at that
limit under the same scheduler and forecast (experiment E6 of acnsim.py), and the feeder is checked at every
15-minute step of the year:

  * hub load profiles: the 5-minute site load of the E6 run, averaged to 15 minutes, for the year beginning Friday
    4 January 2019, mapped day by day onto the SimBench year (2016, which begins on a Friday), so weekdays and
    weekends align. Two cases:
      coincident  every hub of a site draws the same profile, so their peaks add up exactly (as in feeder.py; the
                  most demanding case)
      diverse     every hub draws, for each week of the year, the site's load of a week drawn at random from the
                  four weeks before to the four weeks after (same weekdays, same season), so hubs keep the site's
                  daily and weekly pattern but their sessions do not coincide; hubs cycle through a fixed set of such
                  profiles in placement order, and the draws are the same for every forecast, seed and order, so
                  the comparison between forecasts stays paired
  * feeder: SimBench 1-MV-comm--0-sw with its own load and photovoltaic profiles, voltage band and line and
    transformer ratings; hubs at unity power factor on the load buses of the feeder
  * a step violates when a voltage, line or transformer limit is exceeded and the hubs make it worse than the feeder
    without hubs at that step (photovoltaic overvoltage that the feeder already has at midday is not charged to the
    hubs, whose load lowers it); every step with hub load is checked
  * hosting capacity of a placement order: the largest number of hubs with no violation at any step (bisection) plus
    the largest fraction of the next hub that still fits, as in feeder.py; the orders are the first orders of
    feeder.py's random stream for this feeder, so results are paired with the deterministic analysis order by order

Power flows use robustness.fastpf (pandapower's admittance matrices and Newton-Raphson iteration; validated against
pandapower when the analysis starts, and the validation is written to the outputs).
Outputs (column hub_profiles: coincident or diverse)
  feeder_ts_hosting.csv     hubs and MW hosted per site, forecast and case (mean, median and 5-95 % over orders), the
                            hub's contracted limit, peak and mean load, the coincidence factor of the hub profiles and
                            the binding constraint; the row '{CHECK}' repeats feeder.py's assumption (deployment limit
                            through weekday 07:00 to 19:00) under the time-series check, so its hosting should equal
                            feeder.py's for the same orders
  feeder_ts_by_order.csv    hosting capacity per placement order (paired with feeder_hosting_by_order.csv)
  feeder_ts_contrasts.csv   paired differences between forecasts and the share of the forecast's hosting gain
                            (deployment against no forecast) that a cold start forfeits, for each case
  feeder_ts_validation.json agreement of the power flow with pandapower, base-case violations of the feeder
"""
from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .common import Ctx

SIMBENCH = '1-MV-comm--0-sw'
POLICIES = [('fcfs', 'none', None, 'no forecast (FCFS)'), ('llf', 'oracle', None, 'perfect information'),
            ('llf', 'benchmark', 0.5, 'driver seen'), ('llf', 'deployment', 0.5, 'unseen driver, own history'),
            ('llf', 'cold', 0.5, 'cold start')]
CHECK = 'deployment limit through weekday 07:00-19:00 (check)'
YEAR0, SOURCE0 = pd.Timestamp('2016-01-01'), pd.Timestamp('2019-01-04')     # both Fridays
STEPS = 366 * 96
TOL = 1e-6
DIVERSITY_SEED = 20261126


# ------------------------------------------------------------------------------------------ hub profiles
def site_series(npz_path, origin: pd.Timestamp, period: int) -> pd.Series:
    """15-minute site load (kW) of one ACN-Sim run on its own calendar (weeks without sessions draw nothing)."""
    pw = np.load(npz_path)
    n = max(int(o) + len(pw[f'p{int(o)}']) for o in pw['offsets'])
    load = np.zeros(n)
    for o in pw['offsets']:
        p = pw[f'p{int(o)}']; load[int(o): int(o) + len(p)] += p
    s = pd.Series(load, index=origin + pd.to_timedelta(np.arange(n) * period, unit='min'))
    return s.resample('15min').mean()


def week_shifts(n_variants: int, window: int, seed: int = DIVERSITY_SEED) -> np.ndarray:
    """Shift in weeks (-window..window) for every profile variant and week of the hub year (variants x 53)."""
    rng = np.random.default_rng(seed)
    return rng.integers(-window, window + 1, size=(n_variants, 53))


def year_profiles(s15: pd.Series, shifts: np.ndarray | None = None, strict: bool = True) -> np.ndarray:
    """Hub profiles on the SimBench calendar (variants x STEPS, kW). Without shifts, one profile: the year from SOURCE0.
    With shifts, each week of the year is taken from the source week moved by a whole number of weeks, so weekdays
    stay aligned."""
    cal = pd.date_range(YEAR0, periods=STEPS, freq='15min')
    base = SOURCE0 + (cal - YEAR0)
    week = ((cal - YEAR0).days // 7).to_numpy()
    sh = np.zeros((1, 53), int) if shifts is None else np.asarray(shifts, int)
    out = np.empty((sh.shape[0], STEPS))
    for m in range(sh.shape[0]):
        src = base + pd.to_timedelta(7 * sh[m, week], unit='D')
        if strict and (src.min() < s15.index[0] or src.max() > s15.index[-1]):
            raise ValueError(f'the run covers {s15.index[0]} to {s15.index[-1]}, the hub year needs {src.min()} to {src.max()}')
        out[m] = s15.reindex(src).fillna(0.0).to_numpy(float)          # gaps only in smoke runs (strict=False)
    return out


def hub_profile(npz_path, origin: pd.Timestamp, period: int, strict: bool = True) -> np.ndarray:
    """The coincident hub profile (STEPS, kW) of one ACN-Sim run."""
    return year_profiles(site_series(npz_path, origin, period), None, strict)[0]


def coincidence_factor(P: np.ndarray) -> float:
    """Peak of the summed profiles over the sum of the individual peaks (1 for identical profiles)."""
    peaks = P.max(axis=1).sum()
    return float(P.sum(axis=0).max() / peaks) if peaks > 0 else np.nan


def simbench_orders(cfg: dict, n_orders: int) -> list:
    """The first placement orders that feeder.py draws for the SimBench feeder (same seed, same stream)."""
    import pandapower.networks as pn
    import simbench as sb
    nmax, n_det = int(cfg.get('nmax', 1000)), int(cfg.get('orders', 40))
    rng = np.random.default_rng(20261124)
    for fname in list(cfg.get('feeders', ['ieee33', SIMBENCH])):
        net = pn.case33bw() if fname == 'ieee33' else sb.get_simbench_net(fname)
        ncand = len(set(net.load['bus']) - set(net.ext_grid['bus']))
        orders = [rng.integers(0, ncand, nmax) for _ in range(n_det)]
        if fname == SIMBENCH:
            return orders[:n_orders]
    raise ValueError(f'{SIMBENCH} is not among the feeders of feeder.py')


def check_steps(P: np.ndarray) -> list:
    """Steps with hub load in any profile, most loaded first (a static order; failing steps move to the front)."""
    tot = P.sum(axis=0)
    return [int(t) for t in np.argsort(-tot, kind='stable') if tot[t] > 0]


# ----------------------------------------------------------------------------------------------- feeder
class TSFeeder:
    def __init__(self, name: str = SIMBENCH):
        import simbench as sb
        from .fastpf import FastPF
        warnings.filterwarnings('ignore')
        net = sb.get_simbench_net(name)
        prof = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        P, Q, G = (prof[k] for k in (('load', 'p_mw'), ('load', 'q_mvar'), ('sgen', 'p_mw')))
        self.lp, self.lq, self.sg = P.to_numpy(float), Q.to_numpy(float), G.to_numpy(float)
        if self.lp.shape[0] != STEPS:
            raise ValueError(f'{name}: {self.lp.shape[0]} profile steps, expected {STEPS}')
        self.cand = np.array(sorted(set(net.load['bus']) - set(net.ext_grid['bus'])))     # as feeder.py
        mv = net.bus.index[net.bus['vn_kv'] < 100]
        self.vmin = float(net.bus.loc[mv, 'min_vm_pu'].max()); self.vmax = float(net.bus.loc[mv, 'max_vm_pu'].min())
        self.F = F = FastPF(net)
        self.sq = net.sgen.loc[F.sgen_rows, 'q_mvar'].to_numpy(float)
        self.lrow = P.columns.get_indexer(F.load_rows); self.grow = G.columns.get_indexer(F.sgen_rows)   # profile columns
        if (self.lrow < 0).any() or (self.grow < 0).any() or not P.columns.equals(Q.columns):
            raise ValueError('profiles do not cover every load and generator')
        mvi = F.lk[mv.to_numpy()]
        self.mv = np.unique(mvi[mvi >= 0])
        self.hub_unit = np.stack([F.bus_vector([b], [1.0]) for b in self.cand])              # cand x internal buses
        self._S, self._V, self.base = {}, {}, None

    def S_base(self, t):
        if t not in self._S:
            F = self.F
            self._S[t] = F.sbus(self.lp[t, self.lrow], self.lq[t, self.lrow], self.sg[t, self.grow], self.sq)
        return self._S[t]

    def metrics(self, V):
        ll, th = self.F.loadings(V)
        vm = np.abs(V)[self.mv]
        return float(vm.min()), float(vm.max()), float(ll.max()), float(th.max())

    def base_case(self, steps) -> np.ndarray:
        """Feeder without hubs at the given steps: (vmin, vmax, line, trafo) per step; warm starts are kept."""
        out = np.empty((len(steps), 4))
        V = None
        for i, t in enumerate(steps):
            V, ok = self.F.solve(self.S_base(t), V)
            if not ok:
                raise RuntimeError(f'base case does not converge at step {t}')
            self._V[t] = V
            out[i] = self.metrics(V)
        return out

    def set_base(self, steps, base: np.ndarray, V: np.ndarray | None = None):
        self.base = {int(t): b for t, b in zip(steps, base)}
        if V is not None:
            self._V = {int(t): v for t, v in zip(steps, V)}

    def ok(self, C: np.ndarray, P: np.ndarray, order: list, hot: list) -> tuple[bool, str]:
        """No violation caused by the hubs at any step with hub load. C: hubs per candidate bus and profile
        (cand x variants), P: profiles (variants x STEPS, kW). Steps that failed before are checked first."""
        C = C[:, None] if C.ndim == 1 else C                                              # one profile: counts per bus
        P = np.atleast_2d(P)
        if not C.any():
            return True, ''
        Gv = (C.T @ self.hub_unit) / 1000.0 / self.F.base                                  # variants x buses, per kW
        seen = set()
        for t in hot + order:
            if t in seen:
                continue
            seen.add(t)
            S = self.S_base(t) - P[:, t] @ Gv
            V, conv = self.F.solve(S, self._V.get(t))
            if not conv:
                why = 'power flow did not converge'
            else:
                vmin, vmax, ll, th = self.metrics(V)
                b = self.base[t]
                why = ('voltage' if (vmin < self.vmin and vmin < b[0] - TOL) or (vmax > self.vmax and vmax > b[1] + TOL) else
                       'thermal' if (ll > 100 and ll > b[2] + TOL) or (th > 100 and th > b[3] + TOL) else '')
            if why:
                if t in hot:
                    hot.remove(t)
                hot.insert(0, t)
                return False, why
        return True, ''

    def hosting(self, order_buses: np.ndarray, P: np.ndarray, steps: list, nmax: int) -> tuple[float, str, int]:
        """As feeder.Feeder.hosting, with the time-series check. Hub i of the order (0-based) draws profile i mod
        variants. P may be one profile (STEPS,) or several (variants x STEPS). Returns (hubs, binding, checks)."""
        P = np.atleast_2d(P)
        nv = P.shape[0]
        hot: list = []
        calls = [0]

        def chk(n, frac=0.0):
            C = np.zeros((len(self.cand), nv))
            idx = np.arange(n)
            np.add.at(C, (order_buses[:n], idx % nv), 1.0)
            if frac > 0:
                C[order_buses[n], n % nv] += frac
            calls[0] += 1
            return self.ok(C, P, steps, hot)
        lo, hi = 0, nmax
        okhi, why = chk(hi)
        if okhi:
            return float(hi), 'not reached', calls[0]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            g, w = chk(mid)
            if g:
                lo = mid
            else:
                hi, why = mid, w
        a, b = 0.0, 1.0
        for _ in range(7):
            m = (a + b) / 2
            g, w = chk(lo, m)
            if g:
                a = m
            else:
                b = m
        return lo + a, why, calls[0]


# ------------------------------------------------------------------------------------------------ workers
_W: dict = {}


def _init(profiles: dict, steps: dict, base_steps: list, base: np.ndarray, V: np.ndarray):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                                  # noqa: BLE001
        pass
    F = TSFeeder()
    F.set_base(base_steps, base, V)                                    # limits and warm starts of the base case
    _W.update({'F': F, 'profiles': profiles, 'steps': steps})


def _task(args):
    key, i, order, nmax = args
    F = _W['F']
    h, why, calls = F.hosting(order, _W['profiles'][key], _W['steps'][key], nmax)
    return {'key': key, 'order': i, 'hubs': h, 'binding': why, 'checks': calls}


# ---------------------------------------------------------------------------------------------------- run
def run(ctx: Ctx):
    from .acnsim import rated_kw, run_id, site_runs
    cfg = ctx.cfg.get('feeder', {})
    sc = ctx.cfg['acnsim']
    n_orders = 2 if ctx.smoke else int(cfg.get('ts_orders', 10))
    nmax = int(cfg.get('nmax', 1000))
    diverse = bool(cfg.get('ts_diverse', True))
    n_var, window = int(cfg.get('ts_variants', 20)), int(cfg.get('ts_week_window', 4))
    workers = sc.get('workers', 'auto')
    workers = max(1, (os.cpu_count() or 4) - 4) if workers == 'auto' else int(workers)
    rdir = ctx.p('acnsim', 'runs', 'x').parent
    info = json.loads(ctx.p('acnsim', 'site_info.json').read_text())
    period = int(sc['period_minutes'])
    shifts = week_shifts(n_var, window)
    profiles, limits = {}, {}                                          # key: (site, forecast, hub_profiles)
    for site in sc['sites']:
        for r in site_runs(sc, site):
            label = next(p[3] for p in POLICIES if (p[0], p[1], p[2]) == (r[1], r[3], r[4]))
            f = rdir / f'{run_id(site, r)}.power.npz'
            if not f.exists():
                raise FileNotFoundError(f'{f.name}: run the E6 simulations first')
            s15 = site_series(f, pd.Timestamp(info[site]['origin']), period)
            P = year_profiles(s15, None, strict=not ctx.smoke)
            if ctx.smoke and not P.any():                              # a smoke week outside the hub year
                P[:, ::7] = 0.5 * r[2] * rated_kw(site)
            profiles[(site, label, 'coincident')] = P
            limits[(site, label)] = r[2] * rated_kw(site)
            if diverse:
                Pd = year_profiles(s15, shifts, strict=not ctx.smoke)
                if ctx.smoke and not Pd.any():
                    Pd[:, ::7] = 0.5 * r[2] * rated_kw(site)
                profiles[(site, label, 'diverse')] = Pd
    if not profiles:
        raise RuntimeError('no E6 runs configured (acnsim.e6_capacity_needed)')
    # consistency check: hubs that draw the deployment limit through weekday 07:00-19:00, the assumption of feeder.py;
    # the time-series check of every step should then give feeder.py's hosting capacity for the same orders
    cal = pd.date_range(YEAR0, periods=STEPS, freq='15min')
    win = (cal.weekday < 5) & (cal.hour >= 7) & (cal.hour < 19)
    for site in sc['sites']:
        if (site, 'unseen driver, own history') in limits:
            L = limits[(site, 'unseen driver, own history')]
            profiles[(site, CHECK, 'coincident')] = np.where(win, L, 0.0)[None, :]; limits[(site, CHECK)] = L
    profiles = {k: v.astype(np.float32) for k, v in profiles.items()}     # kW; halves the memory of each worker
    steps = {k: check_steps(v) for k, v in profiles.items()}
    if ctx.smoke:                                                      # smoke runs test the pipeline, not the year
        steps = {k: v[:150] for k, v in steps.items()}
    union = sorted(set().union(*[set(s) for s in steps.values()]))
    F = TSFeeder()
    val = F.F.validate(n=20 if ctx.smoke else 100, seed=1)
    ctx.log(f'feeder_ts: power flow against pandapower {val}')
    allbase = F.base_case(list(range(0, STEPS, 1 if not ctx.smoke else 97)))
    base = F.base_case(union)                                          # last, so the warm starts are those steps'
    vinfo = {'validation': val, 'steps_with_hub_load': len(union), 'vmin_pu': F.vmin, 'vmax_pu': F.vmax,
             'base_case_steps_checked': int(len(allbase)),
             'base_case_violations': {'undervoltage': int((allbase[:, 0] < F.vmin).sum()), 'overvoltage': int((allbase[:, 1] > F.vmax).sum()),
                                      'line': int((allbase[:, 2] > 100).sum()), 'transformer': int((allbase[:, 3] > 100).sum())},
             'hub_year': f'{SOURCE0.date()} to {(SOURCE0 + pd.Timedelta(days=366) - pd.Timedelta(minutes=15))} mapped onto 2016',
             'diverse': {'variants': n_var, 'week_window': window, 'seed': DIVERSITY_SEED} if diverse else None}
    ctx.p('feeder_ts_validation.json').write_text(json.dumps(vinfo, indent=1))
    orders = simbench_orders(cfg, n_orders)
    tasks = [(k, i, o, nmax) for k in profiles for i, o in enumerate(orders)]
    ctx.log(f'feeder_ts: {len(tasks)} searches ({len(profiles)} profile sets x {len(orders)} orders), {len(union)} steps with hub load, {workers} workers')
    res = []
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), initializer=_init,
                             initargs=(profiles, steps, union, base, np.stack([F._V[t] for t in union]))) as ex:
        for j, r in enumerate(ex.map(_task, tasks)):
            res.append(r)
            if (j + 1) % 10 == 0:
                ctx.log(f'feeder_ts: {j + 1}/{len(tasks)} searches')
    po = pd.DataFrame([{'feeder': SIMBENCH, 'site': r['key'][0], 'forecast': r['key'][1], 'hub_profiles': r['key'][2],
                        'order': r['order'], 'hubs': r['hubs'], 'binding': r['binding'], 'checks': r['checks']} for r in res])
    po.to_csv(ctx.p('feeder_ts_by_order.csv'), index=False)
    rows = []
    wd = (cal.weekday < 5)
    for (site, label, kind), g in po.groupby(['site', 'forecast', 'hub_profiles']):
        P = profiles[(site, label, kind)]; h = g['hubs'].to_numpy(float); L = limits[(site, label)]
        rows.append({'feeder': SIMBENCH, 'site': site, 'forecast': label, 'hub_profiles': kind, 'limit_kw': L,
                     'hub_peak_kw': float(P.max(axis=1).mean()), 'hub_mean_kw_weekdays': float(P[:, wd].mean()),
                     'hub_energy_mwh_year': float(P.sum(axis=1).mean() * 0.25 / 1000),
                     'coincidence_factor': coincidence_factor(P) if P.shape[0] > 1 else 1.0, 'profile_variants': P.shape[0],
                     'hubs_mean': float(h.mean()), 'hubs_median': float(np.median(h)), 'hubs_p5': float(np.percentile(h, 5)),
                     'hubs_p95': float(np.percentile(h, 95)), 'mw_contracted_mean': float(h.mean() * L / 1000),
                     'binding_voltage_pct': 100 * float((g['binding'] == 'voltage').mean()),
                     'binding_thermal_pct': 100 * float((g['binding'] == 'thermal').mean()),
                     'not_reached_pct': 100 * float((g['binding'] == 'not reached').mean()), 'orders': len(g)})
    pd.DataFrame(rows).to_csv(ctx.p('feeder_ts_hosting.csv'), index=False)
    con = []
    for (site, kind), g in po.groupby(['site', 'hub_profiles']):
        h = g.pivot(index='order', columns='forecast', values='hubs')
        for a, b in [('unseen driver, own history', 'cold start'), ('perfect information', 'no forecast (FCFS)'),
                     ('unseen driver, own history', 'no forecast (FCFS)'), ('cold start', 'no forecast (FCFS)'),
                     ('driver seen', 'unseen driver, own history')]:
            if a in h and b in h:
                x = (h[a] - h[b]).to_numpy(float)
                con.append({'feeder': SIMBENCH, 'site': site, 'hub_profiles': kind, 'contrast': f'{a} minus {b} (hubs)',
                            'mean': float(x.mean()), 'median': float(np.median(x)), 'p5': float(np.percentile(x, 5)),
                            'p95': float(np.percentile(x, 95)), 'relative_pct_of_b': 100 * float(x.mean() / h[b].mean())})
        if {'unseen driver, own history', 'cold start', 'no forecast (FCFS)'} <= set(h.columns):
            gain = (h['unseen driver, own history'] - h['no forecast (FCFS)']).mean()
            lost = (h['unseen driver, own history'] - h['cold start']).mean()
            con.append({'feeder': SIMBENCH, 'site': site, 'hub_profiles': kind,
                        'contrast': 'share of the forecast hosting gain forfeited by a cold start (%)',
                        'mean': float(100 * lost / gain) if gain else np.nan})
    # diverse against coincident, same forecast and order
    if diverse:
        for (site, label), g in po[po['forecast'] != CHECK].groupby(['site', 'forecast']):
            w = g.pivot(index='order', columns='hub_profiles', values='hubs')
            if {'coincident', 'diverse'} <= set(w.columns):
                x = (w['diverse'] / w['coincident'] - 1).to_numpy(float) * 100
                con.append({'feeder': SIMBENCH, 'site': site, 'hub_profiles': 'diverse vs coincident',
                            'contrast': f'{label}: diverse over coincident hubs (%)', 'mean': float(np.mean(x)),
                            'median': float(np.median(x)), 'p5': float(np.percentile(x, 5)), 'p95': float(np.percentile(x, 95))})
    pd.DataFrame(con).to_csv(ctx.p('feeder_ts_contrasts.csv'), index=False)
    ctx.log('feeder_ts: time-series hosting capacity written')
