"""Grid-level consequences in ACN-Sim (acnportal 0.3.3) on the real Caltech and JPL networks.

Every session of the decomposition window is replayed on the site's three-phase network model (EVSE types,
transformer and phase constraints of acnportal). The scheduler sees forecasts only:
  * estimated departure = forecast departure, requested energy = forecast energy (shifted by the same
    cross-conformal quantiles as in the paper, each population calibrated on its own);
  * acnportal's own Interface passes the TRUE departure to schedulers through SessionInfo.departure
    (used for remaining_time and hence by the optimisation scheduler and by the rate limits). The
    ForecastInterface below replaces it by the forecast, so no scheduler sees the true departure. A session
    still plugged in after its forecast departure is treated as leaving in the next period.
  * the vehicle physically takes its true energy (battery capacity) at up to 6.6 kW and leaves at its true time;
    past its forecast energy it stays at a small remaining demand until full, so the energy forecast sets
    priority and the start of charging but never cuts a session short (as in the per-session policy).
Forecast inputs (2b, CatBoost, one seed): oracle, benchmark (cohort random split), deployment (unseen
driver with own history), cold (history hidden), driver request (deployment where none), no forecast
(first come first served until full), and uncontrolled charging.
Experiments
  E1  real network capacity, least-laxity-first (LLF): undelivered energy against the quantile shift
  E2  transformer capacity swept from 30 % to 100 % of its rating, LLF (and EDF at the raw forecasts):
      capacity needed for <= 2 % undelivered energy (day-block bootstrap), and the capacity that keeps
      undelivered energy <= 2 % on 95 % of days (firm capacity an operator could commit)
  E3  model predictive control (adacharge ASA: energy delivered, TOU energy cost, quick charge), sampled
      weeks: TOU energy cost, peak, demand charge, undelivered energy
  E4  weekday demand-response event (16:00 to 21:00 by default): during the event every constraint limit is
      scaled to a fraction (0 to 0.5), outside it the full network is available; the firm limit is the smallest
      fraction that keeps undelivered energy <= 2 % on at least 95 % of event days, and the dependable load
      reduction is the uncontrolled event-window load minus the realised event-window load under that limit
  E5  weekday morning congestion event (09:00 to 13:00 by default), when workplace charging load is
      highest: the same construction as E4, with event limits suited to the higher load in that window
The capacity sweep includes 60 % of the rating, and paired contrasts between forecasts in the capacity needed use a
day-block bootstrap shared across the forecasts of a site.
Weeks are simulated independently (Monday 00:00 local); sessions spanning a week boundary are counted.
"""
from __future__ import annotations

import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Ctx, write_json

TZ = 'America/Los_Angeles'
PMAX = 6.6
BASE_CAPS = {'caltech': {'transformer_cap': 150.0}, 'jpl': {'first_transformer_cap': 45.0, 'third_fourth_transformer_cap': 150.0},
             'office001': {'transformer_cap': 50.0}}


# ------------------------------------------------------------------------------------- acnportal parts
def make_network(site: str, frac: float):
    from acnportal.acnsim.network.sites import caltech_acn, jpl_acn, office001_acn
    f = {'caltech': caltech_acn, 'jpl': jpl_acn, 'office001': office001_acn}[site]
    return f(basic_evse=False, **{k: v * frac for k, v in BASE_CAPS[site].items()})


def rated_kw(site: str) -> float:
    return float(sum(BASE_CAPS[site].values()))


def _forecast_interface(event: dict | None = None, start: datetime | None = None, period: int = 5):
    from acnportal.acnsim.interface import Interface, SessionInfo

    class ForecastInterface(Interface):
        def infrastructure_info(self):
            # E4: during a demand-response event (weekdays, event hours) the scheduler must keep the site within
            # a fraction of every constraint limit; outside the event the full network is available
            info = super().infrastructure_info()
            if event is not None:
                t = start + pd.Timedelta(minutes=period * self.current_time)
                if t.weekday() < 5 and event['hours'][0] <= t.hour < event['hours'][1]:
                    info.constraint_limits = info.constraint_limits * float(event['frac'])
            return info

        def _active_sessions(self):
            out = []
            for ev in self._active_evs:
                est = max(int(ev.estimated_departure), self.current_time + 1, int(ev.arrival) + 1)
                # The forecast energy sets the remaining demand. A vehicle that has received its forecast but still
                # draws current is kept at a small remaining demand, so it charges when capacity is free and stops
                # only when it is full or leaves (as in the per-session policy of the paper). Being full is
                # observable: the vehicle stops drawing current.
                bat = ev._battery
                full = bat._current_charge >= bat._capacity - 1e-6
                topup = PMAX * self.period / 60 * 1.05             # one period at full power, so charging is not slowed
                remaining = 0.0 if full else max(ev.requested_energy - ev.energy_delivered, topup)
                out.append(SessionInfo(ev.station_id, ev.session_id, ev.energy_delivered + remaining, ev.energy_delivered,
                                       ev.arrival, est, est, self.current_time))
            return out
    return ForecastInterface


def _forecast_ev():
    from acnportal.acnsim.models import EV

    class ForecastEV(EV):
        """requested_energy holds the FORECAST; the session is finished only when the battery is full, because
        acnportal's network drops an EV from the active set once requested_energy is reached."""
        @property
        def fully_charged(self):
            return self._battery._current_charge >= self._battery._capacity - 1e-3
    return ForecastEV


def make_scheduler(name: str, solver: str = 'CLARABEL'):
    from acnportal import algorithms as A
    if name == 'uncontrolled':
        return A.UncontrolledCharging()
    if name == 'llf':
        return A.SortedSchedulingAlgo(A.least_laxity_first)
    if name == 'edf':
        return A.SortedSchedulingAlgo(A.earliest_deadline_first)
    if name == 'fcfs':
        return A.SortedSchedulingAlgo(A.first_come_first_served)
    if name == 'asa':
        import adacharge as ad
        obj = [ad.ObjectiveComponent(ad.total_energy, 1000), ad.ObjectiveComponent(ad.tou_energy_cost),
               ad.ObjectiveComponent(ad.quick_charge, 1e-4)]
        # real EVSEs accept only discrete pilots: quantize and reallocate as in the ACN-Sim papers
        return ad.AdaptiveSchedulingAlgorithm(obj, solver=solver, quantize=True, reallocate=True)
    raise ValueError(name)


def _single_thread():
    """Each simulation worker uses one CPU thread for numpy, so the workers do not oversubscribe the CPUs."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                            # noqa: BLE001
        pass


def simulate_chunk(task: dict) -> dict:
    """One week of one site under one scheduler, capacity and forecast. Returns per-session energy and load."""
    from acnportal.acnsim import Simulator
    from acnportal.acnsim.events import EventQueue, PluginEvent
    from acnportal.acnsim.models import Battery
    from acnportal.signals.tariffs import TimeOfUseTariff
    import warnings
    warnings.filterwarnings('ignore')
    net = make_network(task['site'], task['cap'])
    q = EventQueue()
    EV = _forecast_ev()
    for sid, st, a, d, de, e_true, e_req in task['evs']:
        bat = Battery(max(e_true, 1e-3), 0.0, PMAX)
        q.add_event(PluginEvent(int(a), EV(int(a), int(d), float(max(e_req, 0.05)), st, sid, bat, estimated_departure=int(de))))
    sim = Simulator(net, make_scheduler(task['scheduler'], task.get('solver', 'CLARABEL')), q, task['start'],
                    period=task['period'], signals={'tariff': TimeOfUseTariff(task['tariff'])}, verbose=False,
                    interface_type=_forecast_interface(task.get('event'), task['start'], task['period']))
    try:
        sim.run()
        status = 'ok'
    except Exception as e:                                       # noqa: BLE001  (solver failures are recorded, not hidden)
        status = f'error: {type(e).__name__}: {e}'[:300]
    from acnportal.acnsim.analysis import aggregate_power
    try:
        power = np.asarray(aggregate_power(sim), float)
    except Exception:                                            # noqa: BLE001
        power = np.zeros(1)
    delivered = {sid: float(ev.energy_delivered) for sid, ev in sim.ev_history.items()}
    return {'run': task['run'], 'chunk': task['chunk'], 'offset': task['offset'], 'status': status,
            'delivered': delivered, 'power': power.astype(np.float32)}


# ------------------------------------------------------------------------------------------ inputs
def load_inputs(ctx: Ctx):
    learner, seed = ctx.cfg['acnsim']['learner'], int(ctx.cfg['acnsim']['seed'])
    if ctx.smoke:
        seed = ctx.seeds[0]
    f = pd.read_parquet(ctx.p(f'deploy_forecasts_{learner}_seed{seed}.parquet'))
    s = ctx.sessions()[['session_id', 'site', 'station_id', 'connection_time', 'disconnect_time']].copy()
    s['session_key'] = s['session_id'].astype(str)
    z = f.merge(s.drop(columns='session_id'), on='session_key', how='inner')
    z['t0'] = pd.to_datetime(z['connection_time'], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    z['t1'] = pd.to_datetime(z['disconnect_time'], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    return z


def forecast_table(z: pd.DataFrame, taus) -> dict:
    """(variant, tau) -> (D_hat hours, E_hat kWh) for every session; quantile shifts within each population."""
    from .common import shifted
    out = {('oracle', None): (z['D'].to_numpy(float), z['E'].to_numpy(float))}
    for var in ('benchmark', 'deployment', 'cold'):
        for tau in taus:
            Dh = np.full(len(z), np.nan); Eh = np.full(len(z), np.nan)
            for pop, g in z.groupby('population'):
                idx = g.index.to_numpy()
                d_, e_ = shifted(g['D'].to_numpy(float), g['E'].to_numpy(float), g[f'Dp_{var}'].to_numpy(float),
                                 g[f'Ep_{var}'].to_numpy(float), g['calib_group'].to_numpy(), [tau])
                Dh[idx], Eh[idx] = d_[0], e_[0]
            out[(var, tau)] = (Dh, Eh)
    rd, re_ = z['requested_duration'].to_numpy(float), z['requested_energy'].to_numpy(float)
    has = np.isfinite(rd) & np.isfinite(re_) & (rd > 0) & (re_ > 0)
    Dd, Ed = out[('deployment', 0.5)]
    out[('request', None)] = (np.where(has, rd, Dd), np.where(has, re_, Ed))
    out[('none', None)] = (np.full(len(z), 24.0), np.full(len(z), 200.0))
    return out


def site_sessions(z: pd.DataFrame, site: str, period: int, station_ids: set, log) -> tuple[pd.DataFrame, dict]:
    g = z[z['site'] == site].sort_values('t0').copy()
    info = {'sessions': len(g)}
    g = g[g['station_id'].astype(str).isin(station_ids)]
    info['station_not_in_network'] = info['sessions'] - len(g)
    origin = g['t0'].min().normalize() - pd.Timedelta(days=g['t0'].min().weekday())     # Monday 00:00
    g['week'] = ((g['t0'] - origin).dt.days // 7).astype(int)
    g['wstart'] = origin + pd.to_timedelta(g['week'] * 7, unit='D')
    g['a'] = np.floor((g['t0'] - g['wstart']).dt.total_seconds() / 60 / period).astype(int)
    g['d'] = np.maximum(np.ceil((g['t1'] - g['wstart']).dt.total_seconds() / 60 / period).astype(int), g['a'] + 1)
    # data quirk: overlapping sessions on one EVSE cannot be replayed; the later one is dropped
    keep = np.ones(len(g), bool); last = {}
    for i, (st, w, a, d) in enumerate(zip(g['station_id'].astype(str), g['week'], g['a'], g['d'])):
        k = (st, w)
        if k in last and a < last[k]:
            keep[i] = False
        else:
            last[k] = d
    info['overlapping_dropped'] = int((~keep).sum())
    g = g[keep]
    info['week_boundary_spanning'] = int((g['t1'] > g['wstart'] + pd.Timedelta(days=7)).sum())
    info['replayed'] = len(g); info['weeks'] = int(g['week'].nunique()); info['origin'] = str(origin)
    g['E_ach'] = np.minimum(g['E'], PMAX * (g['t1'] - g['t0']).dt.total_seconds() / 3600)
    g['day'] = g['t0'].dt.strftime('%Y-%m-%d')
    log(f'ACN-Sim {site}: {info}')
    return g, info


# --------------------------------------------------------------------------------------------- runs
def plan_runs(sc: dict, smoke: bool):
    taus, sweep = sc['taus'], sc['sweep_taus']
    fr = sorted(set(sc['cap_fractions']) | set(sc.get('extra_cap_fractions', [0.6])))
    if smoke:
        taus, fr, sweep = [0.5, 0.1], [0.5, 1.0], [0.5]
    runs = [('E1', 'uncontrolled', 1.0, 'oracle', None), ('E1', 'fcfs', 1.0, 'none', None), ('E1', 'llf', 1.0, 'oracle', None),
            ('E1', 'llf', 1.0, 'request', None)]
    runs += [('E1', 'llf', 1.0, v, t) for v in ('benchmark', 'deployment', 'cold') for t in taus]
    for f in fr:
        runs += [('E2', 'llf', f, 'oracle', None), ('E2', 'fcfs', f, 'none', None), ('E2', 'llf', f, 'request', None)]
        runs += [('E2', 'llf', f, v, t) for v in ('benchmark', 'deployment', 'cold') for t in sweep]
        runs += [('E2', 'edf', f, v, 0.5) for v in ('benchmark', 'deployment', 'cold')] + [('E2', 'edf', f, 'oracle', None)]
    asa = [('E3', 'uncontrolled', 1.0, 'oracle', None), ('E3', 'asa', 1.0, 'oracle', None), ('E3', 'asa', 1.0, 'request', None)]
    asa += [('E3', 'asa', 1.0, v, t) for v in ('benchmark', 'deployment', 'cold') for t in (0.5, 0.1)]
    if smoke:
        asa = [('E3', 'uncontrolled', 1.0, 'oracle', None), ('E3', 'asa', 1.0, 'oracle', None), ('E3', 'asa', 1.0, 'deployment', 0.5)]
    ev = [0.0, 0.3] if smoke else list(sc.get('event_cap_fractions', [0.0, 0.1, 0.2, 0.3, 0.5]))
    e4 = [('E4', 'uncontrolled', 1.0, 'oracle', None)]
    for f in ev:
        e4 += [('E4', 'llf', f, 'oracle', None), ('E4', 'fcfs', f, 'none', None), ('E4', 'llf', f, 'request', None)]
        e4 += [('E4', 'llf', f, v, t) for v in ('benchmark', 'deployment', 'cold') for t in sweep]
    # E5: morning congestion event; uncontrolled charging is not limited by events, so its E1 run is the baseline
    ev5 = [0.3] if smoke else list(sc.get('midday_event_cap_fractions', [0.3, 0.4, 0.5, 0.6, 0.8]))
    e5 = []
    for f in ev5:
        e5 += [('E5', 'llf', f, 'oracle', None), ('E5', 'fcfs', f, 'none', None), ('E5', 'llf', f, 'request', None)]
        e5 += [('E5', 'llf', f, v, t) for v in ('benchmark', 'deployment', 'cold') for t in sweep]
    seen, out = set(), []
    for r in runs + asa + e4 + e5:
        if r not in seen:
            seen.add(r); out.append(r)
    return out


def _event_hours(sc: dict) -> dict:
    return {'E4': list(sc.get('event_hours', [16, 21])), 'E5': list(sc.get('midday_event_hours', [9, 13]))}


def run_id(site, r):
    e, sch, f, v, t = r
    return f'{site}__{e}__{sch}__cap{f:.2f}__{v}__tau{"na" if t is None else t}'


def run(ctx: Ctx):
    from acnportal.signals.tariffs import TimeOfUseTariff
    sc = ctx.cfg['acnsim']
    period = int(sc['period_minutes'])
    asa_period = int(sc.get('asa_period_minutes', 15))
    workers = sc.get('workers', 'auto')
    workers = max(1, (os.cpu_count() or 4) - 4) if workers == 'auto' else int(workers)
    if ctx.smoke:
        workers = min(workers, 4)
    z = load_inputs(ctx)
    taus = sorted(set(sc['taus']) | set(sc['sweep_taus']) | {0.5}, reverse=True)
    fc = forecast_table(z, taus)
    runs = plan_runs(sc, ctx.smoke)
    rdir = ctx.p('acnsim', 'runs', 'x').parent
    infos, tasks, meta = {}, [], {}
    for site in sc['sites']:
        if site not in set(z['site']):
            ctx.log(f'ACN-Sim: no sessions for {site}, skipped'); continue
        stations = set(make_network(site, 1.0).station_ids)
        g, info = site_sessions(z, site, period, stations, ctx.log)
        ga, info_a = site_sessions(z, site, asa_period, stations, ctx.log)        # E3 runs at a coarser step
        infos[site] = info; infos[f'{site} (E3, {asa_period}-minute step)'] = info_a
        weeks = sorted(g['week'].unique())
        if ctx.smoke:
            weeks = weeks[len(weeks) // 2: len(weeks) // 2 + 1]
        elif sc.get('max_weeks'):                                  # optional evenly spaced subsample (testing only)
            weeks = [weeks[int(i)] for i in np.linspace(0, len(weeks) - 1, int(sc['max_weeks'])).round().astype(int)]
        n_asa = 1 if ctx.smoke else int(sc['asa_weeks'])
        asa_weeks = [weeks[int(i)] for i in np.linspace(0, len(weeks) - 1, n_asa).round().astype(int)] if weeks else []
        pos = {k: i for i, k in enumerate(z['session_key'])}
        gi = np.array([pos[k] for k in g['session_key']])
        gia = np.array([pos[k] for k in ga['session_key']])
        for r in runs:
            rid = run_id(site, r)
            e3 = r[0] == 'E3'
            per = asa_period if e3 else period
            meta[rid] = {'site': site, 'experiment': r[0], 'scheduler': r[1], 'cap_fraction': r[2], 'forecast': r[3], 'tau': r[4],
                         'period_min': per}
            if (rdir / f'{rid}.parquet').exists():
                continue
            Dh, Eh = fc[(r[3], r[4])]
            gg, ii = (ga, gia) if e3 else (g, gi)
            tasks.append({'rid': rid, 'site': site, 'r': r, 'g': gg, 'Dh': Dh[ii], 'Eh': Eh[ii], 'period': per,
                          'weeks': [w for w in asa_weeks if w in set(ga['week'])] if e3 else weeks})
    write_json(ctx.p('acnsim', 'site_info.json'), infos)
    # optional split of the simulation between computers: run ids are filtered by regular expressions
    import re
    inc, exc = os.environ.get('EVTRUST_ACNSIM_INCLUDE', ''), os.environ.get('EVTRUST_ACNSIM_EXCLUDE', '')
    if inc or exc:
        n0 = len(tasks)
        tasks = [t for t in tasks if (not inc or re.search(inc, t['rid'])) and not (exc and re.search(exc, t['rid']))]
        ctx.log(f'ACN-Sim: run filter include={inc!r} exclude={exc!r}: {len(tasks)} of {n0} remaining runs kept')
    # tasks are generated lazily, run by run (optimisation runs first), with a bounded number in flight
    plan = sorted(tasks, key=lambda t: 0 if t['r'][1] == 'asa' else 1)
    n_chunks = sum(len(t['weeks']) for t in plan)
    ctx.log(f'ACN-Sim: {len(plan)} runs to do, {n_chunks} week-chunks, {workers} workers')

    event_hours = _event_hours(sc)

    def gen():
        for t in plan:
            Dh, Eh, g, r, period = t['Dh'], t['Eh'], t['g'], t['r'], t['period']
            for w in t['weeks']:
                sel = (g['week'] == w).to_numpy()
                gw = g[sel]
                de = gw['a'].to_numpy() + np.ceil(np.nan_to_num(Dh[sel], nan=24.0) * 60 / period).astype(int)
                e_req = gw['E'].to_numpy(float) if r[3] == 'oracle' else np.nan_to_num(Eh[sel], nan=10.0)
                evs = list(zip(gw['session_key'], gw['station_id'].astype(str), gw['a'].astype(int), gw['d'].astype(int),
                               np.maximum(de, gw['a'].to_numpy() + 1).astype(int), gw['E'].to_numpy(float), e_req))
                ev = event_hours.get(r[0]) if r[1] != 'uncontrolled' else None
                yield {'run': t['rid'], 'chunk': int(w), 'site': t['site'], 'cap': 1.0 if r[0] in event_hours else r[2], 'scheduler': r[1],
                       'period': period, 'event': {'hours': ev, 'frac': r[2]} if ev else None,
                       'start': gw['wstart'].iloc[0].to_pydatetime(), 'offset': int(w) * 7 * 24 * 60 // period, 'evs': evs,
                       'tariff': sc['tariff'], 'solver': sc.get('asa_solver', 'CLARABEL')}

    pending = {t['rid']: len(t['weeks']) for t in plan}
    results = {t['rid']: [] for t in plan}
    done_n = 0
    from concurrent.futures import FIRST_COMPLETED, wait
    it = gen()
    with ProcessPoolExecutor(max_workers=workers, initializer=_single_thread) as ex:
        inflight = set()
        while True:
            while len(inflight) < 3 * workers:
                try:
                    inflight.add(ex.submit(simulate_chunk, next(it)))
                except StopIteration:
                    break
            if not inflight:
                break
            fin, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in fin:
                res = fut.result()
                results[res['run']].append(res)
                pending[res['run']] -= 1; done_n += 1
                if done_n % 250 == 0:
                    ctx.log(f'ACN-Sim: {done_n}/{n_chunks} chunks')
                if pending[res['run']] == 0:
                    _save_run(ctx, res['run'], results.pop(res['run']), rdir, meta)
    unfinished = {k: v for k, v in pending.items() if v != 0}
    if unfinished or done_n != n_chunks:                    # never summarise a partial simulation silently
        raise RuntimeError(f'ACN-Sim: {done_n} of {n_chunks} chunks returned; unfinished runs: {list(unfinished)[:5]}')
    if os.environ.get('EVTRUST_ACNSIM_NO_SUMMARY') == '1':
        ctx.log('ACN-Sim: simulation finished; summary skipped (EVTRUST_ACNSIM_NO_SUMMARY=1)'); return
    summarise(ctx, meta, infos)


def _save_run(ctx, rid, chunks, rdir, meta):
    rows, power, status = [], {}, []
    for c in chunks:
        status.append(c['status'])
        for sid, e in c['delivered'].items():
            rows.append((sid, e))
        power[c['offset']] = c['power']
    df = pd.DataFrame(rows, columns=['session_key', 'delivered_kwh'])
    df.to_parquet(rdir / f'{rid}.tmp.parquet', index=False)
    np.savez_compressed(rdir / f'{rid}.power.npz', offsets=np.array(list(power.keys())),
                        **{f'p{k}': v for k, v in power.items()})
    errs = [s for s in status if s != 'ok']
    write_json(rdir / f'{rid}.status.json', {'chunks': len(status), 'errors': len(errs), 'first_errors': errs[:5]})
    os.replace(rdir / f'{rid}.tmp.parquet', rdir / f'{rid}.parquet')
    ctx.log(f'ACN-Sim run done: {rid} ({len(errs)} chunk errors)')


# ---------------------------------------------------------------------------------------- summaries
def _prices(start: datetime, n: int, period: int, tariff: str):
    from acnportal.signals.tariffs import TimeOfUseTariff
    tou = TimeOfUseTariff(tariff)
    ts = pd.date_range(start, periods=n, freq=f'{period}min')
    key = pd.DataFrame({'m': ts.month, 'd': ts.day, 'h': ts.hour, 'mi': ts.minute})
    cache, price, dem = {}, np.empty(n), np.empty(n)
    for i, (m, d, h, mi, t) in enumerate(zip(key['m'], key['d'], key['h'], key['mi'], ts)):
        k = (m, d, t.weekday(), h, mi)
        if k not in cache:
            dt = t.to_pydatetime()
            cache[k] = (tou.get_tariff(dt), tou.get_demand_charge(dt))
        price[i], dem[i] = cache[k]
    return ts, price, dem


def _first_cap(fr, share, target=2.0):
    ok = share <= target
    if not ok.any():
        return np.nan
    k = int(np.argmax(ok))
    if k == 0:
        return float(fr[0])
    x0, x1, y0, y1 = fr[k - 1], fr[k], share[k - 1], share[k]
    return float(x0 + (x1 - x0) * (y0 - target) / (y0 - y1)) if y0 != y1 else float(fr[k])


def _capacity_contrasts(ctx: Ctx, res: pd.DataFrame, cap_draws: dict):
    """Paired differences in the capacity needed for <= 2 % undelivered energy between forecasts at the same site
    and scheduler (same bootstrap days), and the share of the perfect-information capacity value each forecast
    captures relative to first come first served without a forecast."""
    rows = []
    sites = sorted({k[0] for k in cap_draws})
    for site in sites:
        rate = res.loc[res['site'] == site, 'max_demand_charge_usd_per_kw_month'].max()
        get = lambda sch, fc, tau: cap_draws.get((site, sch, fc, tau))                            # noqa: E731
        fcfs, oracle = get('fcfs', 'none', 'na'), get('llf', 'oracle', 'na')
        for tau in ('0.5', '0.1'):
            pairs = [('cold', 'deployment'), ('cold', 'benchmark'), ('deployment', 'benchmark')]
            for a, b in pairs:
                A_, B_ = get('llf', a, tau), get('llf', b, tau)
                if A_ is None or B_ is None or not (A_[2] and B_[2]) or A_[3] != B_[3]:
                    continue
                d = A_[1] - B_[1]; est = A_[0] - B_[0]
                rows.append({'site': site, 'scheduler': 'llf', 'tau': tau, 'contrast': f'{a} - {b}', 'estimate_fraction': est,
                             'ci_low': np.nanpercentile(d, 2.5), 'ci_high': np.nanpercentile(d, 97.5),
                             'estimate_kw': est * rated_kw(site), 'demand_charge_rate_usd_per_kw_month': rate,
                             'indicative_annual_demand_charge_usd': est * rated_kw(site) * rate * 12 if np.isfinite(est) else np.nan})
            if fcfs is None or oracle is None or not (fcfs[2] and oracle[2]):
                continue
            for v in ('benchmark', 'deployment', 'cold'):
                V = get('llf', v, tau)
                if V is None or not V[2] or V[3] != fcfs[3]:
                    continue
                with np.errstate(invalid='ignore', divide='ignore'):
                    sh = 100 * (fcfs[1] - V[1]) / (fcfs[1] - oracle[1])
                est = 100 * (fcfs[0] - V[0]) / (fcfs[0] - oracle[0]) if np.isfinite(fcfs[0] - oracle[0]) and fcfs[0] != oracle[0] else np.nan
                rows.append({'site': site, 'scheduler': 'llf', 'tau': tau,
                             'contrast': f'share of the perfect-information capacity value captured by {v} (%)',
                             'estimate_fraction': est, 'ci_low': np.nanpercentile(sh, 2.5), 'ci_high': np.nanpercentile(sh, 97.5),
                             'estimate_kw': (fcfs[0] - V[0]) * rated_kw(site) if np.isfinite(est) else np.nan,
                             'demand_charge_rate_usd_per_kw_month': rate, 'indicative_annual_demand_charge_usd': np.nan})
    pd.DataFrame(rows).to_csv(ctx.p('acnsim_capacity_contrasts.csv'), index=False)


def summarise(ctx: Ctx, meta: dict, infos: dict):
    sc = ctx.cfg['acnsim']
    z = load_inputs(ctx).set_index('session_key')
    rdir = ctx.p('acnsim', 'runs', 'x').parent
    rows, per_day, price_cache = [], {}, {}
    for rid, m in meta.items():
        f = rdir / f'{rid}.parquet'
        if not f.exists():
            continue
        site, period = m['site'], int(m['period_min'])
        r = pd.read_parquet(f).set_index('session_key')
        g = z.loc[r.index]
        E = g['E'].to_numpy(float)
        Dt = (g['t1'] - g['t0']).dt.total_seconds().to_numpy() / 3600
        E_ach = np.minimum(E, PMAX * Dt)
        unmet = np.clip(E_ach - r['delivered_kwh'].to_numpy(float), 0, None)
        day = g['t0'].dt.strftime('%Y-%m-%d').to_numpy()
        per_day[rid] = pd.DataFrame({'day': day, 'unmet': unmet, 'E_ach': E_ach}).groupby('day').sum()
        pw = np.load(rdir / f'{rid}.power.npz')
        origin = pd.Timestamp(infos[site]['origin'])
        total_n = max(int(o) + len(pw[f'p{int(o)}']) for o in pw['offsets'])
        load = np.zeros(total_n)
        for o in pw['offsets']:
            p = pw[f'p{int(o)}']; load[int(o): int(o) + len(p)] += p
        ck = (site, period)
        if ck not in price_cache or len(price_cache[ck][0]) < total_n:
            price_cache[ck] = _prices(origin.to_pydatetime(), total_n + 7 * 24 * 60 // period, period, sc['tariff'])
        ts, price, dem = (x[:total_n] for x in price_cache[ck])
        energy_kwh = load * period / 60
        month = ts.to_period('M')
        mdf = pd.DataFrame({'month': month, 'load': load, 'dem': dem})
        active = mdf.groupby('month')['load'].transform('max') > 0
        demand = mdf[active].groupby('month').apply(lambda x: x['load'].max() * x['dem'].max()).sum()
        st = __import__('json').loads((rdir / f'{rid}.status.json').read_text())
        dayset = set(day)
        win_stats = {}
        for exp, (h0, h1) in _event_hours(sc).items():
            col = 'event_window' if exp == 'E4' else 'midday_window'
            win = (ts.weekday < 5) & (ts.hour >= h0) & (ts.hour < h1)
            wl = pd.Series(load[win], index=ts[win])
            wday = wl.groupby(wl.index.date).mean()
            wday = wday[[str(d) in dayset for d in wday.index]] if len(wday) else wday      # event days with sessions
            win_stats[f'{col}_mean_kw'] = float(wday.mean()) if len(wday) else np.nan
            win_stats[f'{col}_peak_kw'] = float(wl.max()) if len(wl) else np.nan
        rows.append({**m, 'run_id': rid, **win_stats, 'sessions': len(r), 'undelivered_share_pct': 100 * unmet.sum() / E_ach.sum(),
                     'sessions_short_over_1kwh_pct': 100 * np.mean(unmet > 1), 'peak_kw': float(load.max()),
                     'mean_daily_peak_kw': float(pd.Series(load, index=ts).resample('D').max().replace(0, np.nan).mean()),
                     'energy_kwh': float(energy_kwh.sum()), 'tou_energy_cost_usd': float((energy_kwh * price).sum()),
                     'tou_cost_per_kwh': float((energy_kwh * price).sum() / max(energy_kwh.sum(), 1e-9)),
                     'demand_charge_usd': float(demand), 'max_demand_charge_usd_per_kw_month': float(np.max(dem)) if len(dem) else np.nan,
                     'rated_kw': rated_kw(site), 'chunk_errors': st['errors'], 'chunks': st['chunks']})
    res = pd.DataFrame(rows)
    res.to_csv(ctx.p('acnsim_runs.csv'), index=False)
    # capacity needed (E2) with a day-block bootstrap, and the firm capacity on 95 % of days; the bootstrap days are
    # drawn once per site, so that contrasts between forecasts at a site are paired
    need, cap_draws, site_bs = [], {}, {}
    rng = np.random.default_rng(11)
    e2 = res[res['experiment'] == 'E2'].copy()
    e2['tau_s'] = e2['tau'].map(lambda v: 'na' if pd.isna(v) else f'{float(v):g}')
    for (site, sch, fcast, tau), g in e2.groupby(['site', 'scheduler', 'forecast', 'tau_s'], dropna=False):
        g = g.sort_values('cap_fraction')
        fr = g['cap_fraction'].to_numpy()
        days = sorted(set().union(*[set(per_day[r].index) for r in g['run_id']]))
        U = np.stack([per_day[r]['unmet'].reindex(days, fill_value=0).to_numpy() for r in g['run_id']])
        A = np.stack([per_day[r]['E_ach'].reindex(days, fill_value=0).to_numpy() for r in g['run_id']])
        share = 100 * U.sum(1) / A.sum(1)
        point = _first_cap(fr, share)
        if site not in site_bs or len(site_bs[site][0]) != len(days):
            site_bs[site] = (days, rng.integers(0, len(days), size=(ctx.n_boot, len(days))))
        bs = site_bs[site][1] if site_bs[site][0] == days else rng.integers(0, len(days), size=(ctx.n_boot, len(days)))
        draws = np.array([_first_cap(fr, 100 * U[:, b].sum(1) / A[:, b].sum(1)) for b in bs])
        cap_draws[(site, sch, fcast, tau)] = (point, draws, site_bs[site][0] == days, tuple(fr))
        dshare = 100 * U / np.where(A > 0, A, np.nan)                         # caps x days
        day_need = np.array([fr[np.argmax(dshare[:, i] <= 2.0)] if (dshare[:, i] <= 2.0).any() else np.inf
                             for i in range(len(days)) if np.isfinite(dshare[:, i]).any()])
        firm = float(np.quantile(day_need, 0.95)) if len(day_need) else np.nan
        need.append({'site': site, 'scheduler': sch, 'forecast': fcast, 'tau': tau, 'capacity_needed_fraction': point,
                     'ci_low': np.nanpercentile(draws, 2.5) if np.isfinite(draws).any() else np.nan,
                     'ci_high': np.nanpercentile(draws, 97.5) if np.isfinite(draws).any() else np.nan,
                     'capacity_needed_kw': point * rated_kw(site) if np.isfinite(point) else np.nan,
                     'firm_capacity_95pct_days_fraction': firm,
                     'committable_kw_vs_rating': (1 - firm) * rated_kw(site) if np.isfinite(firm) else np.nan,
                     'days': len(days), 'caps_tested': ';'.join(f'{x:g}' for x in fr),
                     'at_lowest_cap_tested': bool(np.isfinite(point) and point <= fr[0] + 1e-12),
                     'undelivered_pct_by_cap': ';'.join(f'{x:g}:{y:.2f}' for x, y in zip(fr, share))})
    pd.DataFrame(need).to_csv(ctx.p('acnsim_capacity_needed.csv'), index=False)
    _capacity_contrasts(ctx, res, cap_draws)
    # E4 and E5: dependable flexibility during a weekday event: the smallest event limit (fraction of the network's
    # constraint limits during the event) that keeps undelivered energy <= 2 % on at least 95 % of event days, and the
    # load reduction that limit commits against uncontrolled charging in the same window
    dep = []
    for exp, (h0, h1) in _event_hours(sc).items():
        col = 'event_window_mean_kw' if exp == 'E4' else 'midday_window_mean_kw'
        ee = res[res['experiment'] == exp].copy()
        if ee.empty:
            continue
        ee['tau_s'] = ee['tau'].map(lambda v: 'na' if pd.isna(v) else f'{float(v):g}')
        for site, gs in ee.groupby('site'):
            base = res[(res['site'] == site) & (res['scheduler'] == 'uncontrolled') & res['experiment'].isin([exp, 'E1'])]
            base = base.sort_values('experiment', key=lambda s: s.map({exp: 0, 'E1': 1}))
            base_kw = float(base[col].iloc[0]) if len(base) else np.nan
            for (sch, fcast, tau), g in gs[gs['scheduler'] != 'uncontrolled'].groupby(['scheduler', 'forecast', 'tau_s']):
                g = g.sort_values('cap_fraction')
                fr = g['cap_fraction'].to_numpy()
                days = sorted(set().union(*[set(per_day[r].index) for r in g['run_id']]))
                wk = np.array([pd.Timestamp(d).weekday() < 5 for d in days])
                U = np.stack([per_day[r]['unmet'].reindex(days, fill_value=0).to_numpy() for r in g['run_id']])[:, wk]
                A = np.stack([per_day[r]['E_ach'].reindex(days, fill_value=0).to_numpy() for r in g['run_id']])[:, wk]
                with np.errstate(invalid='ignore', divide='ignore'):
                    dshare = 100 * U / np.where(A > 0, A, np.nan)
                okdays = (dshare <= 2.0).mean(1)                                 # share of event days within 2 %
                firm = next((float(f) for f, o in zip(fr, okdays) if o >= 0.95), np.nan)
                # realised load in the event window under the firm limit, against uncontrolled charging in the same window
                sched_kw = float(g.loc[np.isclose(g['cap_fraction'], firm), col].iloc[0]) if np.isfinite(firm) else np.nan
                dep.append({'experiment': exp, 'site': site, 'scheduler': sch, 'forecast': fcast, 'tau': tau,
                            'event_hours': f'{h0}-{h1} weekdays',
                            'firm_event_limit_fraction': firm, 'firm_event_limit_kw': firm * rated_kw(site) if np.isfinite(firm) else np.nan,
                            'uncontrolled_event_mean_kw': base_kw, 'scheduled_event_mean_kw_at_firm_limit': sched_kw,
                            'dependable_reduction_kw': base_kw - sched_kw if np.isfinite(sched_kw) else np.nan,
                            'limits_tested': ';'.join(f'{f:g}' for f in fr),
                            'firm_limit_status': ('not reached within the limits tested' if not np.isfinite(firm) else
                                                  'at the lowest limit tested (true limit may be lower)' if (firm <= fr[0] + 1e-12 and fr[0] > 0) else 'bracketed'),
                            'share_of_event_days_within_2pct_by_limit': ';'.join(f'{f:g}:{100 * o:.1f}' for f, o in zip(fr, okdays)),
                            'undelivered_pct_by_limit': ';'.join(f'{f:g}:{100 * u / a:.2f}' for f, u, a in zip(fr, U.sum(1), A.sum(1))),
                            'event_days': int(wk.sum())})
    pd.DataFrame(dep).to_csv(ctx.p('acnsim_event_dependable_flexibility.csv'), index=False)
    ctx.log('ACN-Sim summary written: ' + ', '.join(f"{r['site']} {r['scheduler']} {r['forecast']} tau{r['tau']}: {r['capacity_needed_fraction']:.3f}"
                                               for r in need))
