"""Information-state calibration: a deployable scheduling rule with a shortfall guarantee for every class of driver.

At plug-in an operator observes the information state of a session: anonymous (no identity), first visit (identified,
no completed session) or returning (identified, with history). The paper's shifts calibrate departure and energy
separately, within populations that are defined retrospectively (drivers with at least five sessions before March 2020,
other identified drivers, anonymous sessions), and the union bound gives P(U > 0) <= 2 gamma within each population.
Here the latest start of the deadline-aware policy is calibrated directly, within each observable information state s:

    S*      = max(D - min(E, P D) / P, 0)       latest start that still delivers the achievable energy
    S_hat   = max(D_hat - E_hat / P, 0)         start implied by the point forecasts
    S_gamma = max(S_hat + q_s(gamma), 0)        q_s(gamma): gamma-quantile of S* - S_hat over the other calibration
                                                groups of state s (cross-conformal, as for the paper's shifts)

U > 0 exactly when S_gamma > S*, so P(U > 0 | s) <= gamma for sessions exchangeable with the calibration sessions of
state s (Proposition 2 of the paper), with one quantile instead of two. The rule needs only what is known at plug-in.

Calibrations compared on every session of the decomposition window (deploy forecasts, one learner and seed, as in the
ACN-Sim study):
  population, D and E   the paper: departure and energy shifted separately within the retrospective populations
  pooled, D and E       one set of shifts for all sessions (the simplest deployable rule)
  state, D and E        departure and energy shifted separately within information states (ablation)
  population, start     latest start calibrated within the retrospective populations (ablation)
  state, start          the proposed rule
Outputs
  infostate_states.csv        sessions per state and agreement with the driver's history count where available
  infostate_flex.csv          network flexibility at 2 % undelivered energy, and each state's flexibility, undelivered
                              energy and P(U > 0) at that network operating point; cluster bootstrap (drivers; days
                              for anonymous sessions), contrasts against the paper's calibration
  infostate_reliability.csv   P(U > 0) by information state against gamma, for every calibration
  infostate_sim.csv           (ACN, stage infostate_sim) the same comparison replayed in ACN-Sim at the real network
                              capacity with least-laxity-first scheduling, undelivered energy by state
"""
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .common import TARGET_UNMET, Ctx, boot_weights, cross_q_matrix, interp_target, shifted, write_json
from evtrust.impact import TAUS

STATES = ('anonymous', 'first visit', 'returning')
CALIBRATIONS = ('population, D and E', 'pooled, D and E', 'state, D and E', 'population, start', 'state, start')
PAPER, PROPOSED = 'population, D and E', 'state, start'


# ------------------------------------------------------------------------------------------------ inputs
def load(ctx: Ctx) -> pd.DataFrame:
    learner, seed = ctx.cfg['acnsim']['learner'], int(ctx.cfg['acnsim']['seed'])
    if ctx.smoke:
        seed = ctx.seeds[0]
    z = pd.read_parquet(ctx.p(f'deploy_forecasts_{learner}_seed{seed}.parquet'))
    z['state'] = information_state(z)
    s = ctx.sessions()
    s = s.assign(session_key=s['session_id'].astype(str)).drop_duplicates('session_key').set_index('session_key')
    key = z['session_key'].astype(str)
    user = s['user_id'].reindex(key).astype('string').to_numpy()
    day = pd.to_datetime(s['connection_time'].reindex(key), utc=True).dt.strftime('%Y-%m-%d').to_numpy()
    z['cluster'] = np.where(z['population'].to_numpy() == 'anonymous', 'd:' + pd.Series(day).astype(str).to_numpy(),
                            'u:' + pd.Series(user).astype(str).to_numpy())
    z['k'] = s['user_prior_count'].reindex(key).to_numpy(float) if 'user_prior_count' in s else np.nan
    return z.reset_index(drop=True)


def information_state(z: pd.DataFrame) -> np.ndarray:
    """Observable at plug-in: anonymous; first visit = identified with no driver history, which is exactly where the
    deployment forecast (own history) equals the forecast with history hidden; returning otherwise."""
    anon = z['population'].to_numpy() == 'anonymous'
    nohist = np.isclose(z['Dp_deployment'], z['Dp_cold']) & np.isclose(z['Ep_deployment'], z['Ep_cold'])
    return np.where(anon, 'anonymous', np.where(nohist, 'first visit', 'returning'))


# --------------------------------------------------------------------------------------------- the rule
def latest_start(D, E, P):
    return np.clip(D - np.minimum(E, P * D) / P, 0.0, None)


def start_shift(D, E, Dp, Ep, groups, strata, P, taus=TAUS):
    """Shifted start S_gamma[t, i] for every level, calibrated within each stratum (cross-conformal over groups)."""
    taus = np.asarray(taus, float)
    S_hat = np.clip(Dp - Ep / P, 0.0, None)
    r = latest_start(D, E, P) - S_hat
    q = np.zeros((len(taus), len(D)))
    for st in np.unique(strata):
        i = np.where(strata == st)[0]
        q[:, i] = cross_q_matrix(r[i], groups[i], taus)
    q[np.isclose(taus, 0.5)] = 0.0                    # 0.5 = the raw forecast, as for the paper's shifts
    return np.clip(S_hat[None] + q, 0.0, None)


def de_shift(D, E, Dp, Ep, groups, strata, taus=TAUS):
    """The paper's shifts of departure and energy, calibrated within each stratum; returns (D_hat, E_hat)."""
    Dh = np.zeros((len(taus), len(D))); Eh = np.zeros_like(Dh)
    for st in np.unique(strata):
        i = np.where(strata == st)[0]
        Dh[:, i], Eh[:, i] = shifted(D[i], E[i], Dp[i], Ep[i], groups[i], taus)
    return Dh, Eh


def start_of(calibration: str, z: pd.DataFrame, var: str, P: float, taus=TAUS):
    """Start of charging S[t, i] under a calibration, and for D-and-E calibrations the shifted (D_hat, E_hat)."""
    D, E = z['D'].to_numpy(float), z['E'].to_numpy(float)
    Dp, Ep = z[f'Dp_{var}'].to_numpy(float), z[f'Ep_{var}'].to_numpy(float)
    g = z['calib_group'].to_numpy()
    strata = {'population': z['population'].to_numpy(), 'pooled': np.zeros(len(z), int),
              'state': z['state'].to_numpy()}[calibration.split(',')[0]]
    if calibration.endswith('start'):
        return start_shift(D, E, Dp, Ep, g, strata, P, taus), None
    Dh, Eh = de_shift(D, E, Dp, Ep, g, strata, taus)
    return np.clip(Dh - Eh / P, 0.0, None), (Dh, Eh)


def outcome(D, E, S, P):
    """Deadline-aware charging from start S at constant power: undelivered energy U, deferral F, oracle deferral."""
    E_ach = np.minimum(E, P * D)
    delivered = np.minimum(E_ach, P * np.clip(D - S, 0.0, None))
    Fo = np.minimum(latest_start(D, E, P), D)
    return E_ach, E_ach - delivered, np.minimum(S, D), Fo


# ------------------------------------------------------------------------------------- session level
def _cluster_sums(z, U, F, E_ach, Fo, keys):
    """Per state: cluster sums of U, F, indicator U > 0 (levels x clusters) and of E_ach, Fo, sessions (clusters)."""
    cid = np.searchsorted(keys, z['cluster'].to_numpy())
    nc = len(keys); out = {}
    for st in STATES:
        m = z['state'].to_numpy() == st
        if not m.any():
            continue
        M = csr_matrix((np.ones(int(m.sum())), (np.arange(int(m.sum())), cid[m])), shape=(int(m.sum()), nc))
        agg = lambda X: np.asarray((M.T @ X.T).T)                                            # noqa: E731
        out[st] = (agg(U[:, m]), agg(F[:, m]), agg((U[:, m] > 1e-9).astype(float)), agg(E_ach[m][None])[0],
                   agg(Fo[m][None])[0], np.asarray(M.sum(0)).ravel())
    return out


def _read(sums, w):
    """Network flexibility at 2 % and each state's flexibility, undelivered share and P(U > 0) at that level."""
    U = sum(s[0] @ w for s in sums.values()); F = sum(s[1] @ w for s in sums.values())
    EA = sum(s[3] @ w for s in sums.values()); FO = sum(s[4] @ w for s in sums.values())
    unmet, flex = 100 * U / EA, 100 * F / FO
    v = {'network flex_pct': interp_target(unmet, flex)}
    if not (np.nanmin(unmet) <= TARGET_UNMET <= np.nanmax(unmet)):
        return v
    o = np.argsort(unmet, kind='stable')
    pos = float(np.interp(TARGET_UNMET, unmet[o], np.arange(len(unmet))[o]))
    idx = np.arange(len(unmet))
    for st, (Us, Fs, Is, EAs, FOs, Ns) in sums.items():
        v[f'{st} flex_pct'] = float(np.interp(pos, idx, 100 * (Fs @ w) / (FOs @ w)))
        v[f'{st} undelivered_pct'] = float(np.interp(pos, idx, 100 * (Us @ w) / (EAs @ w)))
        v[f'{st} P(U>0)_pct'] = float(np.interp(pos, idx, 100 * (Is @ w) / (Ns @ w)))
    v['gamma at network operating point'] = float(np.interp(pos, idx, np.asarray(TAUS)))
    return v


def run(ctx: Ctx):
    P = ctx.power
    z = load(ctx)
    states = z.groupby('state').size().rename('sessions').to_frame()
    if z['k'].notna().any():
        ident = z['state'] != 'anonymous'
        states['agreement_with_history_count_pct'] = np.nan
        agree = (z.loc[ident, 'state'] == 'first visit') == (z.loc[ident, 'k'].fillna(0) == 0)
        states.loc[['first visit', 'returning'], 'agreement_with_history_count_pct'] = 100 * float(agree.mean())
    states.to_csv(ctx.p('infostate_states.csv'))
    keys = np.array(sorted(set(z['cluster'])))
    strata = np.array([k.startswith('d:') for k in keys]).astype(int)
    D, E = z['D'].to_numpy(float), z['E'].to_numpy(float)
    sums, rel = {}, []
    taus_all = np.asarray(TAUS, float)
    chunks = np.array_split(np.arange(len(taus_all)), max(1, int(np.ceil(len(z) * len(taus_all) / 4e6))))
    for var in ('deployment', 'cold', 'benchmark'):
        for cal in CALIBRATIONS:
            parts = []
            for ch in chunks:                       # levels in chunks, so that large networks fit in memory
                S, _ = start_of(cal, z, var, P, taus_all[ch])
                E_ach, U, F, Fo = outcome(D, E, S, P)
                del S
                parts.append(_cluster_sums(z, U, F, E_ach, Fo, keys))
                for st in STATES:
                    m = z['state'].to_numpy() == st
                    for t, tau in zip(range(len(ch)), taus_all[ch]):
                        if any(np.isclose(tau, g) for g in (0.2, 0.1, 0.05, 0.02, 0.01)):
                            rel.append({'forecast': var, 'calibration': cal, 'state': st, 'gamma': float(tau), 'sessions': int(m.sum()),
                                        'P(U>0)_pct': 100 * float(np.mean(U[t, m] > 1e-9)),
                                        'undelivered_pct': 100 * U[t, m].sum() / E_ach[m].sum()})
                del U, F
            sums[(var, cal)] = {st: tuple(np.vstack([p[st][j] for p in parts]) if j < 3 else parts[0][st][j] for j in range(6))
                                for st in parts[0]}
        ctx.log(f'infostate {var} done')
    pd.DataFrame(rel).to_csv(ctx.p('infostate_reliability.csv'), index=False)
    # cluster bootstrap, identified drivers and anonymous days resampled within their stratum
    est = {k: _read(s, np.ones(len(keys))) for k, s in sums.items()}
    rng = np.random.default_rng(20261124)
    draws = {k: [] for k in sums}
    for _ in range(ctx.n_boot):
        w = boot_weights(rng, len(keys), strata)
        for k, s in sums.items():
            draws[k].append(_read(s, w))
    rows = []
    for (var, cal), e in est.items():
        d = pd.DataFrame(draws[(var, cal)])
        for q, val in e.items():
            rows.append({'forecast': var, 'calibration': cal, 'quantity': q, 'estimate': val,
                         'ci_low': np.nanpercentile(d[q], 2.5) if q in d else np.nan,
                         'ci_high': np.nanpercentile(d[q], 97.5) if q in d else np.nan})
        if cal != PAPER:
            base = pd.DataFrame(draws[(var, PAPER)])
            for q in ('network flex_pct', 'first visit undelivered_pct', 'first visit P(U>0)_pct', 'first visit flex_pct',
                      'anonymous undelivered_pct', 'returning flex_pct'):
                if q in e and q in est[(var, PAPER)]:
                    r = {'forecast': var, 'calibration': cal, 'quantity': f'{q} minus paper (pp)',
                         'estimate': e[q] - est[(var, PAPER)][q], 'ci_low': np.nan, 'ci_high': np.nan, 'p_two_sided': np.nan}
                    if q in d and q in base:
                        diff = d[q] - base[q]
                        r.update(ci_low=np.nanpercentile(diff, 2.5), ci_high=np.nanpercentile(diff, 97.5),
                                 p_two_sided=min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))
                    rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(ctx.p('infostate_flex.csv'), index=False)
    pick = out[(out['forecast'] == 'deployment') & (out['quantity'] == 'network flex_pct')]
    ctx.log('infostate network flexibility (deployment): ' + ', '.join(f"{c}={v:.1f}" for c, v in zip(pick['calibration'], pick['estimate'])))


# ------------------------------------------------------------------------------- ACN-Sim at real capacity
def sim_forecasts(z: pd.DataFrame, cal: str, gamma: float, P: float):
    """(D_hat, E_hat) passed to the scheduler. For start calibration, E_hat is the raw energy forecast and D_hat is
    placed so that the scheduler's laxity D_hat - E_hat / P equals the calibrated start."""
    t = int(np.argmin(np.abs(np.asarray(TAUS) - gamma)))
    S, de = start_of(cal, z, 'deployment', P)
    if de is not None:
        return de[0][t], de[1][t]
    Eh = np.clip(z['Ep_deployment'].to_numpy(float), 0.05, None)
    return S[t] + Eh / P, Eh


def run_sim(ctx: Ctx):
    from . import acnsim as A
    sc = ctx.cfg['acnsim']; P = ctx.power
    period = int(sc['period_minutes'])
    gammas = list(ctx.cfg.get('infostate', {}).get('sim_gammas', [0.1, 0.05]))
    cals = list(ctx.cfg.get('infostate', {}).get('sim_calibrations', [PAPER, PROPOSED]))
    workers = sc.get('workers', 'auto')
    workers = max(1, (os.cpu_count() or 4) - 4) if workers == 'auto' else int(workers)
    if ctx.smoke:
        gammas, workers = gammas[:1], min(workers, 2)
    z = load(ctx)
    zs = A.load_inputs(ctx)[['session_key', 'site', 'station_id', 't0', 't1']]
    z = z.merge(zs, on='session_key', how='inner')
    rdir = ctx.p('acnsim', 'runs', 'x').parent
    tasks, meta = [], {}
    for site in sc['sites']:
        if site not in set(z['site']):
            continue
        g, info = A.site_sessions(z, site, period, set(A.make_network(site, 1.0).station_ids), ctx.log)
        weeks = sorted(g['week'].unique())
        if ctx.smoke:
            weeks = weeks[len(weeks) // 2: len(weeks) // 2 + 1]
        pos = {k: i for i, k in enumerate(z['session_key'])}
        gi = np.array([pos[k] for k in g['session_key']])
        for cal in cals:
            for gam in gammas:
                rid = f"{site}__E1__llf__cap1.00__infostate-{cal.replace(', ', '-').replace(' ', '_')}__tau{gam}"
                meta[rid] = {'site': site, 'calibration': cal, 'gamma': gam}
                if (rdir / f'{rid}.parquet').exists():
                    continue
                Dh, Eh = sim_forecasts(z, cal, gam, P)
                tasks.append({'rid': rid, 'site': site, 'g': g, 'Dh': Dh[gi], 'Eh': Eh[gi], 'weeks': weeks})
    ctx.log(f'infostate ACN-Sim: {len(tasks)} runs to do, {workers} workers')

    def gen():
        for t in tasks:
            g = t['g']
            for w in t['weeks']:
                sel = (g['week'] == w).to_numpy(); gw = g[sel]
                de = gw['a'].to_numpy() + np.ceil(np.nan_to_num(t['Dh'][sel], nan=24.0) * 60 / period).astype(int)
                evs = list(zip(gw['session_key'], gw['station_id'].astype(str), gw['a'].astype(int), gw['d'].astype(int),
                               np.maximum(de, gw['a'].to_numpy() + 1).astype(int), gw['E'].to_numpy(float),
                               np.nan_to_num(t['Eh'][sel], nan=10.0)))
                yield {'run': t['rid'], 'chunk': int(w), 'site': t['site'], 'cap': 1.0, 'scheduler': 'llf', 'period': period,
                       'event': None, 'start': gw['wstart'].iloc[0].to_pydatetime(), 'offset': int(w) * 7 * 24 * 60 // period,
                       'evs': evs, 'tariff': sc['tariff']}

    pending = {t['rid']: len(t['weeks']) for t in tasks}
    results = {t['rid']: [] for t in tasks}
    it = gen()
    with ProcessPoolExecutor(max_workers=workers, initializer=A._single_thread) as ex:
        inflight = set()
        while True:
            while len(inflight) < 3 * workers:
                try:
                    inflight.add(ex.submit(A.simulate_chunk, next(it)))
                except StopIteration:
                    break
            if not inflight:
                break
            fin, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in fin:
                res = fut.result()
                results[res['run']].append(res); pending[res['run']] -= 1
                if pending[res['run']] == 0:
                    A._save_run(ctx, res['run'], results.pop(res['run']), rdir, meta)
    if any(v != 0 for v in pending.values()):
        raise RuntimeError('infostate ACN-Sim: unfinished runs')
    # undelivered energy by information state, against the achievable energy of each replayed session
    zz = z.set_index('session_key')
    rows = []
    for rid, m in meta.items():
        r = pd.read_parquet(rdir / f'{rid}.parquet').set_index('session_key')
        g = zz.loc[r.index]
        Dt = (g['t1'] - g['t0']).dt.total_seconds().to_numpy() / 3600
        E_ach = np.minimum(g['E'].to_numpy(float), A.PMAX * Dt)
        unmet = np.clip(E_ach - r['delivered_kwh'].to_numpy(float), 0, None)
        st = g['state'].to_numpy()
        row = {**m, 'sessions': len(r), 'undelivered_pct': 100 * unmet.sum() / E_ach.sum(),
               'P(short>0.1kWh)_pct': 100 * float(np.mean(unmet > 0.1))}
        for s in STATES:
            k = st == s
            row[f'{s} sessions'] = int(k.sum())
            row[f'{s} undelivered_pct'] = 100 * unmet[k].sum() / max(E_ach[k].sum(), 1e-9)
            row[f'{s} P(short>0.1kWh)_pct'] = 100 * float(np.mean(unmet[k] > 0.1)) if k.any() else np.nan
        st_ = __import__('json').loads((rdir / f'{rid}.status.json').read_text())
        row['chunk_errors'] = st_['errors']
        rows.append(row)
    pd.DataFrame(rows).to_csv(ctx.p('infostate_sim.csv'), index=False)
    ctx.log('infostate ACN-Sim summary written')
