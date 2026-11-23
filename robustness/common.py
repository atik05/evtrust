"""Shared context and helpers for the robustness analyses and the ACN-Sim study.

Calibration and evaluation follow the paper exactly: departure and energy forecasts are shifted by
cross-conformal residual quantiles from the other calibration groups (crc32 of the driver id modulo 5),
on the 100-level quantile grid of evtrust.impact, and flexibility is read at 2 % undelivered energy by
linear interpolation along each seed's trade-off curve, then averaged over seeds.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evtrust.coldstart import USER_HISTORY, make_learner, prepare
from evtrust.impact import TAUS

TARGETS = ('kwh_delivered', 'duration_hours')
PKG = Path(__file__).resolve().parents[1]
TARGET_UNMET = 2.0


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg_path: str, network: str, smoke: bool = False):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.net, self.smoke = network, smoke
        self.ncfg = self.cfg['networks'][network]
        root = Path(self.cfg.get('output_dir', 'out'))
        root = root if root.is_absolute() else PKG / root
        self.out = root / ('smoke' if smoke else 'full') / network
        (self.out / 'shards').mkdir(parents=True, exist_ok=True)
        self.seeds = list(self.cfg['seeds'])[:1] if smoke else list(self.cfg['seeds'])
        self.end = self.ncfg.get('decomposition_end', self.cfg['decomposition_end'])
        self.numeric = list(self.cfg['features']['numeric'])
        self.power = float(self.cfg['charging_power_kw'])
        self.n_boot = 40 if smoke else int(self.cfg['n_boot'])
        self._sessions = None
        self._shards = {}

    # paths and logging
    def p(self, *parts) -> Path:
        path = self.out.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, msg: str):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] [{self.net}] {msg}'
        print(line, flush=True)
        with open(self.p('robustness.log'), 'a') as f:
            f.write(line + '\n')

    # learners
    def learner_cfg(self, loss: str | None = None) -> dict:
        cat = dict(self.cfg['catboost'])
        if os.environ.get('EVTRUST_CPU') == '1' or platform.system() == 'Darwin':
            cat['task_type'] = 'CPU'; cat.pop('devices', None)
        if loss:
            cat['loss_function'] = loss
        ft = dict(self.cfg['ft'])
        if self.smoke:
            cat['iterations'] = 40; ft['epochs'] = 1; ft['patience'] = 1
        return {'catboost': cat, 'ft': ft}

    def fitter(self, learner: str, seed: int, loss: str | None = None, numeric=None):
        return make_learner(learner, list(numeric or self.numeric), seed, self.learner_cfg(loss))

    # data
    def sessions(self) -> pd.DataFrame:
        if self._sessions is None:
            src = Path(self.ncfg['sessions'])
            df = pd.read_parquet(src if src.is_absolute() else PKG / src)
            df['user_id'] = df['user_id'].astype('string')
            if self.smoke:
                df = smoke_subset(df)
            self._sessions = df
        return self._sessions

    def shard_dir(self) -> Path:
        d = Path(self.ncfg['shards'])
        return d if d.is_absolute() else PKG / d

    def shards(self, learner: str, target: str) -> pd.DataFrame:
        key = (learner, target)
        if key not in self._shards:
            files = sorted(glob.glob(str(self.shard_dir() / f'cs_{learner}_{target}_*.parquet')))
            if not files:
                raise FileNotFoundError(f'no shards cs_{learner}_{target}_* in {self.shard_dir()}')
            r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if 'y' not in r.columns:
                r = self._restore_labels(r, target)
            r['user'] = r['user'].astype(str)
            r = r[r['seed'].isin(self.seeds)]
            if self.smoke:
                keep = set(self.sessions()['session_id'].astype(str))
                r = r[r['session_key'].astype(str).isin(keep)]
            self._shards[key] = r.reset_index(drop=True)
        return self._shards[key]

    def _restore_labels(self, r: pd.DataFrame, target: str) -> pd.DataFrame:
        """Shards distributed without session data carry predictions only; labels, requests and absolute errors
        are restored from the network's sessions file, from which the shards were built."""
        s = self.sessions() if not self.smoke else self._full_sessions()
        s = s.assign(session_key=s['session_id'].astype(str)).drop_duplicates('session_key').set_index('session_key')
        key = r['session_key'].astype(str)
        missing = ~key.isin(s.index)
        if missing.any():
            raise RuntimeError(f'{int(missing.sum())} shard sessions are not in the sessions file: wrong data version')
        get = lambda col: s[col].reindex(key).to_numpy(float) if col in s else np.full(len(r), np.nan)  # noqa: E731
        r = r.copy()
        r['duration_true'], r['energy_true'] = get('duration_hours'), get('kwh_delivered')
        r['requested_duration'], r['requested_energy'] = get('requested_duration_hours'), get('requested_energy_kwh')
        r['y'] = r['energy_true'] if target == 'kwh_delivered' else r['duration_true']
        preds = [c for c in r.columns if c.startswith('pred_')]
        for c in preds:
            r['err_' + c[5:]] = np.abs(r['y'] - r[c])
        head = ['session_key', 'user', 'site', 'k', 'is_T', 'y', 'duration_true', 'energy_true', 'requested_duration',
                'requested_energy', 'session_index']
        order = head + [x for c in preds for x in (c, 'err_' + c[5:])]
        return r[order + [c for c in r.columns if c not in order]]

    def _full_sessions(self) -> pd.DataFrame:
        src = Path(self.ncfg['sessions'])
        df = pd.read_parquet(src if src.is_absolute() else PKG / src)
        df['user_id'] = df['user_id'].astype('string')
        return df

    def joined(self, learner: str, settings='ABCD') -> dict[int, pd.DataFrame]:
        """Per seed: one row per session with duration forecasts (pred_X) and energy forecasts (pred_X_e)."""
        dur, eng = self.shards(learner, 'duration_hours'), self.shards(learner, 'kwh_delivered')
        out = {}
        for seed in sorted(set(dur['seed']) & set(eng['seed'])):
            a = dur[dur['seed'] == seed].set_index('session_key')
            b = eng[eng['seed'] == seed].set_index('session_key')
            cols = [f'pred_{s}' for s in settings if f'pred_{s}' in b and b[f'pred_{s}'].notna().any()]
            out[seed] = a.join(b[cols], rsuffix='_e', how='inner')
        return out

    def cohort(self, target: str) -> pd.DataFrame:
        """Exactly the rows of the cold-start decomposition (evtrust.coldstart.prepare)."""
        return prepare(self.sessions(), target, self.end)

    def window(self, target: str) -> pd.DataFrame:
        """All sessions of the decomposition window with the target, identified or not."""
        d = self.sessions().copy()
        d['t'] = pd.to_datetime(d['connection_time'], utc=True)
        if self.end:
            d = d[d['t'] < pd.Timestamp(self.end, tz='UTC')]
        return d[d[target].notna()].sort_values('t').reset_index(drop=True)

    def done(self, stage: str) -> Path:
        return self.p('.done', stage)


def smoke_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Small, deterministic subset for the preflight smoke run: whole drivers and whole anonymous days."""
    uid = df['user_id'].astype(str)
    ident = df['user_id'].notna() & (uid.str.strip() != '')
    users = np.array(sorted(uid[ident].unique()))
    keep_u = set(users[[zlib.crc32(u.encode()) % 12 == 0 for u in users]])
    day = pd.to_datetime(df['connection_time'], utc=True).dt.strftime('%Y-%m-%d')
    keep = (ident & uid.isin(keep_u)) | (~ident & day.map(lambda d: zlib.crc32(d.encode()) % 12 == 0))
    return df[keep].reset_index(drop=True)


def stable_group(values, n: int = 5) -> np.ndarray:
    return np.array([zlib.crc32(str(v).encode()) % n for v in values])


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1, default=str))


# ------------------------------------------------------------------------------------- trade-off curves
def cross_q_matrix(resid: np.ndarray, groups: np.ndarray, taus) -> np.ndarray:
    """Quantiles of the residuals of the OTHER groups, for every tau: shape (len(taus), n)."""
    taus = np.asarray(taus, float)
    out = np.zeros((len(taus), len(resid)))
    fin = np.isfinite(resid)
    for g in np.unique(groups):
        sel = groups == g
        other = resid[~sel & fin]
        if len(other):
            out[:, sel] = np.quantile(other, taus)[:, None]
    return out


def shifted(D, E, Dp, Ep, groups, taus=TAUS):
    """Shifted departure and energy forecasts for every tau (0.5 = raw point forecast), as in evtrust.impact."""
    taus = np.asarray(taus, float)
    qd = cross_q_matrix(D - Dp, groups, taus)
    qe = cross_q_matrix(E - Ep, groups, 1 - taus)
    raw = np.isclose(taus, 0.5)
    qd[raw] = 0.0; qe[raw] = 0.0
    return np.maximum(Dp[None] + qd, 0.0), np.maximum(Ep[None] + qe, 0.0)


def alap(D, E, Dh, Eh, P):
    """Deadline-aware as-late-as-possible charging; broadcasts over a leading tau axis. P may be per session."""
    E_ach = np.minimum(E, P * D)
    s = np.clip(Dh - Eh / P, 0.0, None)
    delivered = np.minimum(E_ach, P * np.clip(D - s, 0.0, None))
    return E_ach, E_ach - delivered, np.minimum(s, D)


def curves(D, E, Dp, Ep, groups, P, taus=TAUS):
    """Per-session undelivered energy U[t, i] and deferral F[t, i], achievable energy and oracle deferral."""
    Dh, Eh = shifted(D, E, Dp, Ep, groups, taus)
    _, U, F = alap(D, E, Dh, Eh, P)
    E_ach = np.minimum(E, P * D)
    Fo = np.minimum(np.clip(D - E_ach / P, 0.0, None), D)        # oracle: knows D and E
    return U, E_ach, F, Fo


def interp_target(unmet: np.ndarray, flex: np.ndarray, target: float = TARGET_UNMET) -> float:
    o = np.argsort(unmet, kind='stable')
    if not (np.nanmin(unmet) <= target <= np.nanmax(unmet)):
        return np.nan
    return float(np.interp(target, unmet[o], flex[o]))


class CurveSet:
    """Trade-off curves of several seeds, summed within clusters (drivers or days) for a cluster bootstrap.

    Calls to add() with the same key (seed) are summed, so populations calibrated separately can be pooled
    at the same quantile level."""

    def __init__(self, n_clusters: int):
        self.nc = n_clusters
        self.items = {}

    def add(self, U, EA, F, Fo, cid, mask=None, key=None):
        m = np.ones(U.shape[1], bool) if mask is None else np.asarray(mask, bool)
        if not m.any():
            return
        c = cid[m]
        agg = lambda M: np.stack([np.bincount(c, weights=row[m], minlength=self.nc) for row in M])
        vec = lambda v: np.bincount(c, weights=v[m], minlength=self.nc)
        item = (agg(U), vec(EA), agg(F), vec(Fo), np.bincount(c, minlength=self.nc).astype(float))
        key = len(self.items) if key is None else key
        self.items[key] = tuple(a + b for a, b in zip(self.items[key], item)) if key in self.items else item

    def curve(self, w=None):
        res = []
        for Uc, EAc, Fc, Foc, Nc in self.items.values():
            ww = np.ones(self.nc) if w is None else w
            res.append((100 * (Uc @ ww) / (EAc @ ww), 100 * (Fc @ ww) / (Foc @ ww), (Fc @ ww) / (Nc @ ww)))
        return res

    def value(self, w=None, target=TARGET_UNMET, hours=False):
        if not self.items:
            return np.nan
        vals = []
        for unmet, flex, flex_h in self.curve(w):
            vals.append(interp_target(unmet, flex_h if hours else flex, target))
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

    def raw_unmet(self, w=None):
        """Undelivered share of the raw point forecasts (tau = 0.5, first grid level)."""
        return float(np.nanmean([u[0] for u, _, _ in self.curve(w)])) if self.items else np.nan

    def seed_sd(self, target=TARGET_UNMET, hours=False):
        v = [interp_target(u, fh if hours else f, target) for u, f, fh in self.curve()]
        return float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan


def boot_weights(rng, nc, strata=None):
    if strata is None:
        return np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)
    w = np.zeros(nc)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        np.add.at(w, idx[rng.integers(0, len(idx), len(idx))], 1.0)
    return w


def bootstrap_table(sets: dict, nc: int, B: int, seed: int = 20261117, contrasts: dict | None = None,
                    strata=None, hours_sets=()) -> pd.DataFrame:
    """Point estimates and percentile CIs of flexibility at 2 % undelivered energy (and contrasts)."""
    rng = np.random.default_rng(seed)
    contrasts = contrasts or {}
    point = {k: (s.value(hours=k in hours_sets) if not callable(s) else None) for k, s in sets.items()}
    cpt = {k: f(point) for k, f in contrasts.items()}
    draws = {k: [] for k in sets}; cdraws = {k: [] for k in contrasts}
    for _ in range(B):
        w = boot_weights(rng, nc, strata)
        v = {k: s.value(w, hours=k in hours_sets) for k, s in sets.items()}
        for k in sets:
            draws[k].append(v[k])
        for k, f in contrasts.items():
            cdraws[k].append(f(v))
    rows = []
    for k, s in sets.items():
        x = np.array(draws[k], float)
        rows.append({'quantity': k, 'kind': 'hours' if k in hours_sets else 'flex_pct_of_oracle',
                     'estimate': point[k], 'ci_low': np.nanpercentile(x, 2.5) if np.isfinite(x).any() else np.nan,
                     'ci_high': np.nanpercentile(x, 97.5) if np.isfinite(x).any() else np.nan,
                     'seed_sd': s.seed_sd(hours=k in hours_sets), 'raw_unmet_pct': s.raw_unmet(), 'p_boot': np.nan})
    for k in contrasts:
        x = np.array(cdraws[k], float); x = x[np.isfinite(x)]
        p = max(2 * min(np.mean(x <= 0), np.mean(x >= 0)), 1 / (B + 1)) if len(x) else np.nan
        rows.append({'quantity': k, 'kind': 'contrast', 'estimate': cpt[k],
                     'ci_low': np.percentile(x, 2.5) if len(x) else np.nan,
                     'ci_high': np.percentile(x, 97.5) if len(x) else np.nan,
                     'seed_sd': np.nan, 'raw_unmet_pct': np.nan, 'p_boot': p})
    return pd.DataFrame(rows)


def cluster_ratio(err_a: np.ndarray, err_b: np.ndarray, cluster: np.ndarray, B: int, seed: int = 0):
    """sum(b)/sum(a) - 1 in percent with a cluster bootstrap CI and two-sided p (as evtrust.coldstart)."""
    uniq, cid = np.unique(cluster, return_inverse=True)
    ea = np.bincount(cid, weights=err_a); eb = np.bincount(cid, weights=err_b)
    n = len(uniq); rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    draws = eb[idx].sum(1) / ea[idx].sum(1) - 1
    est = eb.sum() / ea.sum() - 1
    k = min(int((draws <= 0).sum()), int((draws >= 0).sum()))
    return 100 * est, 100 * np.quantile(draws, .025), 100 * np.quantile(draws, .975), min(1.0, 2 * (k + 1) / (B + 1))


def mask_history(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    Z[[c for c in USER_HISTORY if c in Z]] = np.nan
    return Z
