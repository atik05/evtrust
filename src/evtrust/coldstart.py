"""Cross-fitted decomposition of the deployment gap for unseen drivers.

Every setting predicts the same rows, so all comparisons are paired:

  A  random split; a driver's later sessions may sit in training
  B  driver-temporal split; driver seen, but only their past
  C  unseen driver; their own past-only history is available (warm-up)
  D  unseen driver; history hidden (permanent cold start)
  E  random split with whole-dataset driver aggregates (published-style, leaky)

Decomposition on each driver's last 20 percent of sessions:
  future-session leakage  = B vs A
  unseen-driver cost      = C vs B
  value of own history    = D vs C
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

USER_HISTORY = ['user_prior_count', 'user_prior_mean_duration', 'user_prior_median_duration',
                'user_prior_mean_energy']
K_BINS = [(-0.5, 0.5, '0'), (0.5, 1.5, '1'), (1.5, 2.5, '2'), (2.5, 4.5, '3-4'),
          (4.5, 9.5, '5-9'), (9.5, 19.5, '10-19'), (19.5, 1e9, '20+')]


def k_bucket(k: np.ndarray) -> np.ndarray:
    out = np.empty(len(k), dtype=object)
    for lo, hi, name in K_BINS:
        out[(k > lo) & (k <= hi)] = name
    return out


def prepare(df: pd.DataFrame, target: str, end: str | None, min_sessions: int = 5) -> pd.DataFrame:
    d = df[df['user_id'].notna() & (df['user_id'].astype(str).str.strip() != '')].copy()
    d['t'] = pd.to_datetime(d['connection_time'], utc=True)
    if end:
        d = d[d['t'] < pd.Timestamp(end, tz='UTC')]
    d = d[d[target].notna()]
    d = d[d.groupby('user_id')['user_id'].transform('size') >= min_sessions]
    d = d.sort_values('t').reset_index(drop=True)
    d['station_code'] = d['station_id'].astype('category').cat.codes
    d['k'] = d['user_prior_count'].fillna(0).to_numpy(float)
    rank = d.groupby('user_id').cumcount()
    size = d.groupby('user_id')['user_id'].transform('size')
    d['is_T'] = (rank >= np.floor(0.8 * size)).to_numpy()
    return d


def published_style(d: pd.DataFrame) -> pd.DataFrame:
    """Driver aggregates over the whole file, including the row itself and the future."""
    L = d.copy()
    for col, src, agg in [('user_prior_mean_duration', 'duration_hours', 'mean'),
                          ('user_prior_median_duration', 'duration_hours', 'median'),
                          ('user_prior_mean_energy', 'kwh_delivered', 'mean')]:
        L[col] = L.groupby('user_id')[src].transform(agg)
    L['user_prior_count'] = L.groupby('user_id')['user_id'].transform('size')
    return L


def make_learner(name: str, numeric: list[str], seed: int, cfg: dict):
    """Return fit(X, y) -> object with predict(X)."""
    if name == 'hgb':
        from sklearn.ensemble import HistGradientBoostingRegressor

        def fit(X, y):
            return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                                 categorical_features=[X.columns.get_loc('station_code')],
                                                 random_state=seed).fit(X, y)
        return fit
    if name == 'catboost':
        from catboost import CatBoostRegressor
        p = dict(cfg.get('catboost', {}))
        p.setdefault('iterations', 1500); p.setdefault('depth', 8); p.setdefault('learning_rate', 0.05)
        p.setdefault('loss_function', 'MAE'); p.setdefault('verbose', 0)
        p['random_seed'] = seed

        def fit(X, y):
            Xc = X.copy(); Xc['station_code'] = Xc['station_code'].astype(str)
            m = CatBoostRegressor(**p).fit(Xc, y, cat_features=['station_code'])

            class _W:
                def predict(self, Z):
                    Zc = Z.copy(); Zc['station_code'] = Zc['station_code'].astype(str)
                    return m.predict(Zc)
            return _W()
        return fit
    if name == 'ft':
        from .ft_compact import FTRegressor
        p = dict(cfg.get('ft', {}))

        def fit(X, y):
            return FTRegressor(numeric, 'station_code', seed=seed, epochs=int(p.get('epochs', 40)),
                               batch=int(p.get('batch', 1024)), lr=float(p.get('lr', 1e-3)),
                               patience=int(p.get('patience', 6))).fit(X, y)
        return fit
    raise ValueError(name)


def run_unit(df: pd.DataFrame, target: str, learner: str, seed: int, numeric: list[str], cfg: dict,
             end: str | None, with_published: bool = True, n_folds: int = 5) -> pd.DataFrame:
    """One (target, learner, seed) unit. Returns row-level predictions for settings A to E."""
    d = prepare(df, target, end)
    fit = make_learner(learner, numeric, seed, cfg)
    cols = numeric + ['station_code']
    X = d[cols]
    y = d[target].to_numpy(float)
    T = d['is_T'].to_numpy()
    pred = {k: np.full(len(d), np.nan) for k in 'ABCDE'}
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(d):
        pred['A'][te] = fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    pred['B'][T] = fit(X[~T], y[~T]).predict(X[T])
    groups = d['user_id'].astype(str).to_numpy()
    uniq = np.unique(groups)
    perm = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq))))
    gid = np.array([perm[g] for g in groups])
    for tr, te in GroupKFold(n_folds).split(d, groups=gid):
        m = fit(X.iloc[tr], y[tr])
        pred['C'][te] = m.predict(X.iloc[te])
        Xc = X.iloc[te].copy(); Xc[[c for c in USER_HISTORY if c in Xc]] = np.nan
        pred['D'][te] = m.predict(Xc)
    if with_published:
        XL = published_style(d)[cols]
        for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(d):
            pred['E'][te] = fit(XL.iloc[tr], y[tr]).predict(XL.iloc[te])
    out = pd.DataFrame({'session_key': d['session_id'].astype(str).to_numpy(), 'user': groups, 'site': d['site'].to_numpy(), 'k': d['k'].to_numpy(), 'is_T': T, 'y': y,
                        'duration_true': d['duration_hours'].to_numpy(float),
                        'energy_true': d['kwh_delivered'].to_numpy(float),
                        'requested_duration': d.get('requested_duration_hours', pd.Series(np.nan, index=d.index)).to_numpy(float),
                        'requested_energy': d.get('requested_energy_kwh', pd.Series(np.nan, index=d.index)).to_numpy(float),
                        'session_index': np.arange(len(d))})
    for k, v in pred.items():
        out[f'pred_{k}'] = v
        out[f'err_{k}'] = np.abs(y - v)
    out['target'], out['learner'], out['seed'] = target, learner, seed
    return out


# ------------------------------------------------------------------ statistics

def cluster_ratio_ci(z: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0):
    """Relative change sum(b)/sum(a)-1 with a driver-cluster bootstrap CI and a two-sided p-value."""
    g = z.groupby('user').agg(ea=(a, 'sum'), eb=(b, 'sum'))
    ea, eb = g['ea'].to_numpy(), g['eb'].to_numpy()
    n = len(g)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = eb[idx].sum(1) / ea[idx].sum(1) - 1
    est = eb.sum() / ea.sum() - 1
    # finite-sample two-sided p-value, as in Methods: 2(k + 1)/(B + 1), k the smaller tail count (Davison and Hinkley, 1997)
    k = min(int((draws <= 0).sum()), int((draws >= 0).sum()))
    p = min(1.0, 2 * (k + 1) / (n_boot + 1))
    return 100 * est, 100 * np.quantile(draws, .025), 100 * np.quantile(draws, .975), p


def summarise(rows: pd.DataFrame, n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decomposition table and warm-up curve, averaged over seeds within each learner."""
    dec, warm = [], []
    for (target, learner), g in rows.groupby(['target', 'learner']):
        # average errors over seeds per row, so seeds do not count as independent evidence
        key = ['user', 'session_index', 'k', 'is_T']
        m = g.groupby(key, as_index=False)[[f'err_{s}' for s in 'ABCDE']].mean()
        T = m[m['is_T']]
        for a, b, comp in [('err_A', 'err_B', 'future-session leakage'), ('err_B', 'err_C', 'unseen-driver cost'),
                           ('err_C', 'err_D', 'value of own history'), ('err_A', 'err_D', 'total gap'),
                           ('err_A', 'err_E', 'published-style features')]:
            if T[b].isna().all():
                continue
            est, lo, hi, p = cluster_ratio_ci(T.dropna(subset=[a, b]), a, b, n_boot)
            dec.append({'target': target, 'learner': learner, 'component': comp, 'effect_pct': est,
                        'ci_low': lo, 'ci_high': hi, 'p_value': p, 'n_rows': int(len(T)),
                        'n_drivers': int(T['user'].nunique()), 'n_seeds': int(g['seed'].nunique())})
        m['k_bucket'] = k_bucket(m['k'].to_numpy())
        for _, _, b in K_BINS:
            z = m[m['k_bucket'] == b]
            if len(z) < 30:
                continue
            est, lo, hi, p = cluster_ratio_ci(z, 'err_A', 'err_C', n_boot)
            warm.append({'target': target, 'learner': learner, 'k_bucket': b, 'n_rows': int(len(z)),
                         'mae_random': z['err_A'].mean(), 'mae_warm': z['err_C'].mean(),
                         'mae_cold': z['err_D'].mean(), 'warm_vs_random_pct': est,
                         'ci_low': lo, 'ci_high': hi, 'p_value': p})
    dec = pd.DataFrame(dec)
    if not dec.empty:
        from statsmodels.stats.multitest import multipletests
        dec['p_holm'] = multipletests(dec['p_value'], method='holm')[1]
    return dec, pd.DataFrame(warm)
