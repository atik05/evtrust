"""Repeated cross-validated benchmark of established and current tabular learners.

Protocols (folds x repeats):
  random          RepeatedKFold, 5 x 5, all sessions
  driver_grouped  GroupKFold on drivers, 5 folds x 5 permutations; test drivers are unseen but
                  their own past-only history is available, as at deployment
  chronological   rolling origin, 5 folds x 5 seeds; training labels must mature before the
                  test block starts (sessions still plugged in at the boundary are purged)
  site_transfer   leave one site out x 5 seeds, online setting; station identity and station history,
                  which cannot exist at an unseen site, are removed for every learner, while a driver's
                  own completed sessions at the new site accumulate as history
  site_transfer_strict   source-only transfer; user history is removed as well
  chronological_prepandemic   rolling origin restricted to sessions before March 2020

Learners use literature-standard configurations with early stopping on an inner validation
split where the library supports it, and an absolute-error objective wherever available, so
that every model optimises the reported metric. Hyperparameters are not tuned per fold.

Statistics: per-fold scores; Nadeau and Bengio corrected resampled t-test against the best
model with Holm correction; Friedman test with average ranks and the Nemenyi critical
difference for a critical-difference diagram.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

TARGET_HISTORY = {'kwh_delivered': 'user_prior_mean_energy', 'duration_hours': 'user_prior_median_duration'}
STATION_HISTORY = ['station_prior_count', 'station_prior_mean_duration', 'station_prior_mean_energy']
USER_HISTORY = ['user_prior_count', 'user_prior_mean_duration', 'user_prior_median_duration', 'user_prior_mean_energy']
PANDEMIC_BREAK = '2020-03-01'
DEFAULT_MODELS = ['mean', 'user_history', 'ridge', 'random_forest', 'extra_trees', 'hgb',
                  'lightgbm', 'xgboost', 'catboost', 'mlp', 'ft', 'tabpfn']
DISPLAY = {'mean': 'Historical mean', 'user_history': 'User history', 'ridge': 'Ridge',
           'random_forest': 'Random forest', 'extra_trees': 'Extra trees', 'hgb': 'HistGB',
           'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'catboost': 'CatBoost', 'mlp': 'MLP',
           'ft': 'FT-Transformer', 'tabpfn': 'TabPFN v2'}


# ----------------------------------------------------------------------------- learners

def _inner_split(n, seed, frac=0.1):
    idx = np.random.default_rng(seed).permutation(n)
    k = max(1, int(frac * n))
    return idx[k:], idx[:k]


def _sk_pipeline(est, one_hot_station: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def build(num):
        num_t = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler())
        if one_hot_station:
            ct = ColumnTransformer([('num', num_t, num),
                                    ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20), ['station_code'])])
        else:
            ct = ColumnTransformer([('num', num_t, num), ('cat', 'passthrough', ['station_code'])])
        return make_pipeline(ct, est)
    return build


def make_model(name: str, numeric: list[str], target: str, seed: int, cfg: dict):
    """Return fit(X, y) -> predictor with .predict(X). X contains numeric columns and station_code."""
    bcfg = cfg.get('benchmark_models', {})
    gpu = bool(cfg.get('gpu', False))

    if name == 'mean':
        def fit(X, y):
            m = float(np.mean(y))
            return type('M', (), {'predict': lambda self, Z: np.full(len(Z), m)})()
        return fit

    if name == 'user_history':
        col = TARGET_HISTORY[target]
        def fit(X, y):
            m = float(np.mean(y))
            return type('U', (), {'predict': lambda self, Z: (Z[col].fillna(m).to_numpy(float) if col in Z
                                                               else np.full(len(Z), m))})()
        return fit

    if name == 'ridge':
        from sklearn.linear_model import RidgeCV
        build = _sk_pipeline(RidgeCV(alphas=np.logspace(-3, 3, 13)), one_hot_station=True)
        return lambda X, y: build(numeric).fit(X, y)

    if name in ('random_forest', 'extra_trees'):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        Est = RandomForestRegressor if name == 'random_forest' else ExtraTreesRegressor
        p = dict(n_estimators=500, max_features=0.5, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        p.update(bcfg.get(name, {}))
        build = _sk_pipeline(Est(**p), one_hot_station=False)
        return lambda X, y: build(numeric).fit(X, y)

    if name == 'hgb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        def fit(X, y):
            return HistGradientBoostingRegressor(loss='absolute_error', max_iter=1500, learning_rate=0.05,
                                                 max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1,
                                                 n_iter_no_change=50, random_state=seed,
                                                 categorical_features=[X.columns.get_loc('station_code')]).fit(X, y)
        return fit

    if name == 'lightgbm':
        import lightgbm as lgb
        def fit(X, y):
            tr, va = _inner_split(len(y), seed)
            Xc = X.copy(); Xc['station_code'] = Xc['station_code'].astype('category')
            m = lgb.LGBMRegressor(objective='l1', n_estimators=3000, learning_rate=0.03, num_leaves=63,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                                  random_state=seed, verbose=-1, **bcfg.get('lightgbm', {}))
            m.fit(Xc.iloc[tr], y[tr], eval_set=[(Xc.iloc[va], y[va])], eval_metric='l1',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            cats = Xc['station_code'].cat.categories
            class P:
                def predict(self, Z):
                    Zc = Z.copy(); Zc['station_code'] = pd.Categorical(Zc['station_code'], categories=cats)
                    return m.predict(Zc)
            return P()
        return fit

    if name == 'xgboost':
        import xgboost as xgb
        def fit(X, y):
            tr, va = _inner_split(len(y), seed)
            Xc = X.copy(); Xc['station_code'] = Xc['station_code'].astype('category')
            cats = Xc['station_code'].cat.categories
            m = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=3000, learning_rate=0.03,
                                 max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 tree_method='hist', device='cuda' if gpu else 'cpu', enable_categorical=True,
                                 early_stopping_rounds=100, random_state=seed, **bcfg.get('xgboost', {}))
            m.fit(Xc.iloc[tr], y[tr], eval_set=[(Xc.iloc[va], y[va])], verbose=False)
            class P:
                def predict(self, Z):
                    Zc = Z.copy(); Zc['station_code'] = pd.Categorical(Zc['station_code'], categories=cats)
                    return m.predict(Zc)
            return P()
        return fit

    if name == 'catboost':
        from catboost import CatBoostRegressor
        p = {k: v for k, v in cfg.get('catboost', {}).items() if k not in ('use_best_model', 'early_stopping_rounds')}
        p.update({'loss_function': 'MAE', 'random_seed': seed, 'verbose': 0})
        p.setdefault('iterations', 3000); p.setdefault('learning_rate', 0.03); p.setdefault('depth', 8)
        def fit(X, y):
            tr, va = _inner_split(len(y), seed)
            Xc = X.copy(); Xc['station_code'] = Xc['station_code'].astype(str)
            m = CatBoostRegressor(**p, od_type='Iter', od_wait=150)
            m.fit(Xc.iloc[tr], y[tr], cat_features=['station_code'], eval_set=(Xc.iloc[va], y[va]), use_best_model=True)
            class P:
                def predict(self, Z):
                    Zc = Z.copy(); Zc['station_code'] = Zc['station_code'].astype(str)
                    return m.predict(Zc)
            return P()
        return fit

    if name in ('mlp', 'ft'):
        from .ft_compact import FTRegressor
        p = dict(cfg.get('deep', {}))
        def fit(X, y):
            return FTRegressor(numeric, 'station_code', seed=seed, epochs=int(p.get('epochs', 60)),
                               batch=int(p.get('batch', 1024)), lr=float(p.get('lr', 1e-3)),
                               patience=int(p.get('patience', 8)), arch=name).fit(X, y)
        return fit

    if name == 'tabpfn':
        def fit(X, y):
            from tabpfn import TabPFNRegressor            # optional dependency
            n_max = int(bcfg.get('tabpfn', {}).get('max_train_rows', 10000))
            idx = np.random.default_rng(seed).permutation(len(y))[:n_max]
            m = TabPFNRegressor(device='cuda' if gpu else 'cpu', random_state=seed)
            m.fit(X.iloc[idx].to_numpy(np.float32), y[idx])
            return type('T', (), {'predict': lambda self, Z: m.predict(Z.to_numpy(np.float32))})()
        return fit

    raise ValueError(f'unknown model {name}')


# ----------------------------------------------------------------------------- protocols

def _frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = df[df[target].notna()].copy()
    d['t'] = pd.to_datetime(d['connection_time'], utc=True)
    d['t_end'] = pd.to_datetime(d['disconnect_time'], utc=True) if 'disconnect_time' in d else d['t']
    d = d.sort_values('t').reset_index(drop=True)
    d['station_code'] = d['station_id'].astype('category').cat.codes
    d['known_driver'] = d['user_id'].notna() & (d['user_id'].astype(str).str.strip() != '')
    return d


def splits(d: pd.DataFrame, protocol: str, repeats: int, folds: int = 5):
    """Yield (repeat, fold, train_idx, test_idx, columns_to_drop)."""
    n = len(d)
    if protocol == 'random':
        for r in range(repeats):
            for f, (tr, te) in enumerate(KFold(folds, shuffle=True, random_state=r).split(np.arange(n))):
                yield r, f, tr, te, None
    elif protocol == 'driver_grouped':
        known = np.flatnonzero(d['known_driver'].to_numpy())
        users = d.loc[known, 'user_id'].astype(str).to_numpy()
        for r in range(repeats):
            u = np.unique(users); perm = dict(zip(u, np.random.default_rng(r).permutation(len(u))))
            gid = np.array([perm[x] for x in users])
            for f, (tr, te) in enumerate(GroupKFold(folds).split(known, groups=gid)):
                yield r, f, known[tr], known[te], None
    elif protocol == 'chronological':
        t_end = d['t_end'].to_numpy(); t = d['t'].to_numpy()
        for r in range(repeats):
            for f, (tr, te) in enumerate(TimeSeriesSplit(folds).split(np.arange(n))):
                tr = tr[t_end[tr] < t[te[0]]]                 # label maturation before the test block
                yield r, f, tr, te, None
    elif protocol == 'chronological_prepandemic':
        # identical rolling-origin design restricted to sessions before the 2020 break,
        # so the protocol effect is measured without the pandemic regime change
        pre = np.flatnonzero((d['t'] < pd.Timestamp(PANDEMIC_BREAK, tz='UTC')).to_numpy())
        t_end = d['t_end'].to_numpy(); t = d['t'].to_numpy()
        for r in range(repeats):
            for f, (tr, te) in enumerate(TimeSeriesSplit(folds).split(pre)):
                tr, te = pre[tr], pre[te]
                tr = tr[t_end[tr] < t[te[0]]]
                yield r, f, tr, te, None
    elif protocol == 'site_transfer_strict':
        # source-only transfer: nothing observed at the target site is available,
        # including drivers' own earlier sessions there
        sites = d['site'].to_numpy()
        for r in range(repeats):
            for f, s in enumerate(sorted(np.unique(sites))):
                te = np.flatnonzero(sites == s); tr = np.flatnonzero(sites != s)
                yield r, f, tr, te, STATION_HISTORY + USER_HISTORY + ['station_code']
    elif protocol == 'site_transfer':
        sites = d['site'].to_numpy()
        for r in range(repeats):
            for f, s in enumerate(sorted(np.unique(sites))):
                te = np.flatnonzero(sites == s); tr = np.flatnonzero(sites != s)
                yield r, f, tr, te, STATION_HISTORY + ['station_code']   # cannot exist at an unseen site
    else:
        raise ValueError(protocol)


# ----------------------------------------------------------------------------- run

def run(df: pd.DataFrame, numeric: list[str], target: str, protocol: str, models: list[str], cfg: dict,
        repeats: int = 5, folds: int = 5, out_dir: Path | None = None, log=print) -> pd.DataFrame:
    d = _frame(df, target)
    y = d[target].to_numpy(float)
    rows = []
    for r, f, tr, te, drop in splits(d, protocol, repeats, folds):
        if len(tr) < 200 or len(te) < 50:
            continue
        num = [c for c in numeric if c not in (drop or [])]
        X = d[num + ['station_code']].copy()
        if 'station_code' in (drop or []):
            X['station_code'] = 0          # station identity does not transfer: one shared level for every model
        Xte = X.iloc[te]
        for name in models:
            shard = out_dir / f'{target}__{protocol}__{name}__r{r}f{f}.parquet' if out_dir else None
            if shard is not None and shard.exists():
                rows.append(pd.read_parquet(shard)); continue
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pred = make_model(name, num, target, 1000 * r + f, cfg)(X.iloc[tr], y[tr]).predict(Xte)
                err = y[te] - np.asarray(pred, float)
                rec = {'target': target, 'protocol': protocol, 'model': name, 'repeat': r, 'fold': f,
                       'n_train': len(tr), 'n_test': len(te), 'mae': float(np.mean(np.abs(err))),
                       'rmse': float(np.sqrt(np.mean(err ** 2))),
                       'r2': float(1 - np.sum(err ** 2) / np.sum((y[te] - y[te].mean()) ** 2)),
                       'seconds': time.time() - t0, 'status': 'ok'}
            except Exception as exc:                       # optional models may be unavailable
                rec = {'target': target, 'protocol': protocol, 'model': name, 'repeat': r, 'fold': f,
                       'n_train': len(tr), 'n_test': len(te), 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
                       'seconds': time.time() - t0, 'status': f'failed: {type(exc).__name__}: {str(exc)[:120]}'}
            row = pd.DataFrame([rec])
            if shard is not None:
                row.to_parquet(shard, index=False)
            rows.append(row)
            log(f'[benchmark] {target} {protocol} {name} r{r} f{f} mae={rec["mae"]:.3f} ({rec["seconds"]:.0f}s) {rec["status"] if rec["status"]!="ok" else ""}')
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------- statistics

def corrected_ttest(d: np.ndarray, n_train: float, n_test: float) -> tuple[float, float]:
    """Nadeau and Bengio (2003) corrected resampled t-test for repeated cross-validation."""
    from scipy import stats
    d = np.asarray(d, float); J = len(d)
    if J < 2 or np.allclose(d, d[0]):
        return float('nan'), 1.0 if np.allclose(d, 0) else 0.0
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def summarise(scores: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    ok = scores[scores['status'] == 'ok']
    summ = (ok.groupby(['target', 'protocol', 'model'])
              .agg(mae=('mae', 'mean'), mae_sd=('mae', 'std'), rmse=('rmse', 'mean'), r2=('r2', 'mean'),
                   folds=('mae', 'size'), seconds=('seconds', 'mean')).reset_index())
    pair_rows, fried_rows = [], []
    for (target, protocol), g in ok.groupby(['target', 'protocol']):
        piv = g.pivot_table(index=['repeat', 'fold'], columns='model', values='mae').dropna(axis=1, how='any')
        if piv.shape[1] < 2:
            continue
        best = piv.mean().idxmin()
        ntr = g['n_train'].mean(); nte = g['n_test'].mean()
        ps = []
        for m in piv.columns:
            if m == best:
                continue
            t, p = corrected_ttest((piv[m] - piv[best]).to_numpy(), ntr, nte)
            ps.append((m, piv[m].mean() - piv[best].mean(), t, p))
        if ps:
            adj = multipletests([x[3] for x in ps], method='holm')[1]
            for (m, diff, t, p), pa in zip(ps, adj):
                pair_rows.append({'target': target, 'protocol': protocol, 'best_model': best, 'model': m,
                                  'mae_difference': diff, 't_corrected': t, 'p_value': p, 'p_holm': pa,
                                  'significantly_worse': bool(pa < alpha and diff > 0)})
        ranks = piv.rank(axis=1).mean().sort_values()
        k, N = piv.shape[1], piv.shape[0]
        chi = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns]) if k >= 3 else None
        q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
        cd = q * np.sqrt(k * (k + 1) / (6 * N))
        for m, rk in ranks.items():
            fried_rows.append({'target': target, 'protocol': protocol, 'model': m, 'average_rank': rk,
                               'friedman_chi2': None if chi is None else chi.statistic,
                               'friedman_p': None if chi is None else chi.pvalue,
                               'nemenyi_cd': cd, 'n_blocks': N, 'n_models': k})
    return summ, pd.DataFrame(pair_rows), pd.DataFrame(fried_rows)
