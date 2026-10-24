from __future__ import annotations
import os
import platform
import numpy as np
import pandas as pd


def _occupied_at(starts_ns: np.ndarray, ends_ns: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(starts_ns, query_ns, side='right') - 1
    valid = pos >= 0
    out = np.zeros(len(query_ns), dtype=bool)
    idx = pos[valid]
    out[valid] = ends_ns[idx] > query_ns[valid]
    return out


def build_occupancy_panel(sessions: pd.DataFrame, horizons_hours=(1,2,4), freq_minutes=15) -> pd.DataFrame:
    """Build station-query examples at rounded session-arrival times.

    This avoids an enormous all-minutes panel while representing times when an operator is likely to
    receive a new arrival/query. Labels are derived from station connection intervals.
    """
    rows = []
    for site, sdf in sessions.groupby('site'):
        stations = sorted(sdf['station_id'].astype(str).unique())
        query_times = pd.Series(sdf['connection_time'].dt.floor(f'{freq_minutes}min').unique()).dropna().sort_values().to_list()
        timelines = {}
        for station, g in sdf.groupby(sdf['station_id'].astype(str)):
            gg = g.sort_values('connection_time')
            timelines[station] = (
                gg['connection_time'].astype('int64').to_numpy(),
                gg['disconnect_time'].astype('int64').to_numpy(),
            )
        q = pd.DatetimeIndex(query_times)
        q_ns = q.astype('int64').to_numpy()
        base_time = pd.DataFrame({'query_time': q})
        base_time['arrival_hour_sin'] = np.sin(2*np.pi*(q.hour+q.minute/60)/24)
        base_time['arrival_hour_cos'] = np.cos(2*np.pi*(q.hour+q.minute/60)/24)
        base_time['weekday_sin'] = np.sin(2*np.pi*q.dayofweek/7)
        base_time['weekday_cos'] = np.cos(2*np.pi*q.dayofweek/7)
        base_time['is_weekend'] = (q.dayofweek >= 5).astype(int).astype(str)

        for station in stations:
            starts, ends = timelines[station]
            frame = base_time.copy()
            frame['site'] = site
            frame['station_id'] = station
            frame['occupied_now'] = _occupied_at(starts, ends, q_ns).astype(int)
            for h in horizons_hours:
                future_ns = (q + pd.Timedelta(hours=h)).astype('int64').to_numpy()
                frame[f'free_{h}h'] = (~_occupied_at(starts, ends, future_ns)).astype(int)
            rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel['query_id'] = panel['site'].astype(str) + '|' + panel['query_time'].astype(str)
    return panel.sort_values(['query_time','site','station_id']).reset_index(drop=True)


OCCUPANCY_DEFAULTS = {'iterations': 1800, 'depth': 8, 'learning_rate': 0.04, 'l2_leaf_reg': 3.0,
                      'early_stopping_rounds': 120}


def fit_occupancy_catboost(train: pd.DataFrame, y: np.ndarray, validation: pd.DataFrame, y_val: np.ndarray,
                           seed: int, gpu=True, params: dict | None = None):
    from catboost import CatBoostClassifier
    features = ['arrival_hour_sin','arrival_hour_cos','weekday_sin','weekday_cos','occupied_now','site','station_id','is_weekend']
    cat = ['site','station_id','is_weekend']
    xtr = train[features].copy(); xva = validation[features].copy()
    for c in cat:
        xtr[c] = xtr[c].astype(str); xva[c] = xva[c].astype(str)
    # CatBoost GPU requires NVIDIA CUDA. Intel Macs with AMD Radeon run CatBoost on CPU.
    use_gpu = bool(gpu and platform.system() != 'Darwin' and os.environ.get('EVTRUST_CATBOOST_TASK_TYPE','GPU').upper() == 'GPU')
    kwargs = dict(
        loss_function='Logloss', eval_metric='Logloss',
        random_seed=seed, task_type='GPU' if use_gpu else 'CPU',
        verbose=200, allow_writing_files=False
    )
    if use_gpu:
        kwargs['devices'] = '0'
    else:
        kwargs['thread_count'] = int(os.environ.get('EVTRUST_CPU_THREADS', '8'))
    hp = dict(OCCUPANCY_DEFAULTS)
    hp.update(params or {})
    early = int(hp.pop('early_stopping_rounds', 120))
    kwargs.update({k: v for k, v in hp.items() if k not in kwargs})
    model = CatBoostClassifier(**kwargs)
    model.fit(xtr, y, cat_features=[xtr.columns.get_loc(c) for c in cat], eval_set=(xva, y_val),
              early_stopping_rounds=early, use_best_model=True)
    return model, features, cat


def predict_free_probability(model, df: pd.DataFrame, features: list[str], cat: list[str]) -> np.ndarray:
    x = df[features].copy()
    for c in cat:
        x[c] = x[c].astype(str)
    return model.predict_proba(x)[:,1]


def recommendation_metrics(pred: pd.DataFrame, confidence_threshold=.8) -> dict:
    """One recommendation per site/query/horizon: station with highest predicted free probability."""
    chosen = pred.sort_values('p_free', ascending=False).groupby(['query_id','horizon'], as_index=False).first()
    hit = chosen['y_free'].to_numpy(int)
    confident = chosen['p_free'].to_numpy(float) >= confidence_threshold
    cwr = float(np.mean(1-hit[confident])) if confident.any() else float('nan')
    return {
        'recommendation_hit_rate': float(np.mean(hit)),
        'confidently_wrong_rate': cwr,
        'decision_coverage': float(np.mean(confident)),
        'n_queries': int(len(chosen))
    }


def risk_coverage_curve(pred: pd.DataFrame, thresholds=np.linspace(.5,.99,50)) -> pd.DataFrame:
    chosen = pred.sort_values('p_free', ascending=False).groupby(['query_id','horizon'], as_index=False).first()
    rows = []
    for t in thresholds:
        mask = chosen['p_free'] >= t
        if mask.any():
            risk = 1 - chosen.loc[mask, 'y_free'].mean()
        else:
            risk = np.nan
        rows.append({'threshold':float(t), 'coverage':float(mask.mean()), 'risk':float(risk)})
    return pd.DataFrame(rows)
