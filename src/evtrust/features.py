from __future__ import annotations
import numpy as np
import pandas as pd


def _parse_user_input(record, prediction_time: pd.Timestamp) -> tuple[float, float]:
    """Return (requested_energy_kwh, requested_duration_hours) using only inputs available by prediction_time."""
    if not isinstance(record, list):
        return (np.nan, np.nan)
    candidates = []
    for x in record:
        if not isinstance(x, dict):
            continue
        modified = pd.to_datetime(x.get('modifiedAt'), utc=True, errors='coerce')
        if pd.isna(modified):
            # Some exports contain only the initial request without modifiedAt.
            modified = prediction_time
        else:
            try:
                modified = modified.tz_convert(prediction_time.tz)
            except Exception:
                pass
        if modified <= prediction_time:
            candidates.append((modified, x))
    if not candidates:
        return (np.nan, np.nan)
    _, x = max(candidates, key=lambda t: t[0])
    kwh = x.get('kWhRequested', np.nan)
    dep = pd.to_datetime(x.get('requestedDeparture'), utc=True, errors='coerce')
    if pd.isna(dep):
        mins = x.get('minutesAvailable', np.nan)
        duration_h = float(mins) / 60 if mins is not None and pd.notna(mins) else np.nan
    else:
        try:
            dep = dep.tz_convert(prediction_time.tz)
        except Exception:
            pass
        duration_h = (dep - prediction_time).total_seconds() / 3600
    try:
        kwh = float(kwh)
    except Exception:
        kwh = np.nan
    return (kwh, duration_h if duration_h >= 0 else np.nan)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out['connection_time']
    hour = t.dt.hour + t.dt.minute / 60
    dow = t.dt.dayofweek
    month = t.dt.month - 1
    out['arrival_hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['arrival_hour_cos'] = np.cos(2 * np.pi * hour / 24)
    out['weekday_sin'] = np.sin(2 * np.pi * dow / 7)
    out['weekday_cos'] = np.cos(2 * np.pi * dow / 7)
    out['month_sin'] = np.sin(2 * np.pi * month / 12)
    out['month_cos'] = np.cos(2 * np.pi * month / 12)
    out['is_weekend'] = (dow >= 5).astype(int).astype(str)
    return out


def add_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vals = [_parse_user_input(r, t) for r, t in zip(out['user_inputs'], out['connection_time'])]
    out['requested_energy_kwh'] = [v[0] for v in vals]
    out['requested_duration_hours'] = [v[1] for v in vals]
    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly past-only user and station history based on sessions that have already ended.

    Implementation processes rows chronologically and only adds a completed session to history
    after its disconnect time. This is slower than a simple groupby/shift but avoids label maturation leakage.
    """
    out = df.sort_values('connection_time').copy()
    cols = [
        'user_prior_count', 'user_prior_mean_duration', 'user_prior_median_duration', 'user_prior_mean_energy',
        'station_prior_count', 'station_prior_mean_duration', 'station_prior_mean_energy'
    ]
    for c in cols:
        out[c] = np.nan

    # Event queue: before a new connection, activate all sessions whose labels are already mature.
    completed = out.sort_values('disconnect_time')
    ptr = 0
    user_hist: dict[str, list[tuple[float, float]]] = {}
    station_hist: dict[tuple[str, str], list[tuple[float, float]]] = {}

    for idx, row in out.iterrows():
        now = row['connection_time']
        while ptr < len(completed) and completed.iloc[ptr]['disconnect_time'] < now:
            r = completed.iloc[ptr]
            dur, ene = float(r['duration_hours']), float(r['kwh_delivered']) if pd.notna(r['kwh_delivered']) else np.nan
            uid = str(r['user_id'])
            sid = (str(r['site']), str(r['station_id']))
            if uid not in ('<NA>', 'nan', 'None'):
                user_hist.setdefault(uid, []).append((dur, ene))
            station_hist.setdefault(sid, []).append((dur, ene))
            ptr += 1

        uid = str(row['user_id'])
        uh = user_hist.get(uid, [])
        if uh:
            durs = np.array([x[0] for x in uh], dtype=float)
            enes = np.array([x[1] for x in uh], dtype=float)
            out.at[idx, 'user_prior_count'] = len(uh)
            out.at[idx, 'user_prior_mean_duration'] = np.nanmean(durs)
            out.at[idx, 'user_prior_median_duration'] = np.nanmedian(durs)
            out.at[idx, 'user_prior_mean_energy'] = np.nanmean(enes)
        else:
            out.at[idx, 'user_prior_count'] = 0

        sid = (str(row['site']), str(row['station_id']))
        sh = station_hist.get(sid, [])
        if sh:
            durs = np.array([x[0] for x in sh], dtype=float)
            enes = np.array([x[1] for x in sh], dtype=float)
            out.at[idx, 'station_prior_count'] = len(sh)
            out.at[idx, 'station_prior_mean_duration'] = np.nanmean(durs)
            out.at[idx, 'station_prior_mean_energy'] = np.nanmean(enes)
        else:
            out.at[idx, 'station_prior_count'] = 0

    return out.sort_values('connection_time').reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_history_features(add_driver_features(add_time_features(df)))
