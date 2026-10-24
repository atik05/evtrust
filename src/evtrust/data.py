from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

TIME_COLS = ['connectionTime', 'disconnectTime', 'doneChargingTime']


def load_raw_json(raw_dir: str | Path, sites: list[str]) -> pd.DataFrame:
    frames = []
    for site in sites:
        path = Path(raw_dir) / f'{site}_sessions.json'
        if not path.exists():
            raise FileNotFoundError(path)
        rows = json.loads(path.read_text(encoding='utf-8'))
        df = pd.json_normalize(rows)
        df['site'] = site
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _pick_col(df: pd.DataFrame, *names: str, default=np.nan):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def normalize_sessions(raw: pd.DataFrame, timezone: str = 'America/Los_Angeles') -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    df['session_id'] = _pick_col(raw, '_id', 'sessionID', 'session_id').astype(str)
    df['site'] = _pick_col(raw, 'site').astype(str)
    df['station_id'] = _pick_col(raw, 'spaceID', 'stationID', 'station_id').astype(str)
    df['user_id'] = _pick_col(raw, 'userID', 'user_id').astype('string')
    df['kwh_delivered'] = pd.to_numeric(_pick_col(raw, 'kWhDelivered', 'kwhDelivered'), errors='coerce')

    for src, dst in [('connectionTime', 'connection_time'), ('disconnectTime', 'disconnect_time'),
                     ('doneChargingTime', 'done_charging_time')]:
        s = _pick_col(raw, src)
        ts = pd.to_datetime(s, utc=True, errors='coerce')
        try:
            ts = ts.dt.tz_convert(timezone)
        except Exception:
            pass
        df[dst] = ts

    # Preserve raw user inputs to enforce prediction-time filtering later.
    if 'userInputs' in raw.columns:
        df['user_inputs'] = raw['userInputs']
    else:
        df['user_inputs'] = [[] for _ in range(len(df))]

    df['duration_hours'] = (df['disconnect_time'] - df['connection_time']).dt.total_seconds() / 3600
    df['charging_duration_hours'] = (df['done_charging_time'] - df['connection_time']).dt.total_seconds() / 3600
    return df


def clean_sessions(df: pd.DataFrame, max_duration_hours: float = 168, max_energy_kwh: float = 200,
                   min_duration_minutes: float = 5, flow_path: str | Path | None = None) -> pd.DataFrame:
    """Apply the declared cleaning filters and optionally record the row flow.

    A minimum stay is enforced because zero-length and near-zero-length records
    are connector artefacts rather than charging sessions.
    """
    flow = [('raw_sessions', len(df))]
    out = df.copy()
    out = out[out['connection_time'].notna() & out['disconnect_time'].notna()]
    flow.append(('with_valid_timestamps', len(out)))
    out = out[(out['duration_hours'] >= min_duration_minutes / 60) & (out['duration_hours'] <= max_duration_hours)]
    flow.append(('within_duration_bounds', len(out)))
    out = out[(out['kwh_delivered'].isna()) | ((out['kwh_delivered'] >= 0) & (out['kwh_delivered'] <= max_energy_kwh))]
    flow.append(('within_energy_bounds', len(out)))
    out = out.drop_duplicates(subset=['site', 'session_id']).sort_values('connection_time').reset_index(drop=True)
    flow.append(('after_duplicate_removal', len(out)))
    flow.append(('energy_label_available', int(out['kwh_delivered'].notna().sum())))
    if flow_path is not None:
        rows = pd.DataFrame(flow, columns=['stage', 'sessions'])
        rows['removed_from_previous'] = rows['sessions'].shift(1) - rows['sessions']
        Path(flow_path).parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(flow_path, index=False)
    return out
