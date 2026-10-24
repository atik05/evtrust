"""External validation data: City of Palo Alto public charging sessions (ChargePoint Data CY20Q4.csv).

The file is converted to the raw ACN-Data field names so that the pipeline's own normalisation,
cleaning, feature construction, protocols and audit run unchanged. Nothing about the analysis is
adapted to this dataset; only the input is.

Mapping
    connectionTime     Start Date, converted to UTC with Start Time Zone (PDT, PST or UTC)
    disconnectTime     End Date, converted with End Time Zone
    doneChargingTime   connectionTime + Charging Time (hh:mm:ss), the time when energy stopped flowing
    kWhDelivered       Energy (kWh)
    userID             User ID (anonymised driver account), missing for anonymous sessions
    spaceID            Station Name and Port Number (one charging port)
    site               geographic location cluster: addresses within LINK_METRES of each other
    userInputs         none: the dataset has no driver requests
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

TZ_OFFSET_HOURS = {'PDT': -7, 'PST': -8, 'UTC': 0}
LINK_METRES = 600.0
CLUSTER_NAMES = {  # named after the area that contains each cluster's largest address
    '250 Hamilton Ave': 'downtown', '475 Cambridge Ave': 'california_ave',
    '3700 Middlefield Rd': 'mitchell_park', '1213 Newell Rd': 'rinconada',
}


def _haversine(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def location_clusters(d: pd.DataFrame, link_m: float = LINK_METRES) -> pd.Series:
    """Single-linkage clusters of addresses: two addresses join if they are within link_m metres."""
    addr = d.groupby('Address 1').agg(lat=('Latitude', 'mean'), lon=('Longitude', 'mean'), n=('Address 1', 'size'))
    names = list(addr.index); parent = {a: a for a in names}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _haversine((addr.lat[a], addr.lon[a]), (addr.lat[b], addr.lon[b])) <= link_m:
                parent[find(a)] = find(b)
    root = {a: find(a) for a in names}
    label = {}
    for r in set(root.values()):
        members = [a for a in names if root[a] == r]
        biggest = max(members, key=lambda a: addr.n[a])
        label[r] = CLUSTER_NAMES.get(biggest, biggest.lower().replace(' ', '_'))
    return d['Address 1'].map(lambda a: label[root[a]])


def _to_utc(local: pd.Series, zone: pd.Series) -> pd.Series:
    t = pd.to_datetime(local, format='%m/%d/%Y %H:%M', errors='coerce')
    off = zone.map(TZ_OFFSET_HOURS)
    return (t - pd.to_timedelta(off, unit='h')).dt.tz_localize('UTC')


def _hms_hours(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.split(':', expand=True).astype(float)
    return parts[0] + parts[1] / 60 + parts[2] / 3600


def load_raw(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')
    report = {'file_rows': len(d), 'file_md5': hashlib.md5(Path(csv_path).read_bytes()).hexdigest()}
    unknown_tz = ~d['Start Time Zone'].isin(TZ_OFFSET_HOURS) | ~d['End Time Zone'].isin(TZ_OFFSET_HOURS)
    report['rows_unknown_time_zone'] = int(unknown_tz.sum())
    d = d[~unknown_tz].copy()
    start = _to_utc(d['Start Date'], d['Start Time Zone'])
    end = _to_utc(d['End Date'], d['End Time Zone'])
    charge_h = _hms_hours(d['Charging Time (hh:mm:ss)'])
    total_h = _hms_hours(d['Total Duration (hh:mm:ss)'])
    # Start and end are recorded to the minute; the recorded total duration is a cross-check.
    report['median_abs_diff_total_vs_end_minus_start_min'] = float(np.nanmedian(np.abs((end - start).dt.total_seconds() / 60 - total_h * 60)))
    d['site'] = location_clusters(d)
    report['sites'] = d['site'].value_counts().to_dict()
    cols = ['Station Name', 'Port Number', 'Start Date', 'End Date', 'Energy (kWh)', 'User ID']
    key = d[cols].astype(object).map(str).agg('|'.join, axis=1)
    uid = d['User ID'].map(lambda v: None if pd.isna(v) or str(v).strip() == '' else 'pa' + str(v).strip().removesuffix('.0'))
    raw = pd.DataFrame({
        '_id': key.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:16]),
        'site': d['site'].values,
        'spaceID': (d['Station Name'].astype(object).map(str) + ' #' + d['Port Number'].astype(object).map(str)).values,
        'userID': uid.values,
        'kWhDelivered': pd.to_numeric(d['Energy (kWh)'], errors='coerce').values,
        'connectionTime': start.dt.strftime('%Y-%m-%dT%H:%M:%SZ').values,
        'disconnectTime': end.dt.strftime('%Y-%m-%dT%H:%M:%SZ').values,
        'doneChargingTime': (start + pd.to_timedelta(charge_h, unit='h')).dt.strftime('%Y-%m-%dT%H:%M:%SZ').values,
    })
    raw['userInputs'] = [[] for _ in range(len(raw))]
    report['identified_share'] = float(raw['userID'].notna().mean())
    report['charging_longer_than_stay'] = int((charge_h > total_h + 1 / 60).sum())
    return raw, report
