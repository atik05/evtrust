"""When do ACN drivers enter their charging request, relative to plug-in?

    python scripts/request_timing.py --sessions data/processed/sessions.parquet --out results/robustness/acn/request_timing.csv

For every identified session (a user identifier is present) the script reads the user inputs, takes the earliest
modification time and reports the share of sessions with a request, the share whose first request was entered at or
before the recorded connection time (the requests a plug-in forecaster may use), and the delay of the others.
"""
import argparse
import ast
import json

import numpy as np
import pandas as pd


def parse(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return []
    if isinstance(v, dict):
        v = [v]
    try:
        return [x for x in v if isinstance(x, dict)]
    except TypeError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    s = pd.read_parquet(a.sessions)
    ident = s[s['user_id'].notna() & (s['user_id'].astype(str).str.strip() != '')].copy()
    t0 = pd.to_datetime(ident['connection_time'], utc=True)
    delay_s, has = [], []
    for v, c in zip(ident['user_inputs'], t0):
        ts = [pd.to_datetime(x.get('modifiedAt'), utc=True, errors='coerce') for x in parse(v)]
        ts = [t for t in ts if not pd.isna(t)]
        has.append(bool(ts))
        delay_s.append((min(ts) - c).total_seconds() if ts else np.nan)
    d = pd.Series(delay_s).dropna()
    late = d[d > 0]
    row = {'sessions': len(s), 'identified_sessions': len(ident),
           'identified_with_request_pct': 100 * float(np.mean(has)),
           'first_request_at_or_before_plugin_pct': 100 * float((d <= 0).mean()),
           'late_request_delay_median_s': float(late.median()),
           'late_request_delay_q25_s': float(late.quantile(0.25)),
           'late_request_delay_q75_s': float(late.quantile(0.75)),
           'late_request_delay_q90_s': float(late.quantile(0.90)),
           'late_within_5_min_pct': 100 * float((late <= 300).mean())}
    pd.DataFrame([row]).to_csv(a.out, index=False)
    for k, v in row.items():
        print(f'{k}: {v:.1f}' if isinstance(v, float) else f'{k}: {v}')


if __name__ == '__main__':
    main()
