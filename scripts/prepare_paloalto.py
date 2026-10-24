"""Prepare the Palo Alto external-validation data with the pipeline's own cleaning and features.

    python scripts/prepare_paloalto.py --csv /path/to/ChargePoint_Data_CY20Q4.csv --config configs/paloalto.yaml

Writes <processed_dir>/sessions.parquet, results/metrics/data_flow.csv, results/metrics/external_source.json
and marks the download and prepare stages as done, so the rest of the pipeline runs unchanged.
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from evtrust.config import load_config                      # noqa: E402
from evtrust.data import normalize_sessions, clean_sessions  # noqa: E402
from evtrust.features import build_features                 # noqa: E402
from evtrust.external_paloalto import load_raw              # noqa: E402

EXPECTED_MD5 = '15360a7df4c0b4f0b89ea7ec03a54e20'

ap = argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--config', default='configs/paloalto.yaml')
a = ap.parse_args(); cfg = load_config(a.config); d = cfg['data']
t = time.time(); raw, rep = load_raw(a.csv)
rep['md5_matches_verified_copy'] = rep['file_md5'] == EXPECTED_MD5
print(json.dumps(rep, indent=1, default=str), flush=True)
df = normalize_sessions(raw, d['timezone'])
df = clean_sessions(df, d['max_duration_hours'], d['max_energy_kwh'], d['min_duration_minutes'], flow_path='results/metrics/data_flow.csv')
df = build_features(df)
out = Path(d['processed_dir']) / 'sessions.parquet'; out.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(out, index=False)
rep.update(sessions_after_cleaning=len(df), identified_after_cleaning=float(df['user_id'].notna().mean()), seconds=round(time.time() - t))
Path('results/metrics').mkdir(parents=True, exist_ok=True); Path('results/metrics/external_source.json').write_text(json.dumps(rep, indent=1, default=str))
Path('results/.stages').mkdir(parents=True, exist_ok=True)
for s in ['download', 'prepare']: Path(f'results/.stages/{s}.done').touch()
print(f'wrote {out}: {len(df)} sessions in {rep["seconds"]}s', flush=True)
