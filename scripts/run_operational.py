"""Operational analyses from saved predictions: driver request, site level, bias correction.

    python scripts/run_operational.py --config configs/acn.yaml            # all three
    python scripts/run_operational.py --config ... --only request site bias        # any subset

Inputs (already produced by the pipeline):
    results/paper/shards/cs_{learner}_{target}_{seed}.parquet       cold-start predictions
    results/predictions/shards/pred__seed*__site_disjoint__*.parquet unseen-site predictions
    data/processed/sessions.parquet
Outputs: results/operational/*.csv and OPERATIONAL_RESULTS.md. No model is trained.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evtrust import impact, operational as op    # noqa: E402
from evtrust.config import load_config            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/acn.yaml')
    ap.add_argument('--only', nargs='*', default=['impact', 'request', 'site', 'bias'])
    ap.add_argument('--learners', nargs='*', default=['catboost', 'ft'])
    ap.add_argument('--site-learner', default='catboost')
    ap.add_argument('--n-boot', type=int, default=2000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    power = float(cfg.get('paper', {}).get('charging_power_kw', 6.6))
    sessions = pd.read_parquet(Path(cfg['data'].get('processed_dir', 'data/processed')) / 'sessions.parquet')
    out = Path('results/operational'); out.mkdir(parents=True, exist_ok=True)
    shards = Path('results/paper/shards')
    md = ['# Operational analyses', '']

    def cold_rows(learner):
        files = sorted(shards.glob(f'cs_{learner}_*.parquet'))
        if not files:
            raise SystemExit(f'no cold-start shards for {learner} in {shards}')
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    if 'impact' in a.only:
        # Recompute the smart-charging impact files with stable driver groups (replaces a salted hash).
        t = time.time(); paper = Path('results/paper'); tr_all, bk_all, mt_all = [], [], []
        for learner in a.learners:
            r = cold_rows(learner)
            summ, bk, matched, _ = impact.run(r[r['target'] == 'duration_hours'], r[r['target'] == 'kwh_delivered'], power)
            for tab, lst in [(summ, tr_all), (bk, bk_all), (matched, mt_all)]:
                if tab is not None and len(tab):
                    lst.append(tab.assign(learner=learner))
        pd.concat(tr_all, ignore_index=True).to_csv(paper / 'impact_tradeoff.csv', index=False)
        pd.concat(bk_all, ignore_index=True).to_csv(paper / 'impact_by_history.csv', index=False)
        mt = pd.concat(mt_all, ignore_index=True); mt.to_csv(paper / 'impact_matched_reliability.csv', index=False)
        md += ['## Smart-charging impact at matched reliability (stable driver groups)', '', mt.round(2).to_markdown(index=False), '']
        print(f'impact: done in {time.time() - t:.0f}s', flush=True)

    if 'request' in a.only:
        t = time.time(); parts = {}
        for learner in a.learners:
            res = op.request_matched(cold_rows(learner), power, n_boot=a.n_boot)
            for k, v in res.items():
                parts.setdefault(k, []).append(v.assign(learner=learner))
        for k, v in parts.items():
            df = pd.concat(v, ignore_index=True); df.to_csv(out / f'{k}.csv', index=False)
        m = pd.concat(parts['request_matched'], ignore_index=True)
        d = pd.concat(parts['request_differences'], ignore_index=True)
        md += ['## Driver request versus forecasts (identical sessions, matched reliability)', '',
               m.round(2).to_markdown(index=False), '', d.round(3).to_markdown(index=False), '']
        print(f'request: done in {time.time() - t:.0f}s', flush=True)

    if 'site' in a.only:
        t = time.time()
        res = op.site_level(cold_rows(a.site_learner), sessions, power, n_boot=min(a.n_boot, 1000))
        for k, v in res.items():
            v.assign(learner=a.site_learner).to_csv(out / f'{k}.csv', index=False)
        md += ['## Site-level coordinated charging under a capacity limit', '',
               res['site_capacity_needed'].round(3).to_markdown(index=False), '']
        print(f'site: done in {time.time() - t:.0f}s', flush=True)

    if 'bias' in a.only:
        t = time.time()
        files = sorted(Path('results/predictions/shards').glob('pred__*site_disjoint*.parquet'))
        if not files:
            raise SystemExit('no site-disjoint prediction shards found')
        pred = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        per = op.bias_correction(pred, sessions)
        per.to_csv(out / 'bias_correction_by_seed.csv', index=False)
        summ = (per.groupby(['target', 'target_site', 'model', 'budget'])[['mae', 'rmse', 'r2', 'r2_oos', 'n_eval']]
                .mean().reset_index())
        summ.to_csv(out / 'bias_correction.csv', index=False)
        md += ['## Target-site bias correction', '', summ.round(3).to_markdown(index=False), '']
        print(f'bias: done in {time.time() - t:.0f}s', flush=True)

    (out / 'OPERATIONAL_RESULTS.md').write_text('\n'.join(md))
    print('written:', sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main()
