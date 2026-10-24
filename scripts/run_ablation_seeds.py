"""Grouped feature ablation repeated over the five seeds.

The pipeline's ablation stage uses the canonical seed only. This script repeats the
identical procedure (same feature groups, same tuned hyperparameters, same splits) for
every seed in project.five_seeds, so that ablation effects carry a mean and a standard
deviation like every other analysis.

    python scripts/run_ablation_seeds.py --config configs/acn.yaml

Resumable: each (seed, split, target, ablation) fit is stored as one JSON shard.
Writes results/metrics/grouped_ablation_seeds.csv (per seed) and
results/metrics/grouped_ablation_seeds_summary.csv (mean and SD over seeds).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evtrust.config import load_config                                            # noqa: E402
from evtrust.experiments import make_splits, _split_frames, USER_HISTORY, STATION_HISTORY  # noqa: E402
from evtrust.metrics import regression_metrics                                    # noqa: E402
from evtrust.models import fit_catboost                                           # noqa: E402
from evtrust.utils import seed_everything                                         # noqa: E402

TARGETS = {'duration': 'duration_hours', 'energy': 'kwh_delivered'}
CALENDAR = {'arrival_hour_sin', 'arrival_hour_cos', 'weekday_sin', 'weekday_cos',
            'month_sin', 'month_cos', 'is_weekend'}


def groups_for(cfg: dict) -> dict[str, set]:
    numeric, categorical = list(cfg['features']['numeric']), list(cfg['features']['categorical'])
    return {'full': set(),
            'no_driver': set(cfg['features']['driver_features']),
            'no_user_history': set(USER_HISTORY),
            'no_station_history': set(STATION_HISTORY),
            'calendar_only': set(numeric + categorical) - CALENDAR}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/acn.yaml')
    ap.add_argument('--seeds', type=int, default=None, help='use only the first N seeds')
    a = ap.parse_args()
    cfg = load_config(a.config)
    seeds = list(cfg['project']['five_seeds'])[: a.seeds] if a.seeds else list(cfg['project']['five_seeds'])
    numeric, categorical = list(cfg['features']['numeric']), list(cfg['features']['categorical'])
    df = pd.read_parquet(Path(cfg['data'].get('processed_dir', 'data/processed')) / 'sessions.parquet')
    tuned_path = Path('results/metrics/tuned_hyperparameters.json')
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    groups = groups_for(cfg)
    shard_dir = Path('results/ablation_seed_shards'); shard_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_everything(seed)
        for sp in make_splits(df, cfg, seed):
            tr0, _, va0, te0 = _split_frames(df, sp)
            for tname, ycol in TARGETS.items():
                tr, va, te = (x[x[ycol].notna()] for x in (tr0, va0, te0))
                if min(len(tr), len(va), len(te)) < 20:
                    continue
                params = tuned.get(f'{tname}|{sp.name}', {}).get('catboost', cfg['models']['catboost'])
                for name, drop in groups.items():
                    f = shard_dir / f"seed{seed}__{sp.name.replace(':', '__')}__{tname}__{name}.json"
                    if f.exists():
                        continue
                    num = [c for c in numeric if c not in drop]
                    cat = [c for c in categorical if c not in drop]
                    if not num and not cat:
                        continue
                    m = fit_catboost(tr, tr[ycol].to_numpy(float), va, va[ycol].to_numpy(float), num, cat, params, seed)
                    met = regression_metrics(te[ycol], m.predict(te))
                    rec = {'seed': seed, 'target': tname, 'protocol': sp.name.split(':')[0], 'split_name': sp.name,
                           'target_site': sp.metadata.get('target_site'), 'ablation': name, 'n': int(len(te)),
                           **{k: float(v) for k, v in met.items()}}
                    f.write_text(json.dumps(rec))
                    print(f"[ablation] seed {seed} {sp.name} {tname} {name}: MAE {rec['mae']:.3f}", flush=True)

    rows = pd.DataFrame([json.loads(f.read_text()) for f in sorted(shard_dir.glob('*.json'))])
    full = rows[rows.ablation == 'full'][['seed', 'target', 'split_name', 'mae']].rename(columns={'mae': 'full_mae'})
    rows = rows.merge(full, on=['seed', 'target', 'split_name'], how='left')
    rows['delta_mae_vs_full'] = rows['mae'] - rows['full_mae']
    rows['delta_pct_vs_full'] = 100 * rows['delta_mae_vs_full'] / rows['full_mae']
    out = Path('results/metrics'); out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / 'grouped_ablation_seeds.csv', index=False)
    # site-disjoint splits are averaged over target sites with equal weight within each seed
    per_seed = rows.groupby(['target', 'protocol', 'ablation', 'seed'])[['mae', 'delta_pct_vs_full']].mean().reset_index()
    summ = (per_seed.groupby(['target', 'protocol', 'ablation'])
            .agg(mae_mean=('mae', 'mean'), mae_sd=('mae', 'std'),
                 delta_pct_mean=('delta_pct_vs_full', 'mean'), delta_pct_sd=('delta_pct_vs_full', 'std'),
                 seeds=('seed', 'nunique')).reset_index())
    summ.round(4).to_csv(out / 'grouped_ablation_seeds_summary.csv', index=False)
    print(summ.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
