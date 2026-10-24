"""Squared-error companion of the main CatBoost model.

Every scoring rule has its own optimal point forecast: the median for absolute error and the
mean for squared error (Gneiting, 2011). The main learners minimise absolute error, so their
R^2, a squared-error score, is not the quantity they target. This script refits CatBoost with
the squared-error loss on the frozen splits, with the hyperparameters tuned for the main study,
so that R^2 and RMSE are reported for the forecast that is optimal for them.

    python scripts/run_squared_error.py --config configs/acn.yaml

Resumable per (seed, split, target). Writes results/metrics/squared_error_variant.csv
(per seed) and squared_error_summary.csv (mean over seeds, next to the MAE-trained model).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evtrust.config import load_config                          # noqa: E402
from evtrust.experiments import _split_frames, make_splits      # noqa: E402
from evtrust.models import fit_catboost                         # noqa: E402
from evtrust.utils import seed_everything                       # noqa: E402

TARGETS = {'duration': 'duration_hours', 'energy': 'kwh_delivered'}


def scores(y, p, ybar):
    e = y - p
    sse = float(np.sum(e ** 2))
    return {'mae': float(np.mean(np.abs(e))), 'rmse': float(np.sqrt(np.mean(e ** 2))),
            'r2': 1 - sse / float(np.sum((y - y.mean()) ** 2)),
            'r2_oos': 1 - sse / float(np.sum((y - ybar) ** 2)),
            'mae_skill': 1 - float(np.mean(np.abs(e))) / float(np.mean(np.abs(y - ybar)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/acn.yaml')
    ap.add_argument('--seeds', type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seeds = list(cfg['project']['five_seeds'])[: a.seeds] if a.seeds else list(cfg['project']['five_seeds'])
    numeric, categorical = list(cfg['features']['numeric']), list(cfg['features']['categorical'])
    df = pd.read_parquet(Path(cfg['data'].get('processed_dir', 'data/processed')) / 'sessions.parquet')
    tuned_path = Path('results/metrics/tuned_hyperparameters.json')
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    shard_dir = Path('results/squared_error_shards'); shard_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_everything(seed)
        for sp in make_splits(df, cfg, seed):
            tr, ca, va, te = _split_frames(df, sp)
            for tname, ycol in TARGETS.items():
                f = shard_dir / f"seed{seed}__{sp.name.replace(':', '__')}__{tname}.json"
                if f.exists():
                    continue
                ytr, yva, yte = (x[ycol].to_numpy(float) for x in (tr, va, te))
                ok_tr, ok_va, ok_te = np.isfinite(ytr), np.isfinite(yva), np.isfinite(yte)
                params = dict(tuned.get(f'{tname}|{sp.name}', {}).get('catboost', cfg['models']['catboost']))
                params.update({'loss_function': 'RMSE', 'eval_metric': 'RMSE'})
                m = fit_catboost(tr[ok_tr], ytr[ok_tr], va[ok_va], yva[ok_va], numeric, categorical, params, seed)
                pred = np.asarray(m.predict(te[ok_te]), float)
                rec = {'seed': seed, 'protocol': sp.name.split(':')[0], 'split_name': sp.name, 'target': tname,
                       'model': 'catboost_squared_error', 'n': int(ok_te.sum()),
                       **scores(yte[ok_te], pred, float(np.mean(ytr[ok_tr])))}
                f.write_text(json.dumps(rec))
                print(f"[squared-error] seed {seed} {sp.name} {tname}: R2 {rec['r2']:.3f}  R2_oos {rec['r2_oos']:.3f}  MAE {rec['mae']:.3f}", flush=True)

    rows = pd.DataFrame([json.loads(f.read_text()) for f in sorted(shard_dir.glob('*.json'))])
    out = Path('results/metrics'); out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / 'squared_error_variant.csv', index=False)
    # site-disjoint rows are averaged over target sites with equal weight, as in the main tables
    per = rows.groupby(['target', 'protocol', 'seed'])[['mae', 'rmse', 'r2', 'r2_oos', 'mae_skill']].mean().reset_index()
    summ = per.groupby(['target', 'protocol'])[['mae', 'rmse', 'r2', 'r2_oos', 'mae_skill']].mean().add_suffix('_sqerr')
    main_path = out / 'regression_metrics_by_seed.csv'
    if main_path.exists():
        mm = pd.read_csv(main_path)
        mm = mm[mm['model'] == 'catboost']
        cols = [c for c in ['mae', 'rmse', 'r2', 'r2_oos', 'mae_skill'] if c in mm]
        mper = mm.groupby(['target', 'protocol', 'seed'])[cols].mean().reset_index()
        summ = summ.join(mper.groupby(['target', 'protocol'])[cols].mean().add_suffix('_abserr'))
    summ.round(4).to_csv(out / 'squared_error_summary.csv')
    print(summ.round(3).to_string())


if __name__ == '__main__':
    main()
