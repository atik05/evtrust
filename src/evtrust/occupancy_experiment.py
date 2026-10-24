from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .occupancy import fit_occupancy_catboost, predict_free_probability, recommendation_metrics, risk_coverage_curve
from .conformal import mondrian_binary_calibrate, mondrian_prediction_set
from .metrics import binary_metrics
from .utils import ensure_dirs, seed_everything

FEATURES = ['arrival_hour_sin','arrival_hour_cos','weekday_sin','weekday_cos','occupied_now','site','station_id','is_weekend']


def _purge_query_mask(panel: pd.DataFrame, mask: pd.Series, boundary: pd.Timestamp, max_horizon_hours: float) -> pd.Series:
    """Remove earlier query rows whose future occupancy label reaches the next block.

    For an h-hour occupancy target, the outcome is not fully determined until
    query_time + h. Using the longest configured horizon as the embargo keeps
    every earlier-block label strictly before the next block begins.
    """
    out = mask.copy()
    if pd.isna(boundary):
        return out
    safe = panel['query_time'] + pd.Timedelta(hours=float(max_horizon_hours)) < boundary
    return out & safe


def _chronological_parts(panel: pd.DataFrame, max_horizon_hours: float):
    """train -> validation -> calibration -> test with longest-horizon embargo."""
    times = np.sort(panel['query_time'].unique())
    n = len(times)
    if n < 8:
        empty = pd.Series(False, index=panel.index)
        return [empty, empty, empty, empty]
    a = max(1, int(.60*n))
    b = max(a+1, int(.70*n))
    c = max(b+1, int(.80*n))
    b = min(b, n-2); c = min(c, n-1)

    train = panel.query_time.isin(times[:a])
    val = panel.query_time.isin(times[a:b])
    cal = panel.query_time.isin(times[b:c])
    test = panel.query_time.isin(times[c:])

    val_start = pd.Timestamp(times[a])
    cal_start = pd.Timestamp(times[b])
    test_start = pd.Timestamp(times[c])
    train = _purge_query_mask(panel, train, val_start, max_horizon_hours)
    val = _purge_query_mask(panel, val, cal_start, max_horizon_hours)
    cal = _purge_query_mask(panel, cal, test_start, max_horizon_hours)
    return [train, val, cal, test]


def _site_disjoint_parts(panel: pd.DataFrame, target: str, max_horizon_hours: float):
    """Source train/validation + target calibration/test with label-horizon embargo."""
    source = panel[panel.site != target].copy()
    target_df = panel[panel.site == target].sort_values('query_time')
    src_times = np.sort(source.query_time.unique())
    tgt_times = np.sort(target_df.query_time.unique())
    if len(src_times) < 4 or len(tgt_times) < 4:
        empty = pd.Series(False, index=panel.index)
        return [empty, empty, empty, empty]

    cut = max(1, int(.85*len(src_times)))
    cut = min(cut, len(src_times)-1)
    tr = panel.index.isin(source[source.query_time.isin(src_times[:cut])].index)
    va = panel.index.isin(source[source.query_time.isin(src_times[cut:])].index)
    source_val_start = pd.Timestamp(src_times[cut])
    tr = _purge_query_mask(panel, pd.Series(tr, index=panel.index), source_val_start, max_horizon_hours)

    cc = max(1, int(.15*len(tgt_times)))
    cc = min(cc, len(tgt_times)-1)
    cal = panel.index.isin(target_df[target_df.query_time.isin(tgt_times[:cc])].index)
    test = panel.index.isin(target_df[target_df.query_time.isin(tgt_times[cc:])].index)
    target_test_start = pd.Timestamp(tgt_times[cc])
    cal = _purge_query_mask(panel, pd.Series(cal, index=panel.index), target_test_start, max_horizon_hours)

    return [
        pd.Series(tr, index=panel.index) if not isinstance(tr, pd.Series) else tr,
        pd.Series(va, index=panel.index),
        cal,
        pd.Series(test, index=panel.index),
    ]


def _occ_shard(seed: int, split_name: str, horizon, kind: str) -> Path:
    safe = str(split_name).replace(':', '__')
    return Path('results/predictions/occupancy_shards') / f'{kind}__seed{seed}__{safe}__h{horizon}.parquet'


def run_occupancy(panel: pd.DataFrame, cfg: dict, out='results/predictions/occupancy_predictions.parquet'):
    seeds = cfg['project']['five_seeds']
    horizons = cfg['project']['horizons_hours']
    alpha = cfg['conformal']['alpha']
    # Congestion is defined only from information observable at query time:
    # the fraction of stations currently free, never from future test labels.
    threshold = cfg['occupancy'].get(
        'congested_current_free_fraction_threshold',
        cfg['occupancy'].get('congested_free_probability_threshold', 0.50),
    )
    max_horizon = max(float(h) for h in horizons)
    all_pred = []; headline = []; risks = []

    protocols = []
    masks = _chronological_parts(panel, max_horizon)
    protocols.append(('chronological', None, *masks))
    for target in sorted(panel.site.unique()):
        tr, va, cal, test = _site_disjoint_parts(panel, target, max_horizon)
        protocols.append((f'site_disjoint:{target}', target, tr, va, cal, test))

    resume = bool(cfg['project'].get('resume', True))
    Path('results/predictions/occupancy_shards').mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_everything(seed)
        for split_name, target_site, trm, vam, cam, tem in protocols:
            tr = panel[trm].copy(); va = panel[vam].copy(); ca = panel[cam].copy(); te = panel[tem].copy()
            if min(len(tr), len(va), len(ca), len(te)) < 50:
                continue
            for h in horizons:
                pred_shard = _occ_shard(seed, split_name, h, 'pred')
                head_shard = _occ_shard(seed, split_name, h, 'head')
                risk_shard = _occ_shard(seed, split_name, h, 'risk')
                if resume and pred_shard.exists():
                    # Completed horizon: reload instead of refitting after a restart.
                    all_pred.append(pd.read_parquet(pred_shard))
                    if head_shard.exists():
                        headline.extend(pd.read_parquet(head_shard).to_dict('records'))
                    if risk_shard.exists():
                        risks.append(pd.read_parquet(risk_shard))
                    continue
                ycol = f'free_{h}h'
                model, features, cat = fit_occupancy_catboost(
                    tr, tr[ycol].to_numpy(int), va, va[ycol].to_numpy(int), seed,
                    gpu=(str(cfg['models']['catboost'].get('task_type','CPU')).upper() == 'GPU'),
                    params=cfg.get('occupancy', {}).get('model', {}),
                )
                pca = predict_free_probability(model, ca, features, cat)
                pte = predict_free_probability(model, te, features, cat)
                qhat = mondrian_binary_calibrate(ca[ycol].to_numpy(int), pca, alpha)
                pset = mondrian_prediction_set(pte, qhat)
                z = te[['query_id','query_time','site','station_id','occupied_now']].copy()
                z['protocol'] = split_name.split(':')[0]
                z['split_name'] = split_name
                z['target_site'] = target_site
                z['seed'] = seed
                z['horizon'] = h
                z['y_free'] = te[ycol].to_numpy(int)
                z['p_free'] = pte
                z['set_contains_occupied'] = pset[:,0]
                z['set_contains_free'] = pset[:,1]

                current_free = (1.0 - z.groupby('query_id')['occupied_now'].mean()).rename('current_free_fraction')
                z = z.join(current_free, on='query_id')
                z['is_congested_at_query'] = z['current_free_fraction'] < float(threshold)
                z.to_parquet(pred_shard, index=False)
                all_pred.append(z)

                zc = z[z.is_congested_at_query].copy()
                if len(zc):
                    met = {
                        **binary_metrics(zc.y_free, zc.p_free),
                        **recommendation_metrics(zc, cfg['occupancy']['confidence_threshold']),
                    }
                    met.update({
                        'protocol': split_name.split(':')[0],
                        'split_name': split_name,
                        'target_site': target_site,
                        'seed': seed,
                        'horizon': h,
                        'congestion_definition': 'current_free_fraction_at_query',
                        'congestion_threshold': float(threshold),
                        'n_congested_station_rows': int(len(zc)),
                    })
                    headline.append(met)
                    pd.DataFrame([met]).to_parquet(head_shard, index=False)
                    rc = risk_coverage_curve(zc)
                    rc['protocol'] = split_name.split(':')[0]
                    rc['split_name'] = split_name
                    rc['target_site'] = target_site
                    rc['seed'] = seed
                    rc['horizon'] = h
                    rc['congestion_definition'] = 'current_free_fraction_at_query'
                    rc.to_parquet(risk_shard, index=False)
                    risks.append(rc)

    ensure_dirs(Path(out).parent, 'results/metrics')
    if all_pred:
        pd.concat(all_pred, ignore_index=True).to_parquet(out, index=False)
    if headline:
        pd.DataFrame(headline).to_csv('results/metrics/occupancy_metrics.csv', index=False)
    if risks:
        pd.concat(risks, ignore_index=True).to_csv('results/metrics/risk_coverage.csv', index=False)
