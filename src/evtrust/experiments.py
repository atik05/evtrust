from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .splits import random_split, chronological_split, user_disjoint_split, site_disjoint_splits
from .models import fit_ridge, fit_catboost, fit_catboost_quantile
from .conformal import split_conformal_calibrate, split_conformal_interval, cqr_calibrate, cqr_interval
from .metrics import regression_metrics, interval_metrics
from .stats import protocol_optimism, rank_stability, block_metric_ci
from .utils import ensure_dirs, seed_everything

USER_HISTORY = ['user_prior_count','user_prior_mean_duration','user_prior_median_duration','user_prior_mean_energy']
STATION_HISTORY = ['station_prior_count','station_prior_mean_duration','station_prior_mean_energy']


def make_splits(df: pd.DataFrame, cfg: dict, seed: int):
    s = cfg['splits']
    return [
        random_split(df, seed, s['validation_fraction'], s['calibration_fraction'], s['test_fraction']),
        chronological_split(df, s['validation_fraction'], s['calibration_fraction'], s['test_fraction'], s['embargo_hours']),
        user_disjoint_split(df, seed, s['validation_fraction'], s['calibration_fraction'], s['test_fraction']),
        *site_disjoint_splits(df, s['validation_fraction'], s['calibration_fraction'], s['embargo_hours'])
    ]


def _split_frames(df, sp):
    tr, ca, va, te = (df.loc[x].copy() for x in [sp.train, sp.calibration, sp.validation, sp.test])
    if sp.name == 'user_disjoint':
        for frame in (ca, va, te):
            for c in USER_HISTORY:
                if c in frame:
                    frame[c] = 0 if c.endswith('_count') else np.nan
    if sp.name.startswith('site_disjoint'):
        # Strict zero-shot station-history features at the unseen site; local calibration is reserved for uncertainty.
        for frame in (ca, te):
            for c in STATION_HISTORY:
                if c in frame:
                    frame[c] = 0 if c.endswith('_count') else np.nan
    return tr, ca, va, te


def _shard_path(out_dir: Path, seed: int, split_name: str, target: str, kind: str = 'pred') -> Path:
    safe = split_name.replace(':', '__')
    return out_dir / 'shards' / f'{kind}__seed{seed}__{safe}__{target}.parquet'


def _save_predictions(rows: list[pd.DataFrame], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def run_regression(df: pd.DataFrame, cfg: dict, out_dir='results/predictions') -> None:
    numeric = cfg['features']['numeric']
    categorical = cfg['features']['categorical']
    seeds = cfg['project']['five_seeds']
    alpha = cfg['conformal']['alpha']
    targets = {'duration':'duration_hours', 'energy':'kwh_delivered'}
    ensure_dirs(out_dir, 'results/metrics')
    all_rows = []
    budget_rows = []
    rel_frames = []
    tuned_cache = {}

    learned = list(cfg['experiment'].get('learned_models', ['ridge', 'catboost', 'torch_ft']))
    # Heavy libraries are imported only when the corresponding model is requested,
    # so a CPU-only or analysis-only environment does not need torch or optuna.
    if 'torch_ft' in learned:
        from .torch_tabular import fit_torch_ft
    if cfg.get('tuning', {}).get('enabled', False):
        from .tuning import tune_ridge, tune_catboost, tune_torch
    use_cqr = bool(cfg['conformal'].get('cqr', True)) and 'catboost' in learned
    resume = bool(cfg['project'].get('resume', True))
    shard_dir = Path(out_dir) / 'shards'
    shard_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_everything(seed)
        for sp in make_splits(df, cfg, seed):
            tr0, ca0, va0, te0 = _split_frames(df, sp)
            for target_name, ycol in targets.items():
                shard = _shard_path(Path(out_dir), seed, sp.name, target_name)
                if resume and shard.exists():
                    # A completed unit of work is never recomputed after a crash or restart.
                    all_rows.append(pd.read_parquet(shard))
                    rel_shard = _shard_path(Path(out_dir), seed, sp.name, target_name, 'rel')
                    bud_shard = _shard_path(Path(out_dir), seed, sp.name, target_name, 'budget')
                    if rel_shard.exists():
                        rel_frames.append(pd.read_parquet(rel_shard))
                    if bud_shard.exists():
                        budget_rows.append(pd.read_parquet(bud_shard))
                    continue
                unit_rows = []
                unit_rel = []
                unit_budget = []
                tr = tr0[tr0[ycol].notna()].copy(); ca = ca0[ca0[ycol].notna()].copy()
                va = va0[va0[ycol].notna()].copy(); te = te0[te0[ycol].notna()].copy()
                if min(len(tr), len(va), len(te)) < 10 or len(ca) < 5:
                    continue
                ytr, yca, yva, yte = (x[ycol].to_numpy(float) for x in [tr,ca,va,te])
                protocol = sp.name.split(':')[0]
                target_site = sp.metadata.get('target_site')

                # Strong baselines.
                global_mean = float(np.mean(ytr))
                baseline_specs = {
                    'historical_mean': np.full(len(te), global_mean),
                    'user_history': te['user_prior_median_duration'].fillna(global_mean).to_numpy(float)
                        if target_name=='duration' else te['user_prior_mean_energy'].fillna(global_mean).to_numpy(float),
                    'driver_estimate': te['requested_duration_hours'].to_numpy(float)
                        if target_name=='duration' else te['requested_energy_kwh'].to_numpy(float),
                }
                for name,pred in baseline_specs.items():
                    valid = np.isfinite(pred) & np.isfinite(yte)
                    if name != 'driver_estimate':
                        valid[:] = True
                    if valid.sum() == 0:
                        continue
                    z = te.loc[valid, ['session_id','site','user_id','connection_time']].copy()
                    z['row_id'] = te.index.to_numpy()[valid]
                    z['date'] = z['connection_time'].dt.date.astype(str)
                    z['protocol'] = protocol; z['split_name'] = sp.name; z['target_site'] = target_site
                    z['seed'] = seed; z['target'] = target_name; z['model'] = name
                    z['y_true'] = yte[valid]; z['y_pred'] = pred[valid]
                    z['lower'] = np.nan; z['upper'] = np.nan; z['interval_method'] = None
                    unit_rows.append(z)

                # Learned models. Hyperparameters are tuned on train/validation only.
                # To keep the tuning budget equal and computationally tractable, a canonical seed tunes
                # each target/protocol/site once; the selected configuration is then frozen across five seeds.
                key = f'{target_name}|{sp.name}'
                if cfg.get('tuning', {}).get('enabled', False) and key not in tuned_cache:
                    n_trials = int(cfg['tuning']['trials_per_model'])
                    tune_seed = int(cfg['tuning']['canonical_seed'])
                    tuned = {}
                    if 'ridge' in learned:
                        tuned['ridge'] = tune_ridge(tr, ytr, va, yva, numeric, categorical, n_trials, tune_seed)
                    if 'catboost' in learned:
                        tuned['catboost'] = tune_catboost(tr, ytr, va, yva, numeric, categorical, cfg['models']['catboost'], n_trials, tune_seed)
                    if 'torch_ft' in learned:
                        tuned['torch_ft'] = tune_torch(tr, ytr, va, yva, numeric, categorical, cfg['models']['torch_ft'], n_trials, tune_seed, int(cfg['tuning']['torch_tuning_epochs']))
                    tuned_cache[key] = tuned
                defaults = {m: cfg['models'][m] for m in learned}
                hp = {**defaults, **tuned_cache.get(key, {})}
                fitted = []
                if 'ridge' in learned:
                    fitted.append(('ridge', fit_ridge(tr, ytr, numeric, categorical, float(hp['ridge']['alpha']))))
                if 'catboost' in learned:
                    fitted.append(('catboost', fit_catboost(tr, ytr, va, yva, numeric, categorical, hp['catboost'], seed)))
                if 'torch_ft' in learned:
                    fitted.append(('torch_ft', fit_torch_ft(tr, ytr, va, yva, numeric, categorical, hp['torch_ft'], seed)))

                for name, model in fitted:
                    pca = model.predict(ca); pte = model.predict(te)
                    qhat = split_conformal_calibrate(yca, pca, alpha)
                    lo, hi = split_conformal_interval(pte, qhat)
                    z = te[['session_id','site','user_id','connection_time']].copy()
                    z['row_id'] = te.index.to_numpy(); z['date'] = z['connection_time'].dt.date.astype(str)
                    z['protocol'] = protocol; z['split_name'] = sp.name; z['target_site'] = target_site
                    z['seed'] = seed; z['target'] = target_name; z['model'] = name
                    z['y_true'] = yte; z['y_pred'] = pte; z['lower'] = lo; z['upper'] = hi
                    z['alpha'] = alpha; z['interval_method'] = 'split_conformal'
                    unit_rows.append(z)

                    # Full reliability curve from same calibration scores.
                    scores = np.abs(yca-pca)
                    rel = []
                    for a in [.50,.40,.30,.20,.10,.05]:
                        from .conformal import finite_sample_quantile
                        q = finite_sample_quantile(scores, a); l,u = split_conformal_interval(pte,q)
                        met = interval_metrics(yte,l,u,a)
                        rel.append({'seed':seed,'target':target_name,'protocol':protocol,'split_name':sp.name,
                                    'target_site':target_site,'model':name,'alpha':a,**met})
                    unit_rel.append(pd.DataFrame(rel))

                    # Target-site data-efficiency curve: uncertainty adaptation only, model weights frozen.
                    if target_site and name == 'catboost':
                        budgets = cfg['conformal']['calibration_budgets']
                        for budget in budgets:
                            if budget == 0:
                                # Source validation residuals are the zero-shot calibration reference.
                                p_ref = model.predict(va)
                                q_b = split_conformal_calibrate(yva, p_ref, alpha)
                                n_used = 0
                            else:
                                n_used = min(int(budget), len(ca))
                                q_b = split_conformal_calibrate(yca[:n_used], pca[:n_used], alpha)
                            l,u = split_conformal_interval(pte,q_b)
                            unit_budget.append({'seed':seed,'target':target_name,'target_site':target_site,'budget':n_used,
                                                'nominal':1-alpha,**interval_metrics(yte,l,u,alpha)})

                # CQR on CatBoost; proper calibration is kept separate from model fitting.
                if use_cqr:
                    qlo = fit_catboost_quantile(tr,ytr,va,yva,numeric,categorical,hp['catboost'],seed,alpha/2)
                    qhi = fit_catboost_quantile(tr,ytr,va,yva,numeric,categorical,hp['catboost'],seed,1-alpha/2)
                    lo_ca, hi_ca = qlo.predict(ca), qhi.predict(ca)
                    lo_te, hi_te = qlo.predict(te), qhi.predict(te)
                    q_cqr = cqr_calibrate(yca,lo_ca,hi_ca,alpha)
                    lo,hi = cqr_interval(lo_te,hi_te,q_cqr)
                    mid = (lo_te+hi_te)/2
                    z = te[['session_id','site','user_id','connection_time']].copy()
                    z['row_id']=te.index.to_numpy(); z['date']=z.connection_time.dt.date.astype(str)
                    z['protocol']=protocol; z['split_name']=sp.name; z['target_site']=target_site; z['seed']=seed
                    z['target']=target_name; z['model']='catboost_cqr'; z['y_true']=yte; z['y_pred']=mid
                    z['lower']=lo; z['upper']=hi; z['alpha']=alpha; z['interval_method']='cqr'
                    unit_rows.append(z)

                # Flush this unit of work so a crash or restart never repeats it.
                if unit_rows:
                    unit_df = pd.concat(unit_rows, ignore_index=True)
                    unit_df.to_parquet(shard, index=False)
                    all_rows.append(unit_df)
                if unit_rel:
                    rel_df = pd.concat(unit_rel, ignore_index=True)
                    rel_df.to_parquet(_shard_path(Path(out_dir), seed, sp.name, target_name, 'rel'), index=False)
                    rel_frames.append(rel_df)
                if unit_budget:
                    bud_df = pd.DataFrame(unit_budget)
                    bud_df.to_parquet(_shard_path(Path(out_dir), seed, sp.name, target_name, 'budget'), index=False)
                    budget_rows.append(bud_df)

    _save_predictions(all_rows, Path(out_dir)/'regression_predictions.parquet')
    if rel_frames:
        pd.concat(rel_frames, ignore_index=True).to_csv('results/metrics/conformal_reliability.csv', index=False)
    if budget_rows:
        pd.concat(budget_rows, ignore_index=True).to_csv('results/metrics/calibration_budget.csv', index=False)
    if tuned_cache:
        Path('results/metrics/tuned_hyperparameters.json').write_text(json.dumps(tuned_cache, indent=2, default=str), encoding='utf-8')


def _block_metric_ci(g: pd.DataFrame, n_boot: int, seed: int, confidence: float) -> tuple[float,float]:
    return block_metric_ci(g, 'mae', n_boot, seed, confidence)


def _write_rq3_driver_matched_metrics(p: pd.DataFrame) -> None:
    """Evaluate every comparator on exactly the rows where a driver estimate exists.

    The ordinary headline table remains an all-available-row benchmark. This
    dedicated RQ3 table prevents the driver baseline from being compared with
    learned models evaluated on a different row population. Pairwise bootstrap
    inference later also intersects row_id, giving a second matched-row check.
    """
    rows = []
    keys = ['target','protocol','split_name','target_site','seed']
    for group_key, g in p.groupby(keys, dropna=False):
        target, protocol, split_name, target_site, seed = group_key
        driver = g[g.model == 'driver_estimate']
        matched_ids = set(driver.row_id.unique())
        all_ids = set(g.row_id.unique())
        if not matched_ids:
            continue
        for model, gm in g[g.row_id.isin(matched_ids)].groupby('model'):
            # Require exactly the common driver-available evaluation population.
            gm = gm.drop_duplicates('row_id')
            if set(gm.row_id) != matched_ids:
                continue
            met = regression_metrics(gm.y_true, gm.y_pred)
            rows.append({
                'target': target, 'protocol': protocol, 'split_name': split_name,
                'target_site': target_site, 'seed': seed, 'model': model,
                'n_matched': len(matched_ids), 'n_test_population': len(all_ids),
                'driver_availability_fraction': len(matched_ids) / max(1, len(all_ids)),
                **met,
            })
    if not rows:
        return
    out = pd.DataFrame(rows)
    out.to_csv('results/metrics/rq3_driver_matched_metrics_by_seed.csv', index=False)
    metric_cols = ['mae','rmse','medae','p90_abs_error','p95_abs_error','r2','driver_availability_fraction']
    summary = (out.groupby(['target','protocol','split_name','target_site','model'], dropna=False)[metric_cols]
                  .agg(['mean','std']).reset_index())
    summary.columns = ['_'.join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv('results/metrics/rq3_driver_matched_metrics_summary.csv', index=False)


def _write_pandemic_metrics(p: pd.DataFrame, cfg: dict) -> None:
    """Report pre/post-pandemic performance instead of showing only a timeline marker."""
    tz = cfg.get('data', {}).get('timezone', 'America/Los_Angeles')
    break_ts = pd.Timestamp(cfg['splits']['pandemic_break'], tz=tz)
    z = p.copy()
    z['connection_time'] = pd.to_datetime(z['connection_time'])
    z['pandemic_period'] = np.where(z['connection_time'] < break_ts, 'pre_break', 'post_break')
    rows = []
    cols = ['target','protocol','split_name','target_site','seed','model','pandemic_period']
    for keys, g in z.groupby(cols, dropna=False):
        if len(g) < 20:
            continue
        row = dict(zip(cols, keys))
        row.update(regression_metrics(g.y_true, g.y_pred))
        row['n'] = len(g)
        rows.append(row)
    if not rows:
        return
    out = pd.DataFrame(rows)
    out.to_csv('results/metrics/pandemic_stratified_metrics_by_seed.csv', index=False)

    summary = (out.groupby(['target','protocol','split_name','target_site','model','pandemic_period'], dropna=False)
                  .agg(mae_mean=('mae','mean'), mae_sd=('mae','std'),
                       rmse_mean=('rmse','mean'), n_mean=('n','mean')).reset_index())
    summary.to_csv('results/metrics/pandemic_stratified_metrics_summary.csv', index=False)

    # target_site is empty for non-site protocols; pivot_table would drop those rows silently
    summary = summary.assign(target_site=summary['target_site'].fillna('all'))
    piv = summary.pivot_table(
        index=['target','protocol','split_name','target_site','model'],
        columns='pandemic_period', values='mae_mean', aggfunc='first'
    ).reset_index()
    if {'pre_break','post_break'}.issubset(piv.columns):
        piv['mae_post_minus_pre'] = piv['post_break'] - piv['pre_break']
        piv['mae_relative_change_pct'] = 100 * piv['mae_post_minus_pre'] / piv['pre_break'].replace(0, np.nan)
        piv.to_csv('results/metrics/pandemic_pre_post_delta.csv', index=False)


def _write_identified_user_metrics(p: pd.DataFrame) -> None:
    """Like-for-like protocol comparison on sessions with a stable user id.

    The user-disjoint protocol excludes anonymous sessions by construction, so a
    raw protocol-to-protocol comparison would mix different row populations.
    """
    z = p[p['user_id'].notna() & (p['user_id'].astype(str).str.strip() != '')]
    if z.empty:
        return
    rows = []
    for keys, g in z.groupby(['target','protocol','split_name','target_site','seed','model'], dropna=False):
        if len(g) < 20:
            continue
        row = dict(zip(['target','protocol','split_name','target_site','seed','model'], keys))
        row.update(regression_metrics(g.y_true, g.y_pred))
        row['n'] = len(g)
        rows.append(row)
    if not rows:
        return
    out = pd.DataFrame(rows)
    out.to_csv('results/metrics/identified_user_metrics_by_seed.csv', index=False)
    core = out.groupby(['target','protocol','model'], as_index=False)['mae'].mean()
    protocol_optimism(core).to_csv('results/metrics/protocol_optimism_identified_users.csv', index=False)


def _write_cross_site_user_overlap(df: pd.DataFrame, cfg: dict) -> None:
    """How much user history can transfer in the site-disjoint protocol.

    Station history is zeroed at an unseen site, but a driver seen at a source
    site keeps their history. This quantifies that channel.
    """
    z = df[df['user_id'].notna() & (df['user_id'].astype(str).str.strip() != '')]
    if z.empty:
        return
    rows = []
    for target in sorted(df['site'].dropna().unique()):
        tgt = z[z.site == target]
        src_users = set(z[z.site != target]['user_id'].astype(str))
        if tgt.empty:
            continue
        seen = tgt['user_id'].astype(str).isin(src_users)
        rows.append({'target_site': target,
                     'identified_sessions': int(len(tgt)),
                     'sessions_with_user_seen_at_source': int(seen.sum()),
                     'fraction_user_history_transferable': float(seen.mean()),
                     'unique_target_users': int(tgt['user_id'].nunique()),
                     'unique_target_users_seen_at_source': int(tgt.loc[seen, 'user_id'].nunique())})
    if rows:
        pd.DataFrame(rows).to_csv('results/metrics/cross_site_user_overlap.csv', index=False)


def _add_skill_scores(metrics: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample skill relative to the training-mean forecast on the same rows.

    sklearn's r2 compares a model with the mean of the TEST set, a forecast no
    deployed system can know. Under shift that baseline is unrealistically strong
    and r2 can be negative even when the model clearly beats every feasible
    baseline. r2_oos uses the historical-mean model (trained on the training
    block) as the reference, as is standard for forecast evaluation:
        r2_oos    = 1 - SSE(model) / SSE(training mean)
        mae_skill = 1 - MAE(model) / MAE(training mean)
    """
    keys = ['target', 'protocol', 'split_name', 'target_site', 'seed']
    ref = p[p['model'] == 'historical_mean'][keys + ['row_id', 'y_true', 'y_pred']].rename(columns={'y_pred': 'ref'})
    if ref.empty:
        return metrics.assign(r2_oos=np.nan, mae_skill=np.nan)
    out = []
    for k, g in p[p['model'] != 'historical_mean'].groupby(keys + ['model'], dropna=False):
        r = ref
        for col, val in zip(keys, k[:-1]):
            r = r[r[col].isna()] if pd.isna(val) else r[r[col] == val]
        m = g.merge(r[['row_id', 'ref']], on='row_id', how='inner')
        if len(m) < 20:
            continue
        sse_m = np.sum((m['y_true'] - m['y_pred']) ** 2); sse_r = np.sum((m['y_true'] - m['ref']) ** 2)
        mae_m = np.mean(np.abs(m['y_true'] - m['y_pred'])); mae_r = np.mean(np.abs(m['y_true'] - m['ref']))
        out.append(dict(zip(keys + ['model'], k), r2_oos=1 - sse_m / sse_r, mae_skill=1 - mae_m / mae_r))
    sk = pd.DataFrame(out)
    if sk.empty:
        return metrics.assign(r2_oos=np.nan, mae_skill=np.nan)
    merged = metrics.merge(sk, on=keys + ['model'], how='left')
    merged.loc[merged['model'] == 'historical_mean', ['r2_oos', 'mae_skill']] = 0.0
    return merged


def analyze_regression(cfg: dict, pred_path='results/predictions/regression_predictions.parquet') -> pd.DataFrame:
    p = pd.read_parquet(pred_path)
    rows=[]
    group_cols=['target','protocol','split_name','target_site','seed','model']
    for keys,g in p.groupby(group_cols, dropna=False):
        target, protocol, split_name, target_site, seed, model = keys
        met = regression_metrics(g.y_true,g.y_pred)
        ci_lo,ci_hi = _block_metric_ci(g,cfg['bootstrap']['n_boot'],int(seed)+17,cfg['bootstrap']['confidence'])
        row={'target':target,'protocol':protocol,'split_name':split_name,'target_site':target_site,'seed':seed,'model':model,
             **met,'mae_ci_low':ci_lo,'mae_ci_high':ci_hi,'n':len(g)}
        gi=g[g.lower.notna() & g.upper.notna()]
        if len(gi): row.update({f'interval_{k}':v for k,v in interval_metrics(gi.y_true,gi.lower,gi.upper,float(gi.alpha.dropna().iloc[0])).items()})
        rows.append(row)
    metrics=pd.DataFrame(rows)
    metrics=_add_skill_scores(metrics, p)
    metrics.to_csv('results/metrics/regression_metrics_by_seed.csv',index=False)
    # Seed-aggregated headline table.
    agg_cols=['mae','rmse','medae','p90_abs_error','p95_abs_error','r2','r2_oos','mae_skill']
    agg=metrics.groupby(['target','protocol','split_name','target_site','model'],dropna=False)[agg_cols].agg(['mean','std']).reset_index()
    agg.columns=['_'.join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in agg.columns]
    agg.to_csv('results/metrics/regression_metrics_summary.csv',index=False)

    # RQ3 fair comparison: every model is re-scored on the identical driver-available rows.
    _write_rq3_driver_matched_metrics(p)
    # Explicit pandemic sensitivity metrics (not only a vertical line in a figure).
    _write_pandemic_metrics(p, cfg)
    # Protocol comparisons restricted to identified users, since the user-disjoint
    # protocol cannot contain anonymous sessions.
    _write_identified_user_metrics(p)

    # Protocol optimism and rank stability use mean across seeds and pool site-disjoint targets by macro mean.
    core=metrics.groupby(['target','protocol','model'],as_index=False)['mae'].mean()
    protocol_optimism(core).to_csv('results/metrics/protocol_optimism.csv',index=False)
    rank_stability(core).to_csv('results/metrics/rank_stability.csv',index=False)
    return metrics
