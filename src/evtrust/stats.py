from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wasserstein_distance
from statsmodels.stats.multitest import multipletests


def _day_block_draws(sum_a: np.ndarray, sum_b: np.ndarray | None, counts: np.ndarray,
                     n_boot: int, seed: int, metric: str) -> np.ndarray:
    """Resample calendar days with replacement using precomputed per-day sums.

    Mathematically identical to concatenating the sampled days and recomputing
    the metric, but without rebuilding a DataFrame for every draw.
    """
    rng = np.random.default_rng(seed)
    n_days = len(counts)
    idx = rng.integers(0, n_days, size=(n_boot, n_days))
    n = counts[idx].sum(axis=1)
    a = sum_a[idx].sum(axis=1) / n
    if metric == 'rmse':
        a = np.sqrt(a)
    if sum_b is None:
        return a
    b = sum_b[idx].sum(axis=1) / n
    if metric == 'rmse':
        b = np.sqrt(b)
    return a - b


def _per_day_terms(df: pd.DataFrame, truth: str, pred: str, metric: str) -> pd.DataFrame:
    err = np.abs(df[truth].to_numpy(float) - df[pred].to_numpy(float))
    if metric == 'rmse':
        err = err ** 2
    elif metric != 'mae':
        raise ValueError(metric)
    z = pd.DataFrame({'date': df['date'].to_numpy(), 'term': err})
    return z.groupby('date', sort=True).agg(total=('term', 'sum'), n=('term', 'size'))


def block_metric_ci(df: pd.DataFrame, metric: str = 'mae', n_boot: int = 3000, seed: int = 123,
                    confidence: float = .95) -> tuple[float, float]:
    """Day-block bootstrap confidence interval for a single model's metric."""
    days = _per_day_terms(df, 'y_true', 'y_pred', metric)
    if len(days) < 2:
        return float('nan'), float('nan')
    draws = _day_block_draws(days['total'].to_numpy(float), None, days['n'].to_numpy(float),
                             n_boot, seed, metric)
    q = (1 - confidence) / 2
    return float(np.quantile(draws, q)), float(np.quantile(draws, 1 - q))


def paired_day_block_bootstrap(pred_df: pd.DataFrame, model_a: str, model_b: str, metric='mae', n_boot=3000,
                               seed=123, confidence=.95) -> dict:
    """Paired bootstrap of the metric difference A minus B using calendar-day blocks."""
    d = pred_df[pred_df['model'].isin([model_a, model_b])]
    a = d[d.model == model_a][['date', 'row_id', 'y_true', 'y_pred']].rename(columns={'y_pred': 'pred_a'})
    b = d[d.model == model_b][['date', 'row_id', 'y_pred']].rename(columns={'y_pred': 'pred_b'})
    m = a.merge(b, on=['date', 'row_id'], how='inner')
    if m.empty:
        return {'model_a': model_a, 'model_b': model_b, 'metric': metric,
                'difference_a_minus_b': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'),
                'p_a_better': float('nan'), 'n_days': 0, 'n_boot': int(n_boot)}
    da = _per_day_terms(m, 'y_true', 'pred_a', metric)
    db = _per_day_terms(m, 'y_true', 'pred_b', metric)
    days = da.join(db, lsuffix='_a', rsuffix='_b')
    diffs = _day_block_draws(days['total_a'].to_numpy(float), days['total_b'].to_numpy(float),
                             days['n_a'].to_numpy(float), n_boot, seed, metric)
    q = (1 - confidence) / 2
    return {
        'model_a': model_a, 'model_b': model_b, 'metric': metric,
        'difference_a_minus_b': float(np.mean(diffs)),
        'ci_low': float(np.quantile(diffs, q)),
        'ci_high': float(np.quantile(diffs, 1 - q)),
        'p_a_better': float(np.mean(diffs < 0)),
        # two-sided bootstrap p-value with the finite-sample correction of
        # Davison and Hinkley (1997): (k + 1) / (B + 1), so it is never exactly zero
        'p_two_sided': float(min(1.0, 2 * (min(int(np.sum(diffs <= 0)), int(np.sum(diffs >= 0))) + 1)
                                  / (len(diffs) + 1))),
        'n_days': int(len(days)), 'n_boot': int(n_boot)
    }


def protocol_optimism(metrics_df: pd.DataFrame, metric='mae') -> pd.DataFrame:
    base = metrics_df[metrics_df.protocol == 'random'][['target','model',metric]].rename(columns={metric:f'{metric}_random'})
    m = metrics_df.merge(base, on=['target','model'], how='left')
    m['absolute_optimism_gap'] = m[metric] - m[f'{metric}_random']
    m['relative_optimism_pct'] = 100 * m['absolute_optimism_gap'] / m[f'{metric}_random']
    return m


def rank_stability(metrics_df: pd.DataFrame, metric='mae') -> pd.DataFrame:
    rows = []
    for target, g in metrics_df.groupby('target'):
        protocols = sorted(g.protocol.unique())
        ranks = {p: g[g.protocol==p].set_index('model')[metric].rank(method='average') for p in protocols}
        for i,p1 in enumerate(protocols):
            for p2 in protocols[i+1:]:
                common = ranks[p1].index.intersection(ranks[p2].index)
                if len(common) >= 2:
                    tau, pval = kendalltau(ranks[p1].loc[common], ranks[p2].loc[common])
                    rows.append({'target':target,'protocol_a':p1,'protocol_b':p2,'kendall_tau':tau,'p_value':pval,'n_models':len(common)})
    return pd.DataFrame(rows)


def holm_adjust(p_values) -> np.ndarray:
    return multipletests(np.asarray(p_values, float), method='holm')[1]


def continuous_shift(train: pd.Series, test: pd.Series) -> float:
    a = pd.to_numeric(train, errors='coerce').dropna().to_numpy()
    b = pd.to_numeric(test, errors='coerce').dropna().to_numpy()
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else float('nan')


def js_divergence(train: pd.Series, test: pd.Series, eps=1e-12) -> float:
    cats = sorted(set(train.dropna().astype(str)) | set(test.dropna().astype(str)))
    p = train.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(float)
    q = test.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(float)
    m = .5*(p+q)
    kl_pm = np.sum(np.where(p>0, p*np.log((p+eps)/(m+eps)), 0))
    kl_qm = np.sum(np.where(q>0, q*np.log((q+eps)/(m+eps)), 0))
    return float(.5*(kl_pm+kl_qm))
