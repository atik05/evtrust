"""Collect every result into one readable report.

    python scripts/full_report.py                 # Markdown + Excel into results/
    python scripts/full_report.py --format md     # Markdown only

Works after the run completes. If the analysis tables are not written yet, it
falls back to the prediction shards and reports preliminary numbers, clearly
labelled as such.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import re

import pandas as pd

M = Path('results/metrics')
OUT = Path('results')

ORDER = [
    ('data_flow.csv', 'Data flow (cleaning)', 'How many sessions survived each filter.'),
    ('regression_metrics_summary.csv', 'RQ1 Accuracy by protocol', 'MAE, RMSE and R2 averaged over seeds.'),
    ('regression_metrics_by_seed.csv', 'RQ1 Accuracy per seed', 'Seed-level values behind the summary.'),
    ('protocol_optimism.csv', 'RQ1 Optimism of the random split', 'Relative change versus the random protocol.'),
    ('identified_user_metrics_by_seed.csv', 'RQ1 Identified users only', 'Like-for-like comparison across protocols.'),
    ('protocol_optimism_identified_users.csv', 'RQ1 Optimism, identified users', 'Optimism on a common population.'),
    ('rank_stability.csv', 'RQ1 Model ranking stability', 'Kendall agreement between protocols.'),
    ('conformal_reliability.csv', 'RQ2 Conformal coverage', 'Coverage and width at several nominal levels.'),
    ('calibration_budget.csv', 'RQ2 Target-site recalibration', 'Coverage against the number of target labels.'),
    ('rq3_driver_matched_metrics_summary.csv', 'RQ3 Models versus drivers', 'Identical rows for every comparator.'),
    ('rq3_driver_matched_metrics_by_seed.csv', 'RQ3 per seed', 'Seed-level matched comparison.'),
    ('occupancy_metrics.csv', 'RQ4 Occupancy prediction', 'Discrimination and calibration per horizon.'),
    ('paired_model_comparisons.csv', 'Statistical comparisons', 'Day-block bootstrap with Holm correction.'),
    ('distribution_shift.csv', 'Distribution shift', 'How far each test block moves from training.'),
    ('pandemic_stratified_metrics_summary.csv', 'Pandemic stratification', 'Metrics before and after the 2020 break.'),
    ('pandemic_pre_post_delta.csv', 'Pandemic delta', 'Change attributable to the break.'),
    ('cross_site_user_overlap.csv', 'Cross-site user history', 'How much history can transfer to an unseen site.'),
    ('site_transfer_summary.csv', 'Site transfer', 'Source to target performance.'),
    ('contribution_accuracy.csv', 'Remedy R1: gated ensemble (HRGE)', 'MAE of HRGE and its two components per protocol and history group.'),
    ('contribution_gate.csv', 'Remedy R1: gate weights', 'Blending weight chosen per driver-history group on the validation block.'),
    ('contribution_coverage.csv', 'Remedy R2: history-conditional coverage', 'Marginal versus history-conditional calibration per history group.'),
    ('published_style.csv', 'Published-style features', 'Leakage-aware versus whole-dataset user aggregates, random protocol.'),
    ('grouped_ablation.csv', 'Grouped feature ablation', 'Effect of removing each feature group.'),
    ('grouped_ablation_seeds_summary.csv', 'Grouped ablation, five seeds', 'Ablation repeated over the five seeds: mean and SD of the change in MAE.'),
    ('conditional_coverage.csv', 'Conditional coverage', 'Coverage by driver history and time of day.'),
    ('tuned_hyperparameters.json', 'Tuned hyperparameters', 'Frozen configuration per target and protocol.'),
    ('squared_error_summary.csv', 'R2 for the squared-error model', 'R2, out-of-sample R2 and RMSE for the forecast that is optimal for them.'),
    ('../paper/coldstart_decomposition.csv', 'Cold-start decomposition', 'Leakage, unseen-driver cost and value of own history on identical rows.'),
    ('../paper/coldstart_warmup.csv', 'Cold-start warm-up curve', 'Error by the number of completed sessions of the driver.'),
    ('../paper/feature_availability.csv', 'Information availability', 'Plug-in versus charging-duration versus session-duration information.'),
    ('../paper/impact_matched_reliability.csv', 'Smart charging at matched reliability', 'Flexibility retained at 1, 2 and 5 percent undelivered energy.'),
    ('../paper/impact_by_history.csv', 'Smart charging by driver history', 'Undelivered energy and flexibility by history group.'),
    ('../paper/impact_tradeoff.csv', 'Smart charging trade-off', 'Undelivered energy against flexibility across quantile levels.'),
    ('../benchmark/summary.csv', 'Benchmark summary', 'Twelve learners, six protocols, repeated cross-validation.'),
    ('../benchmark/pairwise_vs_best.csv', 'Benchmark pairwise tests', 'Corrected resampled t-tests against the best learner, Holm-adjusted.'),
    ('../benchmark/friedman.csv', 'Benchmark Friedman tests', 'Friedman tests and Nemenyi critical differences.'),
    ('../benchmark/failed_models.csv', 'Benchmark failures', 'Learner fits that failed, if any.'),
    ('../operational/request_matched.csv', 'Driver request vs forecasts', 'Flexibility at matched reliability on sessions with a driver request.'),
    ('../operational/request_differences.csv', 'Driver request differences', 'Request minus forecast at matched reliability, driver-cluster bootstrap.'),
    ('../operational/site_capacity_needed.csv', 'Site capacity needed', 'Capacity needed to keep undelivered energy at 1 or 2 percent, day-block bootstrap.'),
    ('../operational/site_level.csv', 'Site-level scheduling', 'Undelivered energy by policy and capacity limit.'),
    ('../operational/bias_correction.csv', 'Target-site bias correction', 'Unseen-site accuracy after correcting with the first m sessions.'),
]


def _preliminary_from_shards() -> dict[str, pd.DataFrame]:
    files = sorted(glob.glob('results/predictions/shards/pred__*.parquet'))
    if not files:
        return {}
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d['abs_error'] = (d['y_true'] - d['y_pred']).abs()
    acc = (d.groupby(['target', 'protocol', 'model'])
             .agg(mae=('abs_error', 'mean'), n=('abs_error', 'size'),
                  seeds=('seed', 'nunique')).reset_index())
    piv = acc.pivot_table(index=['target', 'model'], columns='protocol', values='mae').reset_index()
    if 'random' in piv.columns:
        for c in [c for c in piv.columns if c not in ('target', 'model', 'random')]:
            piv[f'{c}_vs_random_pct'] = (100 * (piv[c] - piv['random']) / piv['random'])
    iv = d[d['lower'].notna()].copy()
    cov = pd.DataFrame()
    if not iv.empty:
        iv['covered'] = (iv['y_true'] >= iv['lower']) & (iv['y_true'] <= iv['upper'])
        iv['width'] = iv['upper'] - iv['lower']
        cov = (iv.groupby(['target', 'protocol', 'interval_method'])
                 .agg(coverage=('covered', 'mean'), mean_width=('width', 'mean'),
                      n=('covered', 'size')).reset_index())
    return {'units': pd.DataFrame({'completed_units': [len(files)], 'rows': [len(d)],
                                   'seeds': [d.seed.nunique()]}),
            'accuracy': acc, 'accuracy_pivot': piv, 'coverage': cov}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', choices=['md', 'xlsx', 'both'], default='both')
    args = ap.parse_args()

    tables: list[tuple[str, str, pd.DataFrame]] = []
    missing: list[str] = []
    for name, title, note in ORDER:
        p = M / name
        if not p.exists():
            missing.append(name)
            continue
        if p.suffix == '.json':
            import json
            flat = pd.json_normalize(json.loads(p.read_text())).T.reset_index()
            flat.columns = ['setting', 'value']
            tables.append((title, note, flat))
        else:
            tables.append((title, note, pd.read_csv(p)))

    prelim = _preliminary_from_shards() if len(tables) < 5 else {}

    lines = ['# EVTrust-X: full result report', '',
             f'Generated from `results/` in `{Path.cwd()}`.', '']
    if prelim:
        u = prelim['units'].iloc[0]
        lines += ['> **Preliminary.** The analysis stage has not produced its tables yet, so the '
                  f'numbers below come directly from {int(u.completed_units)} completed prediction '
                  f'units ({int(u.rows):,} rows, {int(u.seeds)} seeds). They are pooled over seeds '
                  'without confidence intervals or significance tests, and will change when the run '
                  'finishes.', '']
        for key, title in [('accuracy_pivot', 'Mean absolute error by protocol'),
                           ('accuracy', 'Mean absolute error, long form'),
                           ('coverage', 'Conformal coverage by protocol')]:
            t = prelim.get(key)
            if t is not None and not t.empty:
                lines += [f'## {title}', '', t.round(3).to_markdown(index=False), '']
    for title, note, t in tables:
        lines += [f'## {title}', '', f'_{note}_', '', t.round(4).to_markdown(index=False), '']
    if missing:
        lines += ['## Tables not present yet', '',
                  'These are written by later stages of the pipeline:', '']
        lines += [f'- `{m}`' for m in missing] + ['']

    md = OUT / 'FULL_REPORT.md'
    if args.format in ('md', 'both'):
        md.write_text('\n'.join(lines), encoding='utf-8')
        print(f'Markdown report: {md}')

    if args.format in ('xlsx', 'both'):
        xlsx = OUT / 'FULL_REPORT.xlsx'
        try:
            with pd.ExcelWriter(xlsx) as writer:
                for key, t in prelim.items():
                    t.to_excel(writer, sheet_name=f'prelim_{key}'[:31], index=False)
                used = set()
                for title, _, t in tables:
                    sheet = re.sub(r'[\[\]:*?/\\]', '-', title)[:28]   # Excel forbids [ ] : * ? / \
                    i = 1
                    while sheet in used:
                        sheet = f"{re.sub(r'[\[\]:*?/\\]', '-', title)[:26]}_{i}"; i += 1
                    used.add(sheet)
                    t.to_excel(writer, sheet_name=sheet, index=False)
            print(f'Excel workbook: {xlsx}')
        except Exception as exc:
            print(f'Excel export skipped ({exc!r}); the Markdown report is complete.')

    print(f'{len(tables)} tables included, {len(missing)} still pending.')


if __name__ == '__main__':
    main()
