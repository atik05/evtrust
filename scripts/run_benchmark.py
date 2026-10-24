"""Repeated cross-validated benchmark of twelve learners under four protocols.

    python scripts/run_benchmark.py --config configs/acn.yaml

Resumable: one shard per (target, protocol, model, repeat, fold) under results/benchmark/shards/.
Outputs: results/benchmark/{fold_scores,summary,pairwise_vs_best,friedman}.csv,
results/latex/benchmark_<target>.tex, figures/benchmark/*.pdf and *.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evtrust import benchmark, paper_figures      # noqa: E402
from evtrust.config import load_config             # noqa: E402

TARGETS = ['kwh_delivered', 'duration_hours']
NAME = {'kwh_delivered': 'energy (kWh)', 'duration_hours': 'duration (h)'}


def latex_table(summ: pd.DataFrame, pairs: pd.DataFrame, target: str, path: Path, folds: int, repeats: int):
    protos = [p for p in ['random', 'driver_grouped', 'chronological', 'chronological_prepandemic', 'site_transfer',
                          'site_transfer_strict'] if p in set(summ['protocol'])]
    head = {'random': 'Random', 'driver_grouped': 'Unseen drivers', 'chronological': 'Chronological',
            'site_transfer': 'Leave site out', 'site_transfer_strict': 'Leave site out (strict)',
            'chronological_prepandemic': 'Chronological, pre-2020'}
    z = summ[summ['target'] == target]
    order = z[z['protocol'] == protos[0]].sort_values('mae')['model'].tolist()
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             rf'\caption{{MAE for session {NAME[target]} over {folds}-fold cross-validation repeated {repeats} times '
             r'(mean $\pm$ SD across folds). Bold: best model; $\dagger$: not significantly worse than the best '
             r'(Nadeau--Bengio corrected t-test, Holm-adjusted $p \geq 0.05$).}',
             rf'\label{{tab:benchmark_{target}}}', r'\begin{tabular}{l' + 'r' * len(protos) + '}', r'\toprule',
             'Model & ' + ' & '.join(head[p] for p in protos) + r' \\', r'\midrule']
    for m in order:
        cells = []
        for p in protos:
            v = z[(z['protocol'] == p) & (z['model'] == m)]
            if v.empty:
                cells.append('--'); continue
            txt = f'{v["mae"].iloc[0]:.3f} $\\pm$ {v["mae_sd"].iloc[0]:.3f}'
            pr = pairs[(pairs['target'] == target) & (pairs['protocol'] == p)]
            if not pr.empty and pr['best_model'].iloc[0] == m:
                txt = r'\textbf{' + txt + '}'
            elif not pr[(pr['model'] == m) & (~pr['significantly_worse'])].empty:
                txt += r'$^\dagger$'
            cells.append(txt)
        lines.append(benchmark.DISPLAY.get(m, m) + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/acn.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    bc = cfg.get('benchmark', {})
    models = bc.get('models', benchmark.DEFAULT_MODELS)
    protocols = bc.get('protocols', ['random', 'driver_grouped', 'chronological', 'site_transfer'])
    repeats, folds = int(bc.get('repeats', 5)), int(bc.get('folds', 5))
    run_cfg = {'gpu': bool(bc.get('gpu', False)), 'catboost': dict(cfg['models']['catboost']),
               'deep': bc.get('deep', {}), 'benchmark_models': bc.get('model_params', {})}
    numeric = list(cfg['features']['numeric'])
    df = pd.read_parquet(Path(cfg['data'].get('processed_dir', 'data/processed')) / 'sessions.parquet')
    out = Path('results/benchmark'); shards = out / 'shards'; shards.mkdir(parents=True, exist_ok=True)
    figs = Path('figures/benchmark'); tex = Path('results/latex'); tex.mkdir(parents=True, exist_ok=True)

    scores = []
    protocol_models = bc.get('protocol_models', {})
    for target in TARGETS:
        for protocol in protocols:
            ms = protocol_models.get(protocol, models)          # optional smaller model set per protocol
            scores.append(benchmark.run(df, numeric, target, protocol, ms, run_cfg, repeats, folds, shards,
                                        log=lambda m: print(m, flush=True)))
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(out / 'fold_scores.csv', index=False)
    failed = scores[scores['status'] != 'ok']
    if len(failed):
        failed.groupby(['model', 'status']).size().rename('fits').reset_index().to_csv(out / 'failed_models.csv', index=False)
        print(f'[benchmark] {failed["model"].nunique()} model(s) unavailable or failed; see failed_models.csv')
    summ, pairs, fried = benchmark.summarise(scores)
    summ.to_csv(out / 'summary.csv', index=False)
    pairs.to_csv(out / 'pairwise_vs_best.csv', index=False)
    fried.to_csv(out / 'friedman.csv', index=False)

    paper_figures.style()
    for target in TARGETS:
        paper_figures.benchmark_protocols(summ, figs, target)
        for protocol in protocols:
            paper_figures.cd_diagram(fried, figs, target, protocol)
        latex_table(summ, pairs, target, tex / f'benchmark_{target}.tex', folds, repeats)
    lines = ['# Benchmark results', '', summ.round(3).to_markdown(index=False), '', '## Versus best model', '',
             pairs.round(4).to_markdown(index=False), '', '## Friedman and Nemenyi', '', fried.round(3).to_markdown(index=False)]
    (out / 'BENCHMARK_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'benchmark complete: {out}, {figs}')


if __name__ == '__main__':
    main()
