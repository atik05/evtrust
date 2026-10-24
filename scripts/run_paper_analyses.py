"""Paper analyses: cold-start decomposition, feature availability, smart-charging impact.

    python scripts/run_paper_analyses.py --config configs/acn.yaml

Resumable: one shard per (learner, target, seed) under results/paper/shards/.
Outputs: results/paper/*.csv, results/paper/PAPER_RESULTS.md, results/latex/paper_*.tex,
figures/paper/*.pdf and *.png (600 dpi).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evtrust import availability, coldstart, impact, paper_figures  # noqa: E402
from evtrust.config import load_config                               # noqa: E402

TARGETS = ['kwh_delivered', 'duration_hours']


def _latex(df: pd.DataFrame, caption: str, label: str, path: Path):
    cols = list(df.columns)
    esc = lambda v: str(v).replace('_', r'\_').replace('%', r'\%')
    lines = [r'\begin{table}[t]', r'\centering', r'\small', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\begin{tabular}{' + 'l' * 2 + 'r' * (len(cols) - 2) + '}', r'\toprule',
             ' & '.join(esc(c) for c in cols) + r' \\', r'\midrule']
    lines += [' & '.join(esc(v) for v in r) + r' \\' for r in df.itertuples(index=False)]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/acn.yaml')
    ap.add_argument('--learners', default=None, help='comma list from catboost, ft, hgb')
    ap.add_argument('--seeds', type=int, default=None, help='limit number of seeds')
    args = ap.parse_args()

    cfg = load_config(args.config)
    pc = cfg.get('paper', {})
    learners = (args.learners.split(',') if args.learners else pc.get('learners', ['catboost', 'ft']))
    seeds = list(cfg['project']['five_seeds'])[: args.seeds] if args.seeds else list(cfg['project']['five_seeds'])
    end = pc.get('decomposition_end', '2020-03-01')
    n_boot = int(pc.get('n_boot', 2000))
    power = float(pc.get('charging_power_kw', 6.6))
    numeric = [c for c in cfg['features']['numeric']]
    cat_params = {k: v for k, v in cfg['models']['catboost'].items()
                  if k not in ('early_stopping_rounds', 'use_best_model', 'eval_metric')}
    cat_params['verbose'] = 0
    ft = cfg['models'].get('torch_ft', {})
    learner_cfg = {'catboost': cat_params,
                   'ft': {'epochs': int(pc.get('ft_epochs', 40)), 'batch': ft.get('batch_size', 1024),
                          'lr': ft.get('lr', 8e-4), 'patience': int(pc.get('ft_patience', 6))}}

    df = pd.read_parquet(Path(cfg['data'].get('processed_dir', 'data/processed')) / 'sessions.parquet')
    out = Path('results/paper'); shards = out / 'shards'; shards.mkdir(parents=True, exist_ok=True)
    figs = Path('figures/paper'); tex = Path('results/latex'); tex.mkdir(parents=True, exist_ok=True)

    # 1. cold-start decomposition
    for learner in learners:
        for target in TARGETS:
            for seed in seeds:
                f = shards / f'cs_{learner}_{target}_{seed}.parquet'
                if f.exists():
                    continue
                t0 = time.time()
                coldstart.run_unit(df, target, learner, seed, numeric, learner_cfg, end,
                                   with_published=(learner != 'ft')).to_parquet(f, index=False)
                print(f'[coldstart] {learner} {target} seed {seed} done in {(time.time()-t0)/60:.1f} min', flush=True)
    rows = pd.concat([pd.read_parquet(f) for f in sorted(shards.glob('cs_*.parquet'))], ignore_index=True)
    dec, warm = coldstart.summarise(rows, n_boot)
    dec.to_csv(out / 'coldstart_decomposition.csv', index=False)
    warm.to_csv(out / 'coldstart_warmup.csv', index=False)

    # 2. smart-charging impact, per learner
    tr_all, bk_all, mt_all = [], [], []
    for learner in sorted(rows['learner'].unique()):
        r = rows[rows['learner'] == learner]
        summ, bk, matched, _ = impact.run(r[r['target'] == 'duration_hours'], r[r['target'] == 'kwh_delivered'], power)
        for t, lst in [(summ, tr_all), (bk, bk_all), (matched, mt_all)]:
            if t is not None and len(t):
                lst.append(t.assign(learner=learner))
    trade = pd.concat(tr_all, ignore_index=True); trade.to_csv(out / 'impact_tradeoff.csv', index=False)
    byk = pd.concat(bk_all, ignore_index=True); byk.to_csv(out / 'impact_by_history.csv', index=False)
    matched = pd.concat(mt_all, ignore_index=True); matched.to_csv(out / 'impact_matched_reliability.csv', index=False)

    # 3. feature availability (gradient boosting learner of the main study)
    av_learner = pc.get('availability_learner', 'catboost' if 'catboost' in learners else learners[0])
    f = shards / f'av_{av_learner}.parquet'
    charging_end_share = float(df['charging_duration_hours'].notna().mean()) if 'charging_duration_hours' in df else 0.0
    av = None
    if charging_end_share < 0.5:
        # the information-availability analysis needs the time charging ended; a network that does not
        # record it is reported as not applicable rather than analysed on imputed values
        (out / 'feature_availability_NOT_APPLICABLE.txt').write_text(
            f'charging end recorded for {charging_end_share:.1%} of sessions; analysis requires at least 50%\n')
        print(f'[availability] skipped: charging end recorded for {charging_end_share:.1%} of sessions', flush=True)
    else:
        if not f.exists():
            availability.run(df, numeric, av_learner, seeds, learner_cfg).to_parquet(f, index=False)
            print(f'[availability] {av_learner} done', flush=True)
        av = availability.summarise(pd.read_parquet(f), n_boot); av.to_csv(out / 'feature_availability.csv', index=False)

    # 4. figures
    paper_figures.style()
    paper_figures.decomposition(dec, figs)
    for learner in sorted(rows['learner'].unique()):
        paper_figures.warmup(warm, figs, learner)
        paper_figures.tradeoff(trade[trade['learner'] == learner], figs, learner)
        paper_figures.unmet_by_history(byk[byk['learner'] == learner], figs, learner)
    if av is not None:
        paper_figures.availability(av, figs, av_learner)

    # 5. LaTeX tables
    d = dec[dec['component'] != 'published-style features'].copy()
    d['Effect (95\\% CI)'] = d.apply(lambda r: f"{r.effect_pct:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}]", axis=1)
    d['Holm p'] = d['p_holm'].map(lambda p: '<0.001' if p < 0.001 else f'{p:.3f}')
    _latex(d[['target', 'learner', 'component', 'Effect (95\\% CI)', 'Holm p']],
           'Decomposition of the deployment gap for unseen drivers (change in MAE, driver-cluster bootstrap).',
           'tab:coldstart', tex / 'paper_coldstart.tex')
    if av is not None:
        a = av.copy()
        a['MAE'] = a.apply(lambda r: f'{r.mae:.2f} $\\pm$ {r.mae_sd:.2f}', axis=1)
        a['Change'] = a.apply(lambda r: '' if pd.isna(r.get('change_vs_plug_in_pct')) else
                              f"{r.change_vs_plug_in_pct:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}]", axis=1)
        _latex(a[['scope', 'feature_set', 'MAE', 'Change']], 'Energy MAE (kWh) by information available at prediction time.',
               'tab:availability', tex / 'paper_availability.tex')
    m = matched.copy()
    m['Flexibility retained'] = m.apply(lambda r: f'{r.flex_vs_oracle_pct:.1f}' + ('' if pd.isna(r.flex_sd) else f' $\\pm$ {r.flex_sd:.1f}'), axis=1)
    _latex(m[['learner', 'setting', 'unmet_target_pct', 'Flexibility retained']],
           'Smart-charging flexibility retained (percent of oracle) at matched undelivered-energy levels.',
           'tab:impact', tex / 'paper_impact.tex')

    # 6. headline summary
    lines = ['# Paper results', '', '## Cold-start decomposition', '', dec.round(2).to_markdown(index=False), '',
             '## Warm-up curve', '', warm.round(3).to_markdown(index=False), '',
             '## Feature availability', '', (av.round(3).to_markdown(index=False) if av is not None else 'Not applicable: charging end not recorded.'), '',
             '## Smart-charging impact at matched reliability', '', matched.round(1).to_markdown(index=False), '',
             '## Smart-charging trade-off', '', trade.round(2).to_markdown(index=False), '']
    (out / 'PAPER_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'paper analyses complete: {out}, {figs}, {tex}')


if __name__ == '__main__':
    main()
