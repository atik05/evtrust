from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd
from rich import print

from .config import load_config
from .acn import download_sessions
from .data import load_raw_json, normalize_sessions, clean_sessions
from .features import build_features
from .audit import leakage_audit, write_audit
from .experiments import run_regression, analyze_regression
from .occupancy import build_occupancy_panel
from .occupancy_experiment import run_occupancy
from .plots import (configure_publication_style, study_design_schematic, protocol_forest, rank_heatmap, calibration_curve, calibration_budget, risk_coverage, timeline, transfer_heatmap, conditional_coverage_heatmap, shift_error_scatter, optimism_heatmap, paired_effect_forest, ablation_plot, occupancy_summary)
from .advanced_analysis import distribution_shift_report, conditional_coverage_report, paired_model_comparisons, site_transfer_summary
from .utils import ensure_dirs
from .freeze import freeze_session_splits
from .ablation import run_grouped_ablation

app = typer.Typer(add_completion=False, help='Trustworthy EV charging prediction benchmark')

@app.command()
def download(config: str='configs/main.yaml'):
    cfg=load_config(config)
    paths=download_sessions(cfg['project']['sites'],cfg['data']['raw_dir'],start=cfg['data']['start'],end=cfg['data']['end'])
    print({'downloaded':[str(p) for p in paths]})

@app.command()
def prepare(config: str='configs/main.yaml'):
    cfg=load_config(config)
    raw=load_raw_json(cfg['data']['raw_dir'],cfg['project']['sites'])
    df=normalize_sessions(raw,cfg['data']['timezone'])
    df=clean_sessions(df,cfg['data']['max_duration_hours'],cfg['data']['max_energy_kwh'],
                      float(cfg['data'].get('min_duration_minutes',5)),'results/metrics/data_flow.csv')
    start=pd.Timestamp(cfg['data']['start'],tz=cfg['data']['timezone']); end=pd.Timestamp(cfg['data']['end'],tz=cfg['data']['timezone'])+pd.Timedelta(days=1)
    df=df[(df.connection_time>=start)&(df.connection_time<end)].copy()
    df=build_features(df)
    path=Path(cfg['data']['processed_dir'])/'sessions.parquet'; ensure_dirs(path.parent); df.to_parquet(path,index=False)
    print(f'[green]Prepared {len(df):,} sessions -> {path}[/green]')

@app.command()
def audit(config: str='configs/main.yaml'):
    cfg=load_config(config); path=Path(cfg['data']['processed_dir'])/'sessions.parquet'
    df=pd.read_parquet(path); report=leakage_audit(df,cfg); write_audit(report,'results/metrics/leakage_audit.json')
    print(report)
    if not report['ok']:
        raise typer.Exit(code=2)

@app.command('freeze-splits')
def freeze_splits_cmd(config: str='configs/main.yaml'):
    cfg=load_config(config); df=pd.read_parquet(Path(cfg['data']['processed_dir'])/'sessions.parquet')
    paths=freeze_session_splits(df,cfg)
    print(f'[green]Frozen {len(paths)} split-definition files in results/splits/[/green]')

@app.command('run-regression')
def run_regression_cmd(config: str='configs/main.yaml'):
    cfg=load_config(config); df=pd.read_parquet(Path(cfg['data']['processed_dir'])/'sessions.parquet')
    run_regression(df,cfg); print('[green]Regression experiments complete.[/green]')

@app.command('run-ablation')
def run_ablation_cmd(config: str='configs/main.yaml'):
    cfg=load_config(config); df=pd.read_parquet(Path(cfg['data']['processed_dir'])/'sessions.parquet')
    out=run_grouped_ablation(df,cfg)
    print(f'[green]Grouped ablation complete: {len(out)} rows.[/green]')

@app.command('build-occupancy')
def build_occupancy_cmd(config: str='configs/main.yaml'):
    cfg=load_config(config); df=pd.read_parquet(Path(cfg['data']['processed_dir'])/'sessions.parquet')
    panel=build_occupancy_panel(df,cfg['project']['horizons_hours'],cfg['project']['prediction_frequency_minutes'])
    path=Path(cfg['data']['processed_dir'])/'occupancy_panel.parquet'; panel.to_parquet(path,index=False)
    print(f'[green]Built {len(panel):,} station-query rows -> {path}[/green]')

@app.command('run-occupancy')
def run_occupancy_cmd(config: str='configs/main.yaml'):
    cfg=load_config(config); panel=pd.read_parquet(Path(cfg['data']['processed_dir'])/'occupancy_panel.parquet')
    run_occupancy(panel,cfg); print('[green]Occupancy experiments complete.[/green]')

@app.command()
def analyze(config: str='configs/main.yaml'):
    cfg=load_config(config); m=analyze_regression(cfg)
    sessions=pd.read_parquet(Path(cfg['data']['processed_dir'])/'sessions.parquet')
    pred=pd.read_parquet('results/predictions/regression_predictions.parquet')
    shift=distribution_shift_report(sessions,cfg); shift.to_csv('results/metrics/distribution_shift.csv',index=False)
    cond=conditional_coverage_report(pred,sessions); cond.to_csv('results/metrics/conditional_coverage.csv',index=False)
    pairs=paired_model_comparisons(pred,cfg); pairs.to_csv('results/metrics/paired_model_comparisons.csv',index=False)
    site=site_transfer_summary(m); site.to_csv('results/metrics/site_transfer_summary.csv',index=False)
    from .experiments import _write_cross_site_user_overlap
    _write_cross_site_user_overlap(sessions,cfg)
    print(m.head(20).to_string(index=False))
    print('[green]Advanced statistical reports written to results/metrics/[/green]')

@app.command()
def plot(config: str='configs/main.yaml'):
    cfg=load_config(config); ensure_dirs('figures')
    dpi=int(cfg.get('visualization',{}).get('dpi',600))
    configure_publication_style(dpi)
    study_design_schematic('figures/Fig1_study_design')

    metrics=pd.read_csv('results/metrics/regression_metrics_by_seed.csv')
    summary=metrics.groupby(['target','protocol','model'],as_index=False).agg(
        mae=('mae','mean'),mae_ci_low=('mae_ci_low','mean'),mae_ci_high=('mae_ci_high','mean'))
    protocol_forest(summary,'figures/Fig3_protocol_performance')
    for target in summary.target.unique():
        rank_heatmap(summary,target,f'figures/Fig4_rank_{target}')

    rel=Path('results/metrics/conformal_reliability.csv')
    if rel.exists():
        r=pd.read_csv(rel).groupby(['protocol','alpha'],as_index=False)['coverage'].mean()
        calibration_curve(r,'figures/Fig6_conformal_reliability')

    budget=Path('results/metrics/calibration_budget.csv')
    if budget.exists():
        b=pd.read_csv(budget).groupby(['target_site','budget'],as_index=False).agg(coverage=('coverage','mean'),nominal=('nominal','mean'))
        calibration_budget(b,'figures/Fig7_calibration_budget')

    risk=Path('results/metrics/risk_coverage.csv')
    if risk.exists():
        rr=pd.read_csv(risk)
        group_cols=[c for c in ['protocol','threshold'] if c in rr.columns]
        r=rr.groupby(group_cols,as_index=False).agg(coverage=('coverage','mean'),risk=('risk','mean')) if group_cols else rr
        risk_coverage(r,'figures/Fig8_risk_coverage')

    sessions=pd.read_parquet(Path(cfg['data']['processed_dir'])/'sessions.parquet')
    timeline(sessions,cfg['splits']['pandemic_break'],'figures/Fig2_timeline')
    for target in metrics.target.unique():
        transfer_heatmap(metrics,target,f'figures/Fig5_transfer_{target}')

    cond_path=Path('results/metrics/conditional_coverage.csv')
    if cond_path.exists():
        cc=pd.read_csv(cond_path)
        for target in cc.target.unique():
            for model in sorted(set(cc.model.unique()) & {'catboost','torch_ft','catboost_cqr'}):
                for dimension in sorted(cc[cc.model==model].dimension.unique()):
                    conditional_coverage_heatmap(cc,target,model,dimension,f'figures/FigS_conditional_coverage_{target}_{model}_{dimension}')

    shift_path=Path('results/metrics/distribution_shift.csv'); opt_path=Path('results/metrics/protocol_optimism.csv')
    if shift_path.exists() and opt_path.exists():
        shift_error_scatter(pd.read_csv(shift_path),pd.read_csv(opt_path),'figures/FigS_shift_vs_error')
        optimism_heatmap(pd.read_csv(opt_path),'figures/Fig9_protocol_optimism')

    pairs=Path('results/metrics/paired_model_comparisons.csv')
    if pairs.exists(): paired_effect_forest(pd.read_csv(pairs),'figures/Fig10_paired_effects')

    abl=Path('results/metrics/grouped_ablation.csv')
    if abl.exists(): ablation_plot(pd.read_csv(abl),'figures/FigS_grouped_ablation')

    occ=Path('results/metrics/occupancy_metrics.csv')
    if occ.exists(): occupancy_summary(pd.read_csv(occ),'figures/Fig11_occupancy_recommendation')

    print(f'[green]Publication figures written in PNG/PDF/SVG at {dpi} dpi (PNG) under figures/.[/green]')

if __name__ == '__main__':
    app()
