from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Journal-quality defaults. PDF/SVG stay vector; PNG is exported at 600 dpi by default.
_DPI = int(os.environ.get('EVTRUST_FIG_DPI', '600'))


def configure_publication_style(dpi: int | None = None) -> None:
    global _DPI
    if dpi is not None:
        _DPI = int(dpi)
    mpl.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': _DPI,
        'font.family': 'DejaVu Sans',
        'font.size': 9.0,
        'axes.titlesize': 10.5,
        'axes.labelsize': 9.5,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.0,
        'legend.frameon': False,
        'lines.linewidth': 1.6,
        'lines.markersize': 5,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.18,
        'pdf.fonttype': 42,      # searchable/editable TrueType fonts in PDF
        'ps.fonttype': 42,
        'svg.fonttype': 'none',  # preserve text rather than paths
        'figure.constrained_layout.use': False,
    })


def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=.7)


def _save(fig, out: str | Path, dpi: int | None = None):
    """Save every figure as 600-dpi PNG plus vector PDF and SVG.

    `out` may contain an extension; it is ignored and only the stem is used.
    Files are organized under figures/png, figures/pdf, figures/svg.
    """
    dpi = int(dpi or _DPI)
    out = Path(out)
    base = out.stem if out.suffix else out.name
    root = out.parent if str(out.parent) not in ('', '.') else Path('figures')
    # If caller already passed figures/FigX, use figures as common root.
    if root.name in {'png', 'pdf', 'svg'}:
        root = root.parent
    for fmt in ('png', 'pdf', 'svg'):
        d = root / fmt
        d.mkdir(parents=True, exist_ok=True)
        p = d / f'{base}.{fmt}'
        kwargs = dict(bbox_inches='tight', pad_inches=0.03, facecolor='white')
        if fmt == 'png':
            kwargs['dpi'] = dpi
        fig.savefig(p, **kwargs)
    plt.close(fig)


def study_design_schematic(out: str):
    fig, ax = plt.subplots(figsize=(10.0, 3.7))
    ax.axis('off')
    boxes = [
        ('Leakage-audited\nACN sessions', .08),
        ('Random / temporal /\nuser / site splits', .28),
        ('Baselines + CatBoost +\nFT-Transformer', .49),
        ('Conformal uncertainty +\ntarget recalibration', .70),
        ('Risk-controlled\nrecommend / abstain', .90),
    ]
    for text, x in boxes:
        ax.text(x, .56, text, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.55', fc='white', ec='.25', lw=1.0))
    for (_, x1), (_, x2) in zip(boxes[:-1], boxes[1:]):
        ax.annotate('', xy=(x2-.075,.56), xytext=(x1+.075,.56), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='.2'))
    ax.text(.5, .93, 'TrustEV-X evaluation and deployment pipeline', transform=ax.transAxes,
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(.5, .14, 'Primary endpoints: error, calibration, transfer robustness, selective risk and operational utility',
            transform=ax.transAxes, ha='center', va='center', fontsize=9)
    _save(fig, out)


def protocol_forest(metrics: pd.DataFrame, out: str):
    d = metrics.copy().sort_values(['target','protocol','mae'])
    labels = d['target'].astype(str) + ' | ' + d['protocol'].astype(str) + ' | ' + d['model'].astype(str)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, len(d)*.20)))
    if {'mae_ci_low','mae_ci_high'}.issubset(d.columns):
        lo = np.clip(d['mae'] - d['mae_ci_low'], 0, None)
        hi = np.clip(d['mae_ci_high'] - d['mae'], 0, None)
        ax.errorbar(d['mae'], y, xerr=np.vstack([lo,hi]), fmt='o', capsize=2.0, elinewidth=1.0)
    else:
        ax.scatter(d['mae'], y, s=22)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Mean absolute error (lower is better)')
    ax.set_title('Predictive performance under increasingly realistic evaluation protocols', loc='left', fontweight='bold')
    ax.grid(axis='x')
    _clean_axes(ax)
    _save(fig, out)


def rank_heatmap(metrics: pd.DataFrame, target: str, out: str):
    g = metrics[metrics.target == target].copy()
    mat = g.pivot_table(index='model', columns='protocol', values='mae', aggfunc='mean')
    ranks = mat.rank(axis=0, method='average')
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, .48*len(ranks))))
    im = ax.imshow(ranks.to_numpy(), aspect='auto', cmap='viridis_r')
    ax.set_xticks(range(len(ranks.columns)), ranks.columns, rotation=28, ha='right')
    ax.set_yticks(range(len(ranks.index)), ranks.index)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            if np.isfinite(ranks.iloc[i,j]):
                ax.text(j,i,f'{ranks.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8)
    ax.set_title(f'Model rank stability - {target}', loc='left', fontweight='bold')
    cbar=fig.colorbar(im, ax=ax, pad=.02); cbar.set_label('Rank (1 = best)')
    _save(fig, out)


def calibration_curve(interval_df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(6.6,4.8))
    for label, g in interval_df.groupby('protocol'):
        x = 1-g['alpha'].to_numpy(float); y = g['coverage'].to_numpy(float)
        order = np.argsort(x); ax.plot(x[order], y[order], marker='o', label=label)
    ax.plot([.5,1],[.5,1], linestyle='--', color='.25', lw=1.0, label='Ideal')
    ax.set_xlabel('Nominal coverage'); ax.set_ylabel('Empirical coverage')
    ax.set_xlim(.48,1.01); ax.set_ylim(.48,1.01)
    ax.set_title('Conformal reliability under distribution shift', loc='left', fontweight='bold')
    ax.legend(ncol=2); ax.grid(); _clean_axes(ax)
    _save(fig, out)


def calibration_budget(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(6.7,4.8))
    for site, g in df.groupby('target_site'):
        g=g.sort_values('budget'); ax.plot(g['budget'], g['coverage'], marker='o', label=site)
    if 'nominal' in df.columns and len(df):
        ax.axhline(df['nominal'].iloc[0], linestyle='--', color='.25', lw=1.0, label='Nominal')
    ax.set_xscale('symlog', linthresh=25)
    ax.set_xlabel('Target-site calibration sessions'); ax.set_ylabel('Empirical coverage')
    ax.set_title('Local calibration budget needed to recover uncertainty reliability', loc='left', fontweight='bold')
    ax.legend(ncol=2); ax.grid(); _clean_axes(ax)
    _save(fig, out)


def risk_coverage(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(6.6,4.8))
    if {'protocol','coverage','risk'}.issubset(df.columns):
        for label,g in df.groupby('protocol'):
            g=g.sort_values('coverage'); ax.plot(g['coverage'],g['risk'],marker='o',markersize=3,label=label)
        ax.legend()
    else:
        g=df.sort_values('coverage'); ax.plot(g['coverage'], g['risk'], marker='o', markersize=3)
    ax.set_xlabel('Decision coverage (fraction of cases acted on)'); ax.set_ylabel('Selective risk')
    ax.set_title('Risk-coverage frontier for uncertainty-guided abstention', loc='left', fontweight='bold')
    ax.grid(); _clean_axes(ax); _save(fig, out)


def timeline(sessions: pd.DataFrame, pandemic_break: str, out: str):
    s=sessions.copy(); s['connection_time']=pd.to_datetime(s['connection_time'])
    monthly = s.set_index('connection_time').groupby('site').resample('MS').size().rename('sessions').reset_index()
    fig, ax = plt.subplots(figsize=(9.0,4.5))
    for site,g in monthly.groupby('site'):
        ax.plot(g.connection_time, g.sessions, label=site)
    tz = getattr(s.connection_time.dt, 'tz', None)
    br = pd.Timestamp(pandemic_break)
    if tz is not None and br.tzinfo is None: br=br.tz_localize(tz)
    ax.axvline(br, linestyle='--', color='.25', lw=1.0)
    ax.text(br, ax.get_ylim()[1]*.96, ' Pandemic break', va='top', ha='left', fontsize=8)
    ax.set_ylabel('Sessions per month'); ax.set_xlabel('Calendar time')
    ax.set_title('ACN session volume and temporal regime shift', loc='left', fontweight='bold')
    ax.legend(ncol=3); ax.grid(axis='y'); _clean_axes(ax); _save(fig, out)


def transfer_heatmap(metrics: pd.DataFrame, target: str, out: str):
    g=metrics[(metrics['protocol']=='site_disjoint') & (metrics['target']==target)].copy()
    if g.empty: return
    mat=g.groupby(['target_site','model'])['mae'].mean().unstack('model')
    fig,ax=plt.subplots(figsize=(max(6.8,.85*len(mat.columns)),max(3.0,.65*len(mat))))
    im=ax.imshow(mat.to_numpy(),aspect='auto',cmap='magma_r')
    ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=28,ha='right')
    ax.set_yticks(range(len(mat.index)),mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat.iloc[i,j]): ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title(f'Cross-site transfer performance - {target}', loc='left', fontweight='bold')
    cbar=fig.colorbar(im,ax=ax,pad=.02); cbar.set_label('MAE')
    _save(fig,out)


def conditional_coverage_heatmap(report: pd.DataFrame, target: str, model: str, dimension: str, out: str):
    g=report[(report.target==target)&(report.model==model)&(report.dimension==dimension)].copy()
    if g.empty: return
    mat=g.pivot_table(index='group',columns='protocol',values='coverage',aggfunc='mean')
    fig,ax=plt.subplots(figsize=(7.2,max(3,.50*len(mat))))
    im=ax.imshow(mat.to_numpy(),aspect='auto',vmin=0.5,vmax=1.0,cmap='viridis')
    ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=28,ha='right')
    ax.set_yticks(range(len(mat.index)),mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat.iloc[i,j]): ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title(f'Conditional coverage - {target}, {model}, by {dimension}', loc='left', fontweight='bold')
    cbar=fig.colorbar(im,ax=ax,pad=.02); cbar.set_label('Empirical coverage')
    _save(fig,out)


def shift_error_scatter(shift: pd.DataFrame, optimism: pd.DataFrame, out: str):
    s=shift.groupby('protocol',as_index=False)['normalized_shift'].mean()
    e=optimism.groupby('protocol',as_index=False)['relative_optimism_pct'].mean()
    m=s.merge(e,on='protocol')
    fig,ax=plt.subplots(figsize=(6.5,4.8))
    ax.scatter(m.normalized_shift,m.relative_optimism_pct,s=40)
    for _,r in m.iterrows(): ax.annotate(str(r.protocol),(r.normalized_shift,r.relative_optimism_pct),xytext=(4,4),textcoords='offset points')
    ax.set_xlabel('Mean normalized feature shift'); ax.set_ylabel('Mean MAE optimism gap (%)')
    ax.set_title('Distribution shift versus performance degradation', loc='left', fontweight='bold')
    ax.grid(); _clean_axes(ax); _save(fig,out)


def optimism_heatmap(df: pd.DataFrame, out: str):
    if df.empty: return
    g=df.copy()
    mat=g.pivot_table(index='model',columns=['target','protocol'],values='relative_optimism_pct',aggfunc='mean')
    if mat.empty: return
    fig,ax=plt.subplots(figsize=(max(7.5,.72*len(mat.columns)),max(3.2,.45*len(mat))))
    vmax=np.nanpercentile(np.abs(mat.to_numpy()),95) if np.isfinite(mat.to_numpy()).any() else 1
    vmax=max(vmax,1e-6)
    im=ax.imshow(mat.to_numpy(),aspect='auto',cmap='coolwarm',vmin=-vmax,vmax=vmax)
    labs=[f'{a}\n{b}' for a,b in mat.columns]
    ax.set_xticks(range(len(labs)),labs,rotation=32,ha='right')
    ax.set_yticks(range(len(mat.index)),mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v=mat.iloc[i,j]
            if np.isfinite(v): ax.text(j,i,f'{v:.0f}%',ha='center',va='center',fontsize=7)
    ax.set_title('Protocol-specific optimism relative to random splitting', loc='left', fontweight='bold')
    cbar=fig.colorbar(im,ax=ax,pad=.02); cbar.set_label('Relative MAE gap (%)')
    _save(fig,out)


def paired_effect_forest(df: pd.DataFrame, out: str, max_rows: int=40):
    if df.empty: return
    g=df.copy()
    g['abs_effect']=g['difference_a_minus_b'].abs()
    g=g.sort_values('abs_effect',ascending=False).head(max_rows).sort_values('difference_a_minus_b')
    y=np.arange(len(g))
    labels=(g['target'].astype(str)+' | '+g['protocol'].astype(str)+' | '+g['model_a'].astype(str)+' - '+g['model_b'].astype(str))
    fig,ax=plt.subplots(figsize=(9.0,max(5.0,.22*len(g))))
    x=g['difference_a_minus_b'].to_numpy(float)
    lo=x-g['ci_low'].to_numpy(float); hi=g['ci_high'].to_numpy(float)-x
    ax.errorbar(x,y,xerr=np.vstack([lo,hi]),fmt='o',capsize=2,elinewidth=1)
    ax.axvline(0,color='.25',lw=1,ls='--')
    ax.set_yticks(y,labels); ax.set_xlabel('Paired MAE difference (model A - model B)')
    ax.set_title('Largest paired model effects with day-block bootstrap 95% CIs',loc='left',fontweight='bold')
    ax.grid(axis='x'); _clean_axes(ax); _save(fig,out)


def ablation_plot(df: pd.DataFrame, out: str):
    if df.empty: return
    g=(df[df.ablation!='full'].groupby(['target','protocol','ablation'],as_index=False)['delta_mae_vs_full'].mean())
    if g.empty: return
    g['label']=g['target'].astype(str)+' | '+g['protocol'].astype(str)+' | '+g['ablation'].astype(str)
    g=g.sort_values('delta_mae_vs_full')
    fig,ax=plt.subplots(figsize=(8.8,max(4.5,.22*len(g))))
    y=np.arange(len(g)); ax.scatter(g.delta_mae_vs_full,y,s=24); ax.axvline(0,color='.25',ls='--',lw=1)
    ax.set_yticks(y,g.label); ax.set_xlabel('Change in MAE relative to full feature set')
    ax.set_title('Grouped feature ablation: which information drives generalization?',loc='left',fontweight='bold')
    ax.grid(axis='x'); _clean_axes(ax); _save(fig,out)


def occupancy_summary(df: pd.DataFrame, out: str):
    if df.empty: return
    metric='hit_rate' if 'hit_rate' in df.columns else ('recommendation_hit_rate' if 'recommendation_hit_rate' in df.columns else None)
    if metric is None: return
    g=df.groupby(['protocol','horizon'],as_index=False)[metric].agg(['mean','std']).reset_index()
    protocols=list(g.protocol.unique()); horizons=sorted(g.horizon.unique())
    fig,ax=plt.subplots(figsize=(7.0,4.8))
    x=np.arange(len(horizons)); width=.8/max(1,len(protocols))
    for k,p in enumerate(protocols):
        z=g[g.protocol==p].set_index('horizon').reindex(horizons)
        ax.errorbar(x+(k-(len(protocols)-1)/2)*width,z['mean'],yerr=z['std'],fmt='o',capsize=2,label=p)
    ax.set_xticks(x,[f'{h} h' for h in horizons]); ax.set_xlabel('Prediction horizon'); ax.set_ylabel('Recommendation hit rate')
    ax.set_title('Operational station recommendation performance during congestion',loc='left',fontweight='bold')
    ax.legend(); ax.grid(axis='y'); _clean_axes(ax); _save(fig,out)
