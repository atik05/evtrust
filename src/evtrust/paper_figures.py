"""Publication figures for the cold-start, feature-availability and impact analyses.

Okabe-Ito colour-blind safe palette, vector PDF and 600 dpi PNG, sized for a
single (3.5 in) or double (7.2 in) journal column.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OI = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00',
      'purple': '#CC79A7', 'sky': '#56B4E9', 'yellow': '#F0E442', 'black': '#000000', 'grey': '#999999'}
K_ORDER = ['0', '1', '2', '3-4', '5-9', '10-19', '20+']
TARGET = {'kwh_delivered': 'Energy (kWh)', 'duration_hours': 'Duration (h)'}
LEARNER = {'catboost': 'CatBoost', 'ft': 'FT-Transformer', 'hgb': 'Gradient boosting'}
SETTING = {'A': 'Random split', 'C': 'Unseen driver, own history', 'D': 'Unseen driver, cold start',
           'driver_request': 'Driver request'}
COL = {'A': OI['grey'], 'C': OI['blue'], 'D': OI['red'], 'driver_request': OI['orange']}


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'lines.linewidth': 1.3, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf')
    fig.savefig(out / f'{name}.png', dpi=600)
    plt.close(fig)


def decomposition(dec: pd.DataFrame, out: Path):
    comps = ['future-session leakage', 'unseen-driver cost', 'value of own history', 'total gap']
    learners = [l for l in ['catboost', 'ft', 'hgb'] if l in set(dec['learner'])]
    targets = [t for t in ['kwh_delivered', 'duration_hours'] if t in set(dec['target'])]
    fig, axes = plt.subplots(1, len(targets), figsize=(7.2, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    colors = [OI['blue'], OI['orange'], OI['green']]
    for ax, t in zip(axes, targets):
        y = np.arange(len(comps))
        h = 0.8 / max(len(learners), 1)
        for i, l in enumerate(learners):
            z = dec[(dec['target'] == t) & (dec['learner'] == l)].set_index('component').reindex(comps)
            pos = y + (i - (len(learners) - 1) / 2) * h
            ax.barh(pos, z['effect_pct'], height=h * 0.9, color=colors[i], label=LEARNER.get(l, l))
            ax.errorbar(z['effect_pct'], pos, xerr=[z['effect_pct'] - z['ci_low'], z['ci_high'] - z['effect_pct']],
                        fmt='none', ecolor='black', elinewidth=0.6, capsize=1.5)
        ax.axvline(0, color='black', lw=0.6)
        ax.set_yticks(y, ['Random-split\nleakage', 'Unseen-driver\ncost', 'Value of own\nhistory', 'Total gap'])
        ax.set_xlabel('Change in MAE (%), 95% CI')
        ax.set_title(TARGET[t])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='lower center', ncol=len(learners), bbox_to_anchor=(0.5, -0.16))
    _save(fig, out, 'fig_coldstart_decomposition')


def warmup(warm: pd.DataFrame, out: Path, learner: str):
    targets = [t for t in ['kwh_delivered', 'duration_hours'] if t in set(warm['target'])]
    fig, axes = plt.subplots(1, len(targets), figsize=(7.2, 2.5))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        z = warm[(warm['target'] == t) & (warm['learner'] == learner)].set_index('k_bucket').reindex(K_ORDER).dropna(subset=['mae_random'])
        x = np.arange(len(z))
        ax.plot(x, z['mae_cold'], 'o-', color=COL['D'], ms=3, label=SETTING['D'])
        ax.plot(x, z['mae_warm'], 'o-', color=COL['C'], ms=3, label=SETTING['C'])
        ax.plot(x, z['mae_random'], 's--', color=COL['A'], ms=3, label=SETTING['A'])
        lo = z['mae_random'] * (1 + z['ci_low'] / 100); hi = z['mae_random'] * (1 + z['ci_high'] / 100)
        ax.fill_between(x, lo, hi, color=COL['C'], alpha=0.18, lw=0, label='95% CI, unseen with history')
        ax.set_xticks(x, z.index)
        ax.set_xlabel("Driver's completed sessions before this one")
        ax.set_ylabel(f'MAE, {TARGET[t]}')
        ax.set_title(f'{TARGET[t].split(" (")[0]}: {LEARNER.get(learner, learner)}')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.16))
    _save(fig, out, f'fig_warmup_{learner}')


def availability(av: pd.DataFrame, out: Path, learner: str):
    z = av[av['learner'] == learner]
    scopes = [s for s in ['all', 'caltech', 'jpl', 'office001'] if s in set(z['scope'])]
    sets = ['plug_in', 'plus_charging', 'plus_departure']
    lab = {'plug_in': 'At plug-in', 'plus_charging': '+ charging duration', 'plus_departure': '+ session duration'}
    col = {'plug_in': OI['blue'], 'plus_charging': OI['orange'], 'plus_departure': OI['red']}
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(scopes)); w = 0.26
    for i, fs in enumerate(sets):
        v = z[z['feature_set'] == fs].set_index('scope').reindex(scopes)
        ax.bar(x + (i - 1) * w, v['mae'], w, yerr=v['mae_sd'], color=col[fs], label=lab[fs],
               error_kw={'elinewidth': 0.6, 'capsize': 1.5})
    names = {'all': 'All sites', 'caltech': 'Caltech', 'jpl': 'JPL', 'office001': 'Office001'}
    ax.set_xticks(x, [names.get(s, s) for s in scopes])
    ax.set_ylabel('Energy MAE (kWh)')
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    _save(fig, out, f'fig_feature_availability_{learner}')


def tradeoff(summ: pd.DataFrame, out: Path, learner: str):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for s in ['A', 'C', 'D']:
        z = summ[summ['setting'] == s].sort_values('unmet_share_pct')
        if z.empty:
            continue
        ax.plot(z['unmet_share_pct'], z['flex_vs_oracle_pct'], 'o-', ms=2.5, color=COL[s], label=SETTING[s])
    r = summ[summ['setting'] == 'driver_request']
    if not r.empty:
        ax.plot(r['unmet_share_pct'], r['flex_vs_oracle_pct'], 'D', ms=5, color=COL['driver_request'],
                label=f"Driver request (subset, n = {int(r['n'].iloc[0]):,})")
    ax.plot([0], [100], '*', ms=9, color=OI['black'], label='Oracle')
    ax.axvline(2, color=OI['grey'], lw=0.6, ls=':')
    ax.set_xlabel('Undelivered energy (% of achievable)')
    ax.set_ylabel('Flexibility retained (% of oracle)')
    ax.set_title(f'Smart-charging trade-off: {LEARNER.get(learner, learner)}')
    ax.legend(frameon=False, fontsize=6.5, loc='lower right')
    ax.set_xlim(left=-0.5)
    _save(fig, out, f'fig_impact_tradeoff_{learner}')


def unmet_by_history(bk: pd.DataFrame, out: Path, learner: str):
    z = bk[bk['tau'] == 0.5]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for s in ['A', 'C', 'D']:
        v = z[z['setting'] == s].set_index('k_bucket').reindex(K_ORDER).dropna()
        ax.plot(np.arange(len(v)), v['unmet_share_pct'], 'o-', ms=3, color=COL[s], label=SETTING[s])
    ax.set_xticks(np.arange(len(K_ORDER)), K_ORDER)
    ax.set_xlabel("Driver's completed sessions before this one")
    ax.set_ylabel('Undelivered energy (%)')
    ax.set_title('Point forecasts, as-late-as-possible charging')
    ax.legend(frameon=False, fontsize=6.5)
    _save(fig, out, f'fig_impact_by_history_{learner}')


def cd_diagram(fr: pd.DataFrame, out: Path, target: str, protocol: str):
    """Critical-difference diagram (Demsar 2006): average ranks and Nemenyi cliques."""
    from .benchmark import DISPLAY
    z = fr[(fr['target'] == target) & (fr['protocol'] == protocol)].sort_values('average_rank')
    if z.empty:
        return
    names = [DISPLAY.get(m, m) for m in z['model']]; ranks = z['average_rank'].to_numpy()
    k = len(ranks); cd = float(z['nemenyi_cd'].iloc[0])
    cliques = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if ranks[jj] - ranks[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    half = (k + 1) // 2
    rows = max(half, k - half)
    step = 0.32                                   # vertical units per label row
    top = 0.0
    clique_depth = 0.25 * (len(cliques) + 1)
    bottom = top - clique_depth - step * rows - 0.2
    fig, ax = plt.subplots(figsize=(3.6, 0.28 * (rows + len(cliques)) + 0.9))
    ax.set_xlim(0.2, k + 0.8); ax.set_ylim(bottom, top + 0.9); ax.axis('off')
    ax.plot([1, k], [top, top], color='black', lw=0.8)
    for i in range(1, k + 1):
        ax.plot([i, i], [top, top + 0.1], color='black', lw=0.6)
        ax.text(i, top + 0.18, str(i), ha='center', va='bottom', fontsize=6.5)
    ax.plot([1, 1 + cd], [top + 0.62, top + 0.62], color='black', lw=1.2)
    ax.plot([1, 1], [top + 0.55, top + 0.69], color='black', lw=0.8)
    ax.plot([1 + cd, 1 + cd], [top + 0.55, top + 0.69], color='black', lw=0.8)
    ax.text(1 + cd / 2, top + 0.72, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=6.5)
    for n, (i, j) in enumerate(cliques):
        yy = top - 0.22 - 0.25 * n
        ax.plot([ranks[i] - 0.04, ranks[j] + 0.04], [yy, yy], color=OI['red'], lw=2.2, solid_capstyle='round')
    label_top = top - clique_depth - 0.15
    for i, (nm, rk) in enumerate(zip(names, ranks)):
        left = i < half
        yy = label_top - step * (i if left else k - 1 - i)
        xx = 0.9 if left else k + 0.1
        ax.plot([rk, rk, xx], [top, yy, yy], color='black', lw=0.6)
        ax.text(xx - 0.05 if left else xx + 0.05, yy, f'{nm} ({rk:.2f})', ha='right' if left else 'left',
                va='center', fontsize=6.5)
    _save(fig, out, f'fig_cd_{target}_{protocol}')


def benchmark_protocols(summ: pd.DataFrame, out: Path, target: str):
    """MAE of every learner under every protocol, mean and fold SD."""
    from .benchmark import DISPLAY
    z = summ[summ['target'] == target]
    protos = [p for p in ['random', 'driver_grouped', 'chronological', 'chronological_prepandemic', 'site_transfer',
                          'site_transfer_strict'] if p in set(z['protocol'])]
    order = z[z['protocol'] == protos[0]].sort_values('mae')['model'].tolist()
    lab = {'random': 'Random', 'driver_grouped': 'Unseen drivers', 'chronological': 'Chronological',
           'chronological_prepandemic': 'Chronological, pre-2020', 'site_transfer': 'Leave site out',
           'site_transfer_strict': 'Leave site out, strict'}
    col = [OI['grey'], OI['blue'], OI['orange'], OI['yellow'], OI['red'], OI['purple']]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(order)); w = 0.8 / len(protos)
    for i, p in enumerate(protos):
        v = z[z['protocol'] == p].set_index('model').reindex(order)
        ax.bar(x + (i - (len(protos) - 1) / 2) * w, v['mae'], w, yerr=v['mae_sd'], color=col[i], label=lab[p],
               error_kw={'elinewidth': 0.5, 'capsize': 1})
    ax.set_xticks(x, [DISPLAY.get(m, m) for m in order], rotation=30, ha='right')
    ax.set_ylabel(f'MAE, {TARGET[target]}')
    ax.legend(frameon=False, ncol=len(protos), loc='lower center', bbox_to_anchor=(0.5, 1.0))
    _save(fig, out, f'fig_benchmark_{target}')
