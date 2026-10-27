"""Supplementary Figures S1 to S4 from the pipeline outputs of both networks.

    python make_si_figures.py ACN_RESULTS_DIR PALOALTO_FULL_REPORT.xlsx OUT_DIR

S1 and S2: critical-difference diagrams of the learner benchmark (energy, duration).
S3: cold-start dynamics and smart-charging trade-off for the FT-Transformer on both networks.
S4: site-level undelivered energy for the six scheduling policies at Caltech and JPL.
Each figure is written as vector PDF.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, PAX, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)

MN = {'catboost': 'CatBoost', 'torch_ft': 'FT-Transformer', 'ft': 'FT-Transformer', 'ridge': 'Ridge', 'catboost_cqr': 'CatBoost CQR',
      'historical_mean': 'Historical mean', 'user_history': 'Driver history', 'driver_estimate': 'Driver request',
      'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'hgb': 'HistGB', 'random_forest': 'Random forest', 'extra_trees': 'Extra trees',
      'mlp': 'MLP', 'tabpfn': 'TabPFN v2', 'mean': 'Historical mean'}
PN = {'random': 'Random', 'chronological': 'Chronological', 'site_disjoint': 'Unseen site', 'user_disjoint': 'Unseen driver',
      'driver_grouped': 'Driver-grouped', 'chronological_prepandemic': 'Chronological, pre-2020', 'site_transfer': 'Unseen site',
      'site_transfer_strict': 'Unseen site, strict'}
SN = {'caltech': 'Caltech', 'jpl': 'JPL', 'office001': 'Office001'}
TN = {'duration': 'Duration', 'energy': 'Energy', 'duration_hours': 'Duration', 'kwh_delivered': 'Energy'}


def cd_figure(target, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt_
    plt_.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'], 'font.size': 7,
                          'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fz_ = pd.read_csv(R / 'benchmark/friedman.csv'); fz_ = fz_[fz_.target == target]
    prots = ['random', 'driver_grouped', 'chronological', 'chronological_prepandemic', 'site_transfer', 'site_transfer_strict']
    fig_, axs_ = plt_.subplots(3, 2, figsize=(190 / 25.4, 168 / 25.4), gridspec_kw={'hspace': 0.24, 'wspace': 0.2})
    for i, (ax_, pr) in enumerate(zip(axs_.ravel(), prots)):
        z = fz_[fz_.protocol == pr].sort_values('average_rank')
        ranks = z.average_rank.to_numpy(); names = [MN.get(m_, m_) for m_ in z.model]; k = int(z.n_models.iloc[0]); cd = float(z.nemenyi_cd.iloc[0])
        mx = 0.66 * k
        ax_.set_xlim(1 - mx, k + mx); ax_.set_ylim(0.02, 1.5); ax_.axis('off')
        ax_.plot([1, k], [1, 1], color='#1F2933', lw=0.8)
        for r_ in range(1, k + 1):
            ax_.plot([r_, r_], [1, 1.035], color='#1F2933', lw=0.6); ax_.text(r_, 1.05, str(r_), ha='center', va='bottom', fontsize=5.8)
        ax_.plot([1, 1 + cd], [1.22, 1.22], color='#D55E00', lw=1.2)
        for xx in (1, 1 + cd):
            ax_.plot([xx, xx], [1.195, 1.245], color='#D55E00', lw=1.0)
        ax_.text(1 + cd / 2, 1.255, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=5.8, color='#D55E00')
        half = (len(ranks) + 1) // 2
        for j, (rk, nm) in enumerate(zip(ranks, names)):
            left = j < half; level = j if left else len(ranks) - 1 - j
            y_ = 0.78 - 0.118 * level
            xe = 1 - 0.08 * k if left else k + 0.08 * k
            ax_.plot([rk, rk, xe], [1, y_, y_], color='#5B6670', lw=0.55)
            ax_.text(xe - 0.03 * k if left else xe + 0.03 * k, y_, f'{nm} ({rk:.1f})' if left else f'({rk:.1f}) {nm}',
                     ha='right' if left else 'left', va='center', fontsize=5.9)
        cliques = []
        for a_ in range(len(ranks)):
            b_ = a_
            while b_ + 1 < len(ranks) and ranks[b_ + 1] - ranks[a_] < cd:
                b_ += 1
            if b_ > a_ and not any(c0 <= a_ and b_ <= c1 for c0, c1 in cliques):
                cliques.append((a_, b_))
        for q, (a_, b_) in enumerate(cliques):
            yq = 0.95 - 0.034 * (q % 5)
            ax_.plot([ranks[a_] - 0.04, ranks[b_] + 0.04], [yq, yq], color='#1F2933', lw=2.0, solid_capstyle='round')
        ax_.text(1 - mx, 1.49, f'({chr(97 + i)}) {PN[pr]}', ha='left', va='top', fontsize=7.5)
    fig_.savefig(path, bbox_inches='tight'); plt_.close(fig_)

for t, tl in [('kwh_delivered', 'energy'), ('duration_hours', 'duration')]:
    cd_figure(t, OUT / f'si_cd_{tl}.pdf')

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'], 'font.size': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
C = {'blue': '#0072B2', 'sky': '#56B4E9', 'verm': '#D55E00', 'orange': '#E69F00', 'grey': '#8C959F', 'black': '#1F2933'}
MM = 1 / 25.4
tag = lambda ax, l: ax.text(-0.16, 1.07, l, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
w = pd.read_csv(R / 'paper/coldstart_warmup.csv'); tr = pd.read_csv(R / 'paper/impact_tradeoff.csv')
_PS = pd.ExcelFile(PAX).sheet_names
def PAS(name):
    hits = [x for x in _PS if x == name or name.startswith(x) or x.startswith(name)]
    assert len(hits) == 1, (name, hits)
    return pd.read_excel(PAX, sheet_name=hits[0])
pa_w = PAS('Cold-start warm-up curve'); pa_tr = PAS('Smart charging trade-off'); pa_dc = PAS('Cold-start decomposition')
fig, axs = plt.subplots(2, 3, figsize=(190 * MM, 118 * MM), gridspec_kw={'wspace': 0.45, 'hspace': 0.75})
order = ['0', '1', '2', '3-4', '5-9', '10-19', '20+']
for row, (ww, tt, net) in enumerate([(w, tr, 'ACN workplace sites'), (pa_w, pa_tr, 'Palo Alto public network')]):
    for ax, (t, unit, title) in zip(axs[row, :2], [('kwh_delivered', 'kWh', 'Energy: warm-up'), ('duration_hours', 'h', 'Duration: warm-up')]):
        z = ww[(ww.target == t) & (ww.learner == 'ft')].copy(); z['k'] = z.k_bucket.astype(str); z = z.set_index('k').loc[order]; x = np.arange(7)
        ax.plot(x, z.mae_random, '--o', color=C['grey'], ms=3, label='Random split'); ax.plot(x, z.mae_warm, '-o', color=C['blue'], ms=3, label='Unseen, history')
        ax.plot(x, z.mae_cold, '-s', color=C['verm'], ms=3, label='Cold start'); ax.set_xticks(x); ax.set_xticklabels(order, fontsize=5.8)
        ax.set_xlabel("Driver's completed sessions before this one"); ax.set_ylabel(f'MAE ({unit})'); ax.set_title(title, loc='left', fontsize=8)
    z = tt[tt.learner == 'ft']
    for st, lab, col in [('A', 'Driver seen', C['blue']), ('C', 'Unseen, history', C['sky']), ('D', 'Cold start', C['verm'])]:
        q = z[z.setting == st].sort_values('unmet_share_pct'); axs[row, 2].plot(q.unmet_share_pct, q.flex_vs_oracle_pct, '-o', ms=2.6, lw=1.0, color=col, label=lab, markevery=max(1, len(q) // 10))
    axs[row, 2].set_xlim(0, 30); axs[row, 2].set_ylim(0, 102); axs[row, 2].axvline(2, color=C['grey'], lw=0.6, ls=':')
    axs[row, 2].set_xlabel('Undelivered energy (% of achievable)'); axs[row, 2].set_ylabel('Flexibility retained (% of oracle)')
    axs[row, 2].set_title('Smart charging', loc='left', fontsize=8)
    for ax_ in axs[row]:
        ax_.annotate(net + ', FT-Transformer', xy=(0, 1), xycoords='axes fraction', xytext=(0, 17), textcoords='offset points', fontsize=6.8, color=C['grey'])
for ax_, l_ in zip(axs.ravel(), 'abcdef'):
    ax_.annotate(l_, xy=(-0.16, 1), xycoords='axes fraction', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight='bold', va='bottom')
h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.03), fontsize=6.8)
axs[0, 2].legend(frameon=False, fontsize=6.3, loc='lower right')
fig.savefig(OUT / 'si_ft.pdf', bbox_inches='tight'); plt.close(fig)
sl = pd.read_csv(R / 'operational/site_level.csv')
fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM), gridspec_kw={'wspace': 0.25})
pols = [('oracle', 'Perfect information', C['black'], '-'), ('A', 'Driver seen', C['blue'], '-'), ('C', 'Unseen, history', C['sky'], '-'),
        ('D', 'Cold start', C['verm'], '-'), ('request (D otherwise)', 'Request, else cold start', C['orange'], '--'), ('first come first served', 'Arrival order', C['grey'], '--')]
for ax, site in zip(axs, ['caltech', 'jpl']):
    for p_, lab, col, ls in pols:
        q = sl[(sl.site == site) & (sl.policy == p_)].sort_values('cap_fraction')
        ax.plot(100 * q.cap_fraction, q.undelivered_share_pct.clip(lower=1e-3), ls, color=col, label=lab, lw=1.1)
    ax.set_yscale('log'); ax.set_ylim(0.01, 60); ax.axhline(1, color=C['grey'], lw=0.6, ls=':')
    ax.set_xlabel('Site capacity limit (% of uncontrolled peak)'); ax.set_ylabel('Undelivered energy (%)'); ax.set_title(SN[site] + ': shared capacity limit', loc='left', fontsize=8)
axs[0].legend(frameon=False, fontsize=6.5)
for ax_, l_ in zip(axs, 'ab'): tag(ax_, l_)
fig.savefig(OUT / 'si_site.pdf', bbox_inches='tight'); plt.close(fig)
print(f'Supplementary Figures S1 to S4 written to {OUT}')
