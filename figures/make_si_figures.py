"""Supplementary Figures S1, S2 and S4 from the pipeline outputs of both networks.

    python make_si_figures.py ACN_RESULTS_DIR PALOALTO_FULL_REPORT.xlsx OUT_DIR

S1 and S2: critical-difference diagrams of the learner benchmark (energy, duration).
S4: cold-start dynamics and smart-charging trade-off for the FT-Transformer on both networks
(S3, information availability, is drawn by make_figures.py).
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


def cd_cliques(ranks, cd):
    """Maximal groups of consecutive learners whose mean ranks differ by less than the critical difference."""
    out = []
    for a_ in range(len(ranks)):
        b_ = a_
        while b_ + 1 < len(ranks) and ranks[b_ + 1] - ranks[a_] < cd:
            b_ += 1
        if b_ > a_ and not any(c0 <= a_ and b_ <= c1 for c0, c1 in out):
            out.append((a_, b_))
    return out


def cd_rows(cliques, ranks, gap=0.5):
    """Rows for the clique bars such that bars sharing a row never overlap, so no two groups read as one."""
    ends, rows = [], []
    for a_, b_ in cliques:
        for r_, e_ in enumerate(ends):
            if ranks[a_] > e_ + gap:
                ends[r_] = ranks[b_]; rows.append(r_); break
        else:
            ends.append(ranks[b_]); rows.append(len(ends) - 1)
    return rows


def cd_figure(target, path):
    """Critical-difference diagrams (Demsar 2006) of the six benchmark protocols. Lower mean rank is better; learners
    joined by a bar do not differ by the Nemenyi test. CatBoost and the FT-Transformer, the learners of the main
    analyses, are drawn in colour so that their position can be followed across protocols."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt_
    plt_.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'], 'font.size': 7,
                          'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    INK = '#000000'
    HL = {'CatBoost': '#0072B2', 'FT-Transformer': '#D55E00'}
    fz_ = pd.read_csv(R / 'benchmark/friedman.csv'); fz_ = fz_[fz_.target == target]
    prots = ['random', 'driver_grouped', 'chronological', 'chronological_prepandemic', 'site_transfer', 'site_transfer_strict']
    fig_, axs_ = plt_.subplots(3, 2, figsize=(190 / 25.4, 176 / 25.4), gridspec_kw={'hspace': 0.12, 'wspace': 0.1})
    for i, (ax_, pr) in enumerate(zip(axs_.ravel(), prots)):
        z = fz_[fz_.protocol == pr].sort_values(['average_rank', 'model'])
        ranks = z.average_rank.to_numpy(float); names = [MN.get(m_, m_) for m_ in z.model]
        k = int(z.n_models.iloc[0]); cd = float(z.nemenyi_cd.iloc[0])
        nb = int(z.n_blocks.iloc[0]) if 'n_blocks' in z else None
        pv = float(z.friedman_p.iloc[0]) if 'friedman_p' in z else None
        # the rank axis fills the middle of the panel and the learner names the outer thirds
        W = (k - 1) / 0.40; x_l = 1 - 0.30 * W
        ax_.set_xlim(x_l, x_l + W); ax_.set_ylim(-0.05, 1.62); ax_.axis('off')
        y0 = 1.0
        ax_.plot([1, k], [y0, y0], color=INK, lw=0.8, solid_capstyle='butt')
        for r_ in range(1, k + 1):
            ax_.plot([r_, r_], [y0, y0 + 0.035], color=INK, lw=0.6)
            ax_.text(r_, y0 + 0.05, str(r_), ha='center', va='bottom', fontsize=5.8)
        ax_.text(1, y0 + 0.17, 'better', ha='left', va='bottom', fontsize=6, style='italic')
        ax_.annotate('', xy=(1 - 0.012 * W, y0 + 0.2), xytext=(1 - 0.002 * W, y0 + 0.2),
                     arrowprops=dict(arrowstyle='-|>', lw=0.6, color=INK, mutation_scale=6))
        # critical difference
        xc = k - cd
        ax_.plot([xc, k], [y0 + 0.24, y0 + 0.24], color=INK, lw=1.0)
        for xx in (xc, k):
            ax_.plot([xx, xx], [y0 + 0.215, y0 + 0.265], color=INK, lw=0.8)
        ax_.text((xc + k) / 2, y0 + 0.28, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=6.3)
        # learners: best half to the left, rest to the right
        half = (len(ranks) + 1) // 2
        for j, (rk, nm) in enumerate(zip(ranks, names)):
            left = j < half; level = j if left else len(ranks) - 1 - j
            y_ = 0.70 - 0.115 * level
            xe = 1 - 0.02 * W if left else k + 0.02 * W
            col = HL.get(nm, INK); lw = 0.9 if nm in HL else 0.5
            ax_.plot([rk, rk, xe], [y0, y_, y_], color=col, lw=lw)
            lab = f'{nm} ({rk:.1f})' if left else f'({rk:.1f}) {nm}'
            ax_.text(xe - 0.008 * W if left else xe + 0.008 * W, y_, lab, ha='right' if left else 'left', va='center', fontsize=6.1,
                     color=col, fontweight='bold' if nm in HL else 'normal')
        # Nemenyi groups, packed into rows without overlap
        cl = cd_cliques(ranks, cd)
        for (a_, b_), r_ in zip(cl, cd_rows(cl, ranks)):
            yq = y0 - 0.06 - 0.045 * r_
            ax_.plot([ranks[a_] - 0.05, ranks[b_] + 0.05], [yq, yq], color=INK, lw=2.2, solid_capstyle='round')
        sub = f'{k} learners, {nb} folds' + (', Friedman p < 0.001' if pv is not None and pv < 0.001 else (f', Friedman p = {pv:.3f}' if pv is not None else '')) if nb else ''
        ax_.text(x_l, 1.6, f'({chr(97 + i)}) {PN[pr]}', ha='left', va='top', fontsize=8, fontweight='bold')
        ax_.text(x_l, 1.49, sub, ha='left', va='top', fontsize=6.5)
    fig_.savefig(path, bbox_inches='tight'); plt_.close(fig_)

for t, tl in [('kwh_delivered', 'energy'), ('duration_hours', 'duration')]:
    cd_figure(t, OUT / f'si_cd_{tl}.pdf')

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'], 'font.size': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
C = {'blue': '#0072B2', 'sky': '#56B4E9', 'verm': '#D55E00', 'orange': '#E69F00', 'grey': '#8C959F', 'black': '#000000'}
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
for row, (ww, tt, net) in enumerate([(w, tr, 'ACN'), (pa_w, pa_tr, 'Palo Alto')]):
    for ax, (t, unit, title) in zip(axs[row, :2], [('kwh_delivered', 'kWh', 'energy by driver history'), ('duration_hours', 'h', 'duration by driver history')]):
        z = ww[(ww.target == t) & (ww.learner == 'ft')].copy(); z['k'] = z.k_bucket.astype(str); z = z.set_index('k').loc[order]; x = np.arange(7)
        ax.plot(x, z.mae_random, '--o', color=C['grey'], ms=3, label='Random split'); ax.plot(x, z.mae_warm, '-o', color=C['blue'], ms=3, label='Unseen, history')
        ax.plot(x, z.mae_cold, '-s', color=C['verm'], ms=3, label='Cold start'); ax.set_xticks(x); ax.set_xticklabels(order, fontsize=5.8)
        ax.set_xlabel("Driver's completed sessions before this one"); ax.set_ylabel(f'MAE ({unit})'); ax.set_title(f'{net}, {title}', loc='left', fontsize=8)
    z = tt[tt.learner == 'ft']
    for st, lab, col in [('A', 'Driver seen', C['blue']), ('C', 'Unseen, history', C['sky']), ('D', 'Cold start', C['verm'])]:
        q = z[z.setting == st].sort_values('unmet_share_pct'); axs[row, 2].plot(q.unmet_share_pct, q.flex_vs_oracle_pct, '-o', ms=2.6, lw=1.0, color=col, label=lab, markevery=max(1, len(q) // 10))
    axs[row, 2].set_xlim(0, 30); axs[row, 2].set_ylim(0, 102); axs[row, 2].axvline(2, color=C['grey'], lw=0.6, ls=':')
    axs[row, 2].set_xlabel('Undelivered energy (% of achievable)'); axs[row, 2].set_ylabel('Flexibility retained (% of oracle)')
    axs[row, 2].set_title(f'{net}, per-session charging', loc='left', fontsize=8)
for ax_, l_ in zip(axs.ravel(), 'abcdef'):
    ax_.annotate(l_, xy=(-0.16, 1), xycoords='axes fraction', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight='bold', va='bottom')
h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.03), fontsize=6.8)
axs[0, 2].legend(frameon=False, fontsize=6.3, loc='lower right')
fig.savefig(OUT / 'si_ft.pdf', bbox_inches='tight'); plt.close(fig)
print(f'Supplementary Figures S1, S2 and S4 written to {OUT}')
