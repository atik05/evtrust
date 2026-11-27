"""Supplementary Figures S1, S2 and S4 from the pipeline outputs of both networks.

    python make_si_figures.py ACN_RESULTS_DIR PALOALTO_FULL_REPORT.xlsx OUT_DIR

S1 and S2: critical-difference diagrams of the learner benchmark (energy, duration).
S4: cold-start dynamics and smart-charging trade-off for the FT-Transformer on both networks
(S3, information availability, is drawn by make_figures.py).
Each figure is written as vector PDF, SVG and PNG at the Supplementary text width (166 mm), printed unscaled.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import *  # noqa: E402,F401,F403  shared style: sizes, colours, panel headings, save()

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
    analyses, are drawn in their colours of Figs. 2 and 3 so that their position can be followed across protocols.
    Each panel is laid out in points: the learners are listed on the left in order of rank."""
    HL = {MODEL['catboost'][0]: MODEL['catboost'][1], MODEL['ft'][0]: MODEL['ft'][1]}
    fz_ = pd.read_csv(R / 'benchmark/friedman.csv'); fz_ = fz_[fz_.target == target]
    prots = ['random', 'driver_grouped', 'chronological', 'chronological_prepandemic', 'site_transfer', 'site_transfer_strict']
    ROW, TOP, HEAD, GAP = 9.6, 26.0, 18.0, 0.1       # label row spacing, space above the axis, heading band (pt), column gap

    def need(pr):
        z = fz_[fz_.protocol == pr].sort_values(['average_rank', 'model'])
        r_ = z.average_rank.to_numpy(float); c_ = cd_cliques(r_, float(z.nemenyi_cd.iloc[0])); rw = cd_rows(c_, r_)
        return TOP + 5 + 4 * (max(rw) + 1 if rw else 0) + 6 + ROW * (len(r_) - 1) + 5
    rowh = [max(need(prots[2 * r]), need(prots[2 * r + 1])) for r in range(3)]
    Hpt = sum(rowh) + 3 * HEAD
    fig_ = plt.figure(figsize=(SI_WIDTH, Hpt / 72))
    axs_ = []
    ytop_ = Hpt
    for r in range(3):
        ytop_ -= HEAD
        for c in range(2):
            ax_ = fig_.add_axes([c * (0.5 + GAP / 2), (ytop_ - rowh[r]) / Hpt, 0.5 - GAP / 2, rowh[r] / Hpt])
            ax_.set_ylim(0, rowh[r]); axs_.append(ax_)      # one unit is one point
        ytop_ -= rowh[r]
    for i, (ax_, pr) in enumerate(zip(axs_, prots)):
        z = fz_[fz_.protocol == pr].sort_values(['average_rank', 'model'])
        ranks = z.average_rank.to_numpy(float); names = [MN.get(m_, m_) for m_ in z.model]
        k = int(z.n_models.iloc[0]); cd = float(z.nemenyi_cd.iloc[0])
        nb = int(z.n_blocks.iloc[0]) if 'n_blocks' in z else None
        pv = float(z.friedman_p.iloc[0]) if 'friedman_p' in z else None
        # the rank axis fills the right 56% of the panel and the learner names the left part
        W = 11 / 0.56; x_l = 1 - 0.42 * W
        HP = ax_.get_ylim()[1]
        ax_.set_xlim(x_l, x_l + W); ax_.axis('off')
        y0 = HP - TOP
        ax_.plot([1, k], [y0, y0], color=C['black'], lw=0.8, solid_capstyle='butt')
        for r_ in range(1, k + 1):
            ax_.plot([r_, r_], [y0, y0 + 3], color=C['black'], lw=0.6)
            if k < 10 or r_ % 2 == 0:                     # every second rank is numbered when there are many learners
                ax_.text(r_, y0 + 4, str(r_), ha='center', va='bottom', fontsize=8)
        # critical difference, drawn from the worst rank
        xc = k - cd
        ax_.plot([xc, k], [y0 + 20, y0 + 20], color=C['black'], lw=1.0)
        for xx in (xc, k):
            ax_.plot([xx, xx], [y0 + 17.5, y0 + 22.5], color=C['black'], lw=0.8)
        ax_.text(xc - 0.15, y0 + 20, f'CD = {cd:.2f}', ha='right', va='center', fontsize=8)
        # Nemenyi groups, packed into rows without overlap, just below the axis
        cl = cd_cliques(ranks, cd); rows = cd_rows(cl, ranks)
        for (a_, b_), r_ in zip(cl, rows):
            yq = y0 - 5 - 4 * r_
            ax_.plot([ranks[a_] - 0.05, ranks[b_] + 0.05], [yq, yq], color=C['black'], lw=2.2, solid_capstyle='round')
        ytop = y0 - 5 - 4 * (max(rows) + 1 if rows else 0) - 6
        # learners: best at the top, each joined to its mean rank
        for j, (rk, nm) in enumerate(zip(ranks, names)):
            y_ = ytop - ROW * j
            col = HL.get(nm, C['black']); lw = 0.9 if nm in HL else 0.5
            ax_.plot([rk, rk, x_l + 0.36 * W], [y0, y_, y_], color=col, lw=lw)
            ax_.text(x_l + 0.35 * W, y_, f'{nm} ({rk:.1f})', ha='right', va='center', fontsize=8, color=col,
                     fontweight='bold' if nm in HL else 'normal')
        sub = f'{nb} folds' if nb else ''
        if pv is not None:
            sub += (', ' if sub else '') + ('Friedman p < 0.001' if pv < 0.001 else f'Friedman p = {pv:.3f}')
        head(ax_, chr(97 + i), f'{PN[pr]}, {sub}' if sub else PN[pr])
    save(fig_, str(path), SI_WIDTH)


for t, tl in [('kwh_delivered', 'energy'), ('duration_hours', 'duration')]:
    cd_figure(t, OUT / f'si_cd_{tl}')

w = pd.read_csv(R / 'paper/coldstart_warmup.csv'); tr = pd.read_csv(R / 'paper/impact_tradeoff.csv')
_PS = pd.ExcelFile(PAX).sheet_names
def PAS(name):
    hits = [x for x in _PS if x == name or name.startswith(x) or x.startswith(name)]
    assert len(hits) == 1, (name, hits)
    return pd.read_excel(PAX, sheet_name=hits[0])
pa_w = PAS('Cold-start warm-up curve'); pa_tr = PAS('Smart charging trade-off')
fig, axs = plt.subplots(2, 3, figsize=(SI_WIDTH, 112 * MM), gridspec_kw={'wspace': 0.42, 'hspace': 0.72, 'top': 0.93, 'bottom': 0.17})
order = ['0', '1', '2', '3-4', '5-9', '10-19', '20+']
KEY = {'A': 'seen', 'C': 'history', 'D': 'cold'}
for row, (ww, tt, net) in enumerate([(w, tr, 'ACN'), (pa_w, pa_tr, 'Palo Alto')]):
    for c, (ax, (t, unit, title)) in enumerate(zip(axs[row, :2], [('kwh_delivered', 'kWh', 'energy'), ('duration_hours', 'h', 'duration')])):
        z = ww[(ww.target == t) & (ww.learner == 'ft')].copy(); z['k'] = z.k_bucket.astype(str); z = z.set_index('k').loc[order]; x = np.arange(7)
        ax.fill_between(x, z.mae_warm, z.mae_cold, color=C['verm'], alpha=0.09, lw=0)
        setting_line(ax, x, z.mae_random, 'seen'); setting_line(ax, x, z.mae_warm, 'history'); setting_line(ax, x, z.mae_cold, 'cold')
        ax.set_xticks(x); ax.set_xticklabels([o.replace('-', '\u2013') for o in order], rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('Completed sessions'); ax.set_ylabel(f'MAE ({unit})')
        head(ax, 'abcdef'[3 * row + c], f'{net}, {title}')
    ax = axs[row, 2]; z = tt[tt.learner == 'ft']
    for st in ['A', 'C', 'D']:
        q = z[z.setting == st].sort_values('unmet_share_pct')
        setting_line(ax, q.unmet_share_pct, q.flex_vs_oracle_pct, KEY[st], markevery=0.1)
    ax.set_xlim(0, 30); ax.set_ylim(0, 102); ax.axvline(2, color=C['grey'], lw=0.6, ls=':')
    ax.text(2.6, 93, '2%', fontsize=8, color=C['black'])
    ax.set_xlabel('Undelivered energy (%)'); ax.set_ylabel('Flexibility retained (%)')
    head(ax, 'abcdef'[3 * row + 2], f'{net}, per session')
fig.legend(handles=setting_handles(['seen', 'history', 'cold']), frameon=False, ncol=3, loc='lower center',
           bbox_to_anchor=(0.5, -0.01), columnspacing=2.0, handlelength=2.2)
save(fig, str(OUT / 'si_ft'), SI_WIDTH)
print(f'Supplementary Figures S1, S2 and S4 written to {OUT}')
