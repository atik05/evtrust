"""Figures 2 to 4 and Supplementary Fig. S3 from the pipeline outputs of both networks.

    python make_figures.py ACN_RESULTS_DIR PALOALTO_FULL_REPORT.xlsx OUT_DIR [ACNSIM_RESULTS_DIR]

ACNSIM_RESULTS_DIR holds the ACN-Sim tables of the robustness run (default: results/robustness/acn of this
repository); they draw the site-level panels (d) to (f) of Fig. 4.

Each figure is written as vector PDF (fonts embedded), editable SVG (text kept as text) and a 600 dpi PNG.

Figures 2 to 4 are written at the IEEE two-column width (7.16 in, 515.5 pt) and printed unscaled, so that all
text is at least 8 pt at print size, as IEEE recommends; colours follow the Okabe-Ito palette, so that colour
is never the only cue.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, PX, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
RS = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).resolve().parents[1] / 'results' / 'robustness' / 'acn'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import *  # noqa: E402,F401,F403  shared style: sizes, colours, panel headings, save()
from figstyle import save as _save  # noqa: E402


def save(fig, name, width=TEXTWIDTH):
    _save(fig, OUT / name, width)


_SHEETS = pd.ExcelFile(PX).sheet_names
def P(name):
    hits = [s for s in _SHEETS if s == name or name.startswith(s) or s.startswith(name)]
    assert len(hits) == 1, (name, hits)
    return pd.read_excel(PX, sheet_name=hits[0])


def grid(ax, axis='y'):
    ax.grid(False)            # no gridlines





# figure heights (mm) and grid margins; the widths are fixed at the IEEE text width by save()
F2H, F2HS, F2T, F2B = 84.02, 0.8, 0.9, 0.21
F3H, F3HS, F3T, F3B = 77.34, 0.85, 0.9, 0.21
F4H, F4HS, F4T, F4B = 98, 0.66, 0.925, 0.205
NETS = [('ACN', 'ACN workplace sites'), ('PA', 'Palo Alto public network')]
SHORT = {'ACN': 'ACN', 'PA': 'Palo Alto'}

# ------------------------------------------------------------ Figure 2: protocols and model choice
def protocol_table(df):
    return df.groupby(['target', 'protocol', 'model'])[['mae_mean', 'mae_std']].mean().reset_index()   # sites weighted equally
acc = {'ACN': protocol_table(pd.read_csv(R / 'metrics/regression_metrics_summary.csv')),
       'PA': protocol_table(P('RQ1 Accuracy by protocol'))}
prot = ['random', 'chronological', 'site_disjoint', 'user_disjoint']
plab = {'ACN': ['Random', 'Chrono-\nlogical', 'Unseen\nsite', 'Unseen\ndriver'],
        'PA': ['Random', 'Chrono-\nlogical', 'Unseen\nlocation', 'Unseen\ndriver']}
fig = plt.figure(figsize=(TEXTWIDTH, F2H * MM))          # printed at the full text width, unscaled
gs = fig.add_gridspec(2, 2, hspace=F2HS, wspace=0.08, left=0.075, right=0.585, top=F2T, bottom=F2B)
gr = fig.add_gridspec(1, 1, left=0.75, right=0.855, top=F2T, bottom=F2B)     # rank panel; its labels fill the margins
# rows are the targets and columns the networks, so that each row shares its MAE axis
panels = {('energy', 'ACN'): ('a', gs[0, 0]), ('energy', 'PA'): ('b', gs[0, 1]),
          ('duration', 'ACN'): ('d', gs[1, 0]), ('duration', 'PA'): ('e', gs[1, 1])}
ylim = {t: 1.25 * max(((acc[n][(acc[n].target == t) & acc[n].model.isin(['catboost', 'torch_ft'])].mae_mean
                        + acc[n][(acc[n].target == t) & acc[n].model.isin(['catboost', 'torch_ft'])].mae_std.fillna(0)).max())
                      for n, _ in NETS) for t in ['energy', 'duration']}
row_ax = {}
for (t, n), (letter, spec) in panels.items():
    ax = fig.add_subplot(spec, sharey=row_ax.get(t)); s_ = acc[n]; unit = 'kWh' if t == 'energy' else 'h'
    x = np.arange(len(prot)); w = 0.36
    for k, (mod, key) in enumerate([('catboost', 'catboost'), ('torch_ft', 'ft')]):
        lab, col = MODEL[key]
        z = s_[(s_.target == t) & (s_.model == mod)].set_index('protocol').loc[prot]
        ax.bar(x + (k - 0.5) * w, z.mae_mean, w, yerr=z.mae_std.fillna(0), color=col, label=lab,
               error_kw=ERR, edgecolor='white', linewidth=0.3)
    zz = s_[(s_.target == t) & (s_.model.isin(['catboost', 'torch_ft']))]
    base = s_[(s_.target == t) & (s_.model == 'catboost') & (s_.protocol == 'random')].mae_mean.iloc[0]
    for i, p_ in enumerate(prot[1:], start=1):
        g = zz[zz.protocol == p_]; v = s_[(s_.target == t) & (s_.model == 'catboost') & (s_.protocol == p_)].mae_mean.iloc[0]
        ax.text(i, (g.mae_mean + g.mae_std.fillna(0)).max() + 0.035 * ylim[t], pct_label(100 * (v / base - 1)),
                ha='center', fontsize=8, color=C['blue'])
    ax.set_ylim(0, ylim[t]); ax.set_xlim(-0.56, 3.56); ax.set_xticks(x); ax.set_xticklabels(plab[n]); grid(ax)
    if n == 'ACN':
        ax.set_ylabel(f'{t.capitalize()} MAE ({unit})'); row_ax[t] = ax
    else:
        ax.tick_params(labelleft=False)
    head(ax, letter, f'{SHORT[n]}, {t}')
    if letter == 'a':
        h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=((0.075 + 0.585) / 2, -0.01))
fr = pd.read_csv(R / 'benchmark/friedman.csv'); fr = fr[fr.target == 'kwh_delivered']
a = fr[fr.protocol == 'random'].set_index('model').average_rank; b = fr[fr.protocol == 'site_transfer'].set_index('model').average_rank
names = {'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'random_forest': 'Random forest', 'hgb': 'HistGB', 'extra_trees': 'Extra trees',
         'tabpfn': 'TabPFN v2', 'ft': 'FT-Transformer', 'mlp': 'MLP', 'catboost': 'CatBoost', 'ridge': 'Ridge',
         'user_history': 'Driver history', 'mean': 'Historical mean'}
ax = fig.add_subplot(gr[0])
hi = {'lightgbm': C['verm'], 'xgboost': C['verm'], 'ridge': C['green'], 'user_history': C['green']}
mods = list(a.index); la = repel([a[m_] for m_ in mods], 0.8); lb = repel([b[m_] for m_ in mods], 0.8)
for k, mod in enumerate(mods):
    col = hi.get(mod, C['grey']); lw = 1.5 if mod in hi else 0.8; tc = col if mod in hi else C['black']
    ax.plot([0, 1], [a[mod], b[mod]], '-o', color=col, lw=lw, ms=2.6, zorder=3 if mod in hi else 2)
    ax.plot([-0.05, 0], [la[k], a[mod]], color=col, lw=0.4); ax.plot([1, 1.05], [b[mod], lb[k]], color=col, lw=0.4)
    ax.text(-0.07, la[k], f'{names[mod]} {a[mod]:.1f}', ha='right', va='center', fontsize=8, color=tc)
    ax.text(1.07, lb[k], f'{names[mod]} {b[mod]:.1f}', ha='left', va='center', fontsize=8, color=tc)
ax.set_xlim(-0.12, 1.12); ax.set_ylim(12.6, 0.4); ax.set_xticks([0, 1]); ax.set_xticklabels(['Random', 'Unseen\nsite'])
ax.spines['left'].set_visible(False); ax.set_yticks([])
head(ax, 'c', 'ACN, mean rank of the learners, energy')
save(fig, 'fig2_protocols')

# ------------------------------------------------------------ Figure 3: cold start, 2 x 3 layout
dec = {'ACN': pd.read_csv(R / 'paper/coldstart_decomposition.csv'), 'PA': P('Cold-start decomposition')}
wu = {'ACN': pd.read_csv(R / 'paper/coldstart_warmup.csv'), 'PA': P('Cold-start warm-up curve')}
comps = ['future-session leakage', 'unseen-driver cost', 'value of own history', 'total gap']
clab = ['Random-split leakage', 'Unseen-driver cost', 'Value of own history', 'Total gap']
fig = plt.figure(figsize=(TEXTWIDTH, F3H * MM))          # printed at the full text width, unscaled
outer = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.25], wspace=0.24, left=0.16, right=0.995, top=F3T, bottom=F3B)
left = outer[0].subgridspec(2, 2, hspace=F3HS, wspace=0.08); right = outer[1].subgridspec(2, 1, hspace=F3HS)
axs = np.array([[fig.add_subplot(left[r, 0]), fig.add_subplot(left[r, 1]), fig.add_subplot(right[r])] for r in range(2)])
for r, (n, nlab) in enumerate(NETS):
    for c, (t, title) in enumerate([('kwh_delivered', 'energy'), ('duration_hours', 'duration')]):
        xmax = max(dec[m][(dec[m].target == t) & dec[m].component.isin(comps)].ci_high.max() for m, _ in NETS) * 1.08
        ax = axs[r, c]; d = dec[n]; y = np.arange(len(comps)); h = 0.36
        for k, l in enumerate(['catboost', 'ft']):
            lab, col = MODEL[l]
            z = d[(d.target == t) & (d.learner == l)].set_index('component').loc[comps]
            ax.barh(y + (k - 0.5) * h, z.effect_pct, h, xerr=[z.effect_pct - z.ci_low, z.ci_high - z.effect_pct],
                    color=col, label=lab, error_kw=ERR, edgecolor='white', linewidth=0.3)
        lo = min(min(0, dec[m][dec[m].target == t].ci_low.min() * 1.2) for m, _ in NETS)
        ax.set_xlim(lo, xmax); ax.set_yticks(y); ax.invert_yaxis()
        ax.set_yticklabels(clab if c == 0 else [])
        ax.set_xlabel('Change in MAE (%)'); grid(ax, 'x')
        head(ax, 'abcdef'[3 * r + c], f'{SHORT[n]}, {title}')
    order = ['0', '1', '2', '3-4', '5-9', '10-19', '20+']
    ax = axs[r, 2]; z = wu[n][(wu[n].target == 'kwh_delivered') & (wu[n].learner == 'catboost')].copy()
    z['k'] = z.k_bucket.astype(str); z = z.set_index('k').loc[order]; x = np.arange(len(order))
    ax.fill_between(x, z.mae_warm, z.mae_cold, color=C['verm'], alpha=0.09, lw=0)
    setting_line(ax, x, z.mae_random, 'seen'); setting_line(ax, x, z.mae_warm, 'history'); setting_line(ax, x, z.mae_cold, 'cold')
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_xlabel('Completed sessions of the driver'); grid(ax)
    ax.set_ylabel('MAE (kWh)'); head(ax, 'abcdef'[3 * r + 2], f'{SHORT[n]}, energy by driver history')
h1, l1 = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[0, 2].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, frameon=False, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.01),
           columnspacing=1.3, handlelength=1.8, handletextpad=0.4)
save(fig, 'fig3_coldstart')

# ------------------------------------------------------------ Supplementary Fig. S3: information availability
av = {'ACN': pd.read_csv(R / 'paper/feature_availability.csv'), 'PA': P('Information availability')}
scopes = {'ACN': (['all', 'caltech', 'jpl', 'office001'], ['All sites', 'Caltech', 'JPL', 'Office001']),
          'PA': (['all', 'downtown', 'california_ave', 'mitchell_park', 'rinconada'], ['All', 'Downtown', 'California\nAve', 'Mitchell\nPark', 'Rinconada'])}
sets = [('plug_in', 'At plug-in', C['blue']), ('plus_charging', '+ charging duration', C['orange']), ('plus_departure', '+ session duration', C['verm'])]
fig, axs = plt.subplots(1, 2, figsize=(SI_WIDTH, 64 * MM), gridspec_kw={'width_ratios': [4, 5], 'wspace': 0.12, 'top': 0.86}, sharey=True)
top = max(av[n][av[n].scope.isin(scopes[n][0])].mae.max() for n, _ in NETS) * 1.18
for ax, (n, nlab), letter in zip(axs, NETS, 'ab'):
    sc, sl = scopes[n]; x = np.arange(len(sc)); w = 0.27
    for k, (fs, lab, col) in enumerate(sets):
        z = av[n][av[n].feature_set == fs].set_index('scope').loc[sc]
        ax.bar(x + (k - 1) * w, z.mae, w, color=col, label=lab, edgecolor='white', linewidth=0.3)
        if fs == 'plus_charging':
            zd = av[n][av[n].feature_set == 'plus_departure'].set_index('scope').loc[sc]
            for i in range(len(sc)):
                ax.text(i + w / 2, max(z.mae.iloc[i], zd.mae.iloc[i]) + 0.03 * top, pct_label(z.change_vs_plug_in_pct.iloc[i]),
                        ha='center', fontsize=8, color=C['black'])
    ax.set_xticks(x); ax.set_xticklabels(sl); ax.set_ylim(0, top); grid(ax)
    head(ax, letter, f'{SHORT[n]}, energy error by information')
axs[0].set_ylabel('Energy MAE (kWh)'); axs[1].legend(frameon=False, loc='upper right')
save(fig, 'si_fig_availability', width=SI_WIDTH)     # Supplementary text width, unscaled

# ------------------------------------------------------------ Figure 4: operation (per-session and site level)
tr = {'ACN': pd.read_csv(R / 'paper/impact_tradeoff.csv'), 'PA': P('Smart charging trade-off')}
tr = {n: d[d.learner == 'catboost'] for n, d in tr.items()}
rc = pd.read_csv(R / 'operational/request_curves.csv'); rc = rc[(rc.learner == 'catboost') & (rc.group == 'all')]
need = pd.read_csv(RS / 'acnsim_capacity_needed.csv'); need['tau'] = need['tau'].astype(str)
con = pd.read_csv(RS / 'acnsim_capacity_contrasts.csv')
fig, axs = plt.subplots(2, 3, figsize=(TEXTWIDTH, F4H * MM), gridspec_kw={'hspace': F4HS, 'wspace': 0.3,      # full text width, unscaled
                        'left': 0.08, 'right': 0.995, 'top': F4T, 'bottom': F4B})
KEY = {'A': 'seen', 'C': 'history', 'D': 'cold', 'request': 'request'}


def curve(ax, df, keys):
    for k in keys:
        z = df[df.setting == k].sort_values('unmet_share_pct')
        # the curves have 100 quantile levels; markers are spaced evenly along each line, not by quantile
        setting_line(ax, z.unmet_share_pct, z.flex_vs_oracle_pct, KEY[k], markevery=0.1)
    ax.axvline(2, color=C['grey'], lw=0.6, ls=':'); ax.text(2.6, 93, '2%', fontsize=8, color=C['black'])
    ax.set_xlim(0, 30); ax.set_ylim(0, 102); ax.set_xlabel('Undelivered energy (%)')


for ax, (n, _), letter in zip(axs[0, :2], NETS, 'ab'):
    curve(ax, tr[n], ['A', 'C', 'D']); head(ax, letter, f'{SHORT[n]}, per session')
curve(axs[0, 2], rc, ['A', 'D', 'request']); head(axs[0, 2], 'c', 'ACN, sessions with a request')
axs[0, 0].set_ylabel('Flexibility retained (%)')
RATED = {'caltech': 150.0, 'jpl': 195.0}
ZERO = 0.004                      # zero undelivered energy is drawn here and labelled 0 below an axis break
pol = [(('llf', 'oracle', 'na'), 'oracle'), (('llf', 'deployment', '0.5'), 'history'),
       (('llf', 'cold', '0.5'), 'cold'), (('fcfs', 'none', 'na'), 'fcfs')]
for ax, site, title, letter in [(axs[1, 0], 'caltech', 'Caltech', 'd'), (axs[1, 1], 'jpl', 'JPL', 'e')]:
    for (sch, fc, tau), key in pol:
        z = need[(need.site == site) & (need.scheduler == sch) & (need.forecast == fc) & (need.tau == tau)].iloc[0]
        d = {float(k): float(v) for k, v in (kv.split(':') for kv in z.undelivered_pct_by_cap.split(';'))}
        x = np.array(sorted(d)); y = np.array([d[k] for k in x]); keep = x >= 0.3
        setting_line(ax, 100 * x[keep], np.where(y[keep] > 0, np.clip(y[keep], 0.006, None), ZERO), key)
    ax.set_yscale('log'); ax.set_ylim(0.0022, 60); ax.axhline(2, color=C['grey'], lw=0.6, ls=':')
    ax.text(101, 2.3, '2%', fontsize=8, color=C['black'], ha='right', va='bottom')
    ax.set_yticks([ZERO, 0.01, 0.1, 1, 10]); ax.set_yticklabels(['0', '0.01', '0.1', '1', '10'])
    ax.minorticks_off()
    yb = np.sqrt(ZERO * 0.01)                                  # axis break between 0 and 0.01
    for dy in (0.92, 1.08):
        ax.plot([-0.022, 0.022], [yb * dy / 1.03, yb * dy * 1.03], color=C['black'], lw=0.6, clip_on=False,
                transform=ax.get_yaxis_transform())
    ax.set_xlim(27, 103); ax.set_xticks([30, 40, 50, 60, 70, 100])
    ax.set_xlabel(f'Capacity (% of {RATED[site]:.0f} kVA rating)')
    if site == 'caltech':
        ax.set_ylabel('Undelivered energy (%)')
    head(ax, letter, f'{title}, ACN-Sim')
ax = axs[1, 2]
fcs = [('benchmark', 'seen'), ('deployment', 'history'), ('cold', 'cold')]
pos = 0; ticks = []
for site in ['caltech', 'jpl']:
    for fc, key in fcs:
        _, col, _ = SETTING[key]
        for tau, mk, off, fill in [(0.5, 'o', -0.15, col), (0.1, 's', 0.15, 'white')]:
            z = con[(con.site == site) & (con.tau == tau) &
                    (con.contrast == f'share of the perfect-information capacity value captured by {fc} (%)')].iloc[0]
            ax.errorbar(pos + off, z.estimate_fraction, yerr=[[z.estimate_fraction - z.ci_low], [z.ci_high - z.estimate_fraction]],
                        fmt=mk, ms=MS, color=col, mfc=fill, mec=col, mew=0.9, **ERR)
        pos += 1
    ticks.append(pos - 2); pos += 0.7
ax.axhline(0, color=C['grey'], lw=0.6)
ax.set_xticks(ticks); ax.set_xticklabels(['Caltech', 'JPL']); ax.set_ylim(-12, 108); ax.set_yticks([0, 50, 100]); ax.set_xlim(-0.6, pos - 0.4)
ax.set_ylabel('Capacity value (%)')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=C['grey'], ms=MS, label='Raw forecast'),
                   Line2D([], [], marker='s', ls='', color=C['grey'], mfc='white', ms=MS, label=r'Shifted, $\gamma=0.1$')],
          frameon=False, loc='upper left', ncol=1, bbox_to_anchor=(-0.02, 1.04), handletextpad=0.2)
head(ax, 'f', 'Capacity value of the forecast')
hs = [Line2D([], [], ls='--' if k == 'fcfs' else '-', marker=SETTING[k][2], color=SETTING[k][1], ms=MS, lw=LW, label=SETTING[k][0])
      for k in ['seen', 'history', 'cold', 'request', 'oracle', 'fcfs']]
fig.legend(handles=hs, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.01), columnspacing=2.0, handlelength=2.2)
save(fig, 'fig4_operation')
print('figures written to', OUT)
