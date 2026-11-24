"""Figures 2 to 4 and Supplementary Fig. S3 from the pipeline outputs of both networks.

    python make_figures.py ACN_RESULTS_DIR PALOALTO_FULL_REPORT.xlsx OUT_DIR [ACNSIM_RESULTS_DIR]

ACNSIM_RESULTS_DIR holds the ACN-Sim tables of the robustness run (default: results/robustness/acn of this
repository); they draw the site-level panels (d) to (f) of Fig. 4.

Each figure is written as vector PDF (fonts embedded), editable SVG (text kept as text) and a 600 dpi PNG.

Double-column figures are 190 mm wide and single-column figures 90 mm, with text of at least
6.5 pt at print size and the Okabe-Ito palette, so that colour is never the only cue.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, PX, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
RS = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).resolve().parents[1] / 'results' / 'robustness' / 'acn'
MM = 1 / 25.4
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
                     'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 6.8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600})
C = {'blue': '#0072B2', 'sky': '#56B4E9', 'verm': '#D55E00', 'orange': '#E69F00', 'green': '#009E73',
     'purple': '#CC79A7', 'grey': '#8C959F', 'black': '#000000'}


def repel(vals, gap):
    """Shift label positions apart by at least gap, keeping their order and mean."""
    order = np.argsort(vals); pos = np.array(vals, float)[order]
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap:
                d = (gap - (pos[i] - pos[i - 1])) / 2; pos[i - 1] -= d; pos[i] += d; moved = True
        if not moved:
            break
    out = np.empty(len(pos)); out[order] = pos
    return out


def tag(ax, s, x=-0.14):
    ax.annotate(s, xy=(x, 1), xycoords='axes fraction', xytext=(0, 6), textcoords='offset points',
                fontsize=9, fontweight='bold', va='bottom')


def save(fig, name):
    for ext, kw in [('pdf', {}), ('svg', {}), ('png', {'dpi': 1000})]:
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', **kw)
    plt.close(fig)


_SHEETS = pd.ExcelFile(PX).sheet_names
def P(name):
    hits = [s for s in _SHEETS if s == name or name.startswith(s) or s.startswith(name)]
    assert len(hits) == 1, (name, hits)
    return pd.read_excel(PX, sheet_name=hits[0])


def pct_label(v):
    return f'{v:+.0f}%'.replace('-', '\u2212')


def grid(ax, axis='y'):
    ax.grid(False)            # no gridlines





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
fig = plt.figure(figsize=(190 * MM, 112 * MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 1.45], hspace=0.6, wspace=0.36)
panels = {('ACN', 'energy'): ('a', gs[0, 0]), ('ACN', 'duration'): ('b', gs[0, 1]),
          ('PA', 'energy'): ('d', gs[1, 0]), ('PA', 'duration'): ('e', gs[1, 1])}
ylim = {t: 1.25 * max(((acc[n][(acc[n].target == t) & acc[n].model.isin(['catboost', 'torch_ft'])].mae_mean
                        + acc[n][(acc[n].target == t) & acc[n].model.isin(['catboost', 'torch_ft'])].mae_std.fillna(0)).max())
                      for n, _ in NETS) for t in ['energy', 'duration']}
for (n, t), (letter, spec) in panels.items():
    ax = fig.add_subplot(spec); s = acc[n]; unit = 'kWh' if t == 'energy' else 'h'
    x = np.arange(len(prot)); w = 0.36
    for k, (mod, lab, col) in enumerate([('catboost', 'CatBoost', C['blue']), ('torch_ft', 'FT-Transformer', C['orange'])]):
        z = s[(s.target == t) & (s.model == mod)].set_index('protocol').loc[prot]
        ax.bar(x + (k - 0.5) * w, z.mae_mean, w, yerr=z.mae_std.fillna(0), color=col, label=lab, capsize=1.5,
               error_kw={'elinewidth': 0.6}, edgecolor='white', linewidth=0.3)
    zz = s[(s.target == t) & (s.model.isin(['catboost', 'torch_ft']))]
    base = s[(s.target == t) & (s.model == 'catboost') & (s.protocol == 'random')].mae_mean.iloc[0]
    for i, p in enumerate(prot[1:], start=1):
        g = zz[zz.protocol == p]; v = s[(s.target == t) & (s.model == 'catboost') & (s.protocol == p)].mae_mean.iloc[0]
        ax.text(i, (g.mae_mean + g.mae_std.fillna(0)).max() + 0.035 * ylim[t], pct_label(100 * (v / base - 1)),
                ha='center', fontsize=6.5, color=C['blue'])
    ax.set_ylim(0, ylim[t]); ax.set_xticks(x); ax.set_xticklabels(plab[n], fontsize=6.3); ax.set_ylabel(f'MAE ({unit})'); grid(ax)
    ax.set_title(f'{SHORT[n]}, {t}', loc='left'); tag(ax, letter)
    if letter == 'a':
        h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, ['CatBoost', 'FT-Transformer'], frameon=False, ncol=2,
           loc='upper left', bbox_to_anchor=(0.05, 1.0))
fr = pd.read_csv(R / 'benchmark/friedman.csv'); fr = fr[fr.target == 'kwh_delivered']
a = fr[fr.protocol == 'random'].set_index('model').average_rank; b = fr[fr.protocol == 'site_transfer'].set_index('model').average_rank
names = {'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'random_forest': 'Random forest', 'hgb': 'HistGB', 'extra_trees': 'Extra trees',
         'tabpfn': 'TabPFN v2', 'ft': 'FT-Transformer', 'mlp': 'MLP', 'catboost': 'CatBoost', 'ridge': 'Ridge',
         'user_history': 'Driver history', 'mean': 'Historical mean'}
ax = fig.add_subplot(gs[:, 2])
hi = {'lightgbm': C['verm'], 'xgboost': C['verm'], 'ridge': C['green'], 'user_history': C['green']}
mods = list(a.index); la = repel([a[m_] for m_ in mods], 0.62); lb = repel([b[m_] for m_ in mods], 0.62)
for k, mod in enumerate(mods):
    col = hi.get(mod, C['grey']); lw = 1.5 if mod in hi else 0.8; tc = col if mod in hi else C['black']
    ax.plot([0, 1], [a[mod], b[mod]], '-o', color=col, lw=lw, ms=2.6, zorder=3 if mod in hi else 2)
    ax.plot([-0.05, 0], [la[k], a[mod]], color=col, lw=0.4); ax.plot([1, 1.05], [b[mod], lb[k]], color=col, lw=0.4)
    ax.text(-0.07, la[k], f'{names[mod]} {a[mod]:.1f}', ha='right', va='center', fontsize=6.3, color=tc)
    ax.text(1.07, lb[k], f'{names[mod]} {b[mod]:.1f}', ha='left', va='center', fontsize=6.3, color=tc)
ax.set_xlim(-1.25, 1.9); ax.set_ylim(12.9, 0.1); ax.set_xticks([0, 1]); ax.set_xticklabels(['Random', 'Unseen site'])
ax.set_ylabel('Mean Friedman rank, energy (1 = best)'); ax.spines['left'].set_visible(False); ax.set_yticks([])
ax.set_title('ACN, learner ranks for energy', loc='left'); tag(ax, 'c', -0.06)
save(fig, 'fig2_protocols')

# ------------------------------------------------------------ Figure 3: cold start, 2 x 3 layout
dec = {'ACN': pd.read_csv(R / 'paper/coldstart_decomposition.csv'), 'PA': P('Cold-start decomposition')}
wu = {'ACN': pd.read_csv(R / 'paper/coldstart_warmup.csv'), 'PA': P('Cold-start warm-up curve')}
comps = ['future-session leakage', 'unseen-driver cost', 'value of own history', 'total gap']
clab = ['Random-split\nleakage', 'Unseen-driver\ncost', 'Value of own\nhistory', 'Total gap']
fig = plt.figure(figsize=(182 * MM, 85 * MM))
outer = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.25], wspace=0.2, left=0.105, right=0.995, top=0.95, bottom=0.18)
left = outer[0].subgridspec(2, 2, hspace=0.7, wspace=0.1); right = outer[1].subgridspec(2, 1, hspace=0.7)
axs = np.array([[fig.add_subplot(left[r, 0]), fig.add_subplot(left[r, 1]), fig.add_subplot(right[r])] for r in range(2)])
for r, (n, nlab) in enumerate(NETS):
    for c, (t, title) in enumerate([('kwh_delivered', 'energy'), ('duration_hours', 'duration')]):
        xmax = max(dec[m][(dec[m].target == t) & dec[m].component.isin(comps)].ci_high.max() for m, _ in NETS) * 1.08
        ax = axs[r, c]; d = dec[n]; y = np.arange(len(comps)); h = 0.36
        for k, (l, lab, col) in enumerate([('catboost', 'CatBoost', C['blue']), ('ft', 'FT-Transformer', C['orange'])]):
            z = d[(d.target == t) & (d.learner == l)].set_index('component').loc[comps]
            ax.barh(y + (k - 0.5) * h, z.effect_pct, h, xerr=[z.effect_pct - z.ci_low, z.ci_high - z.effect_pct],
                    color=col, label=lab, capsize=1.5, error_kw={'elinewidth': 0.6})
        lo = min(min(0, dec[m][dec[m].target == t].ci_low.min() * 1.2) for m, _ in NETS)
        ax.set_xlim(lo, xmax); ax.set_yticks(y); ax.invert_yaxis()
        ax.set_yticklabels(clab if c == 0 else [])
        ax.set_xlabel('Change in MAE (%), 95% CI'); grid(ax, 'x')
        ax.set_title(f'{SHORT[n]}, {title}', loc='left'); tag(ax, 'abcdef'[3 * r + c], x=-0.42 if c == 0 else -0.08)
    order = ['0', '1', '2', '3-4', '5-9', '10-19', '20+']
    ax = axs[r, 2]; z = wu[n][(wu[n].target == 'kwh_delivered') & (wu[n].learner == 'catboost')].copy()
    z['k'] = z.k_bucket.astype(str); z = z.set_index('k').loc[order]; x = np.arange(len(order))
    ax.plot(x, z.mae_random, '--o', color=C['grey'], ms=3, label='Random split (driver seen)')
    ax.plot(x, z.mae_warm, '-o', color=C['blue'], ms=3, label='Unseen driver, own history')
    ax.plot(x, z.mae_cold, '-s', color=C['verm'], ms=3, label='Unseen driver, cold start')
    ax.fill_between(x, z.mae_warm, z.mae_cold, color=C['verm'], alpha=0.09, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_xlabel('Completed sessions of the driver'); grid(ax)
    ax.set_ylabel('Energy MAE (kWh)'); ax.set_title(f'{SHORT[n]}, energy by driver history', loc='left')
    tag(ax, 'abcdef'[3 * r + 2], x=-0.2)
h1, l1 = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[0, 2].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, frameon=False, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0.0),
           columnspacing=1.4, handlelength=1.8, handletextpad=0.5)
save(fig, 'fig3_coldstart')

# ------------------------------------------------------------ Supplementary Fig. S3: information availability
av = {'ACN': pd.read_csv(R / 'paper/feature_availability.csv'), 'PA': P('Information availability')}
scopes = {'ACN': (['all', 'caltech', 'jpl', 'office001'], ['All sites', 'Caltech', 'JPL', 'Office001']),
          'PA': (['all', 'downtown', 'california_ave', 'mitchell_park', 'rinconada'], ['All', 'Downtown', 'California\nAve', 'Mitchell\nPark', 'Rinconada'])}
sets = [('plug_in', 'At plug-in', C['blue']), ('plus_charging', '+ charging duration', C['orange']), ('plus_departure', '+ session duration', C['verm'])]
fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 64 * MM), gridspec_kw={'width_ratios': [4, 5], 'wspace': 0.22}, sharey=True)
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
                        ha='center', fontsize=6.3, color=C['black'])
    ax.set_xticks(x); ax.set_xticklabels(sl, fontsize=6.5); ax.set_ylim(0, top); grid(ax)
    ax.set_title(f'{SHORT[n]}, energy error by information', loc='left'); tag(ax, letter)
axs[0].set_ylabel('Energy MAE (kWh)'); axs[1].legend(frameon=False, loc='upper right', fontsize=6.3)
save(fig, 'si_fig_availability')

# ------------------------------------------------------------ Figure 4: operation (per-session and site level)
tr = {'ACN': pd.read_csv(R / 'paper/impact_tradeoff.csv'), 'PA': P('Smart charging trade-off')}
tr = {n: d[d.learner == 'catboost'] for n, d in tr.items()}
rc = pd.read_csv(R / 'operational/request_curves.csv'); rc = rc[(rc.learner == 'catboost') & (rc.group == 'all')]
need = pd.read_csv(RS / 'acnsim_capacity_needed.csv'); need['tau'] = need['tau'].astype(str)
con = pd.read_csv(RS / 'acnsim_capacity_contrasts.csv')
fig, axs = plt.subplots(2, 3, figsize=(190 * MM, 118 * MM), gridspec_kw={'hspace': 0.62, 'wspace': 0.38})
styles = {'A': ('Driver seen', C['blue'], '-o'), 'C': ('Unseen driver, own history', C['sky'], '-^'),
          'D': ('Unseen driver, cold start', C['verm'], '-s'), 'request': ('Driver request', C['orange'], '-D')}


def curve(ax, df, keys):
    for k in keys:
        z = df[df.setting == k].sort_values('unmet_share_pct')
        lab, col, mk = styles[k]
        # the curves have 100 quantile levels; about ten markers identify each line without crowding it
        ax.plot(z.unmet_share_pct, z.flex_vs_oracle_pct, mk, color=col, ms=3.0, lw=1.2, label=lab, markevery=max(1, len(z) // 10))
    ax.axvline(2, color=C['grey'], lw=0.6, ls=':'); ax.text(2.6, 93, '2%', fontsize=6.3, color=C['black'])
    ax.set_xlim(0, 30); ax.set_ylim(0, 102); ax.set_xlabel('Undelivered energy (% of achievable)')
    ax.set_ylabel('Flexibility retained (% of oracle)')


for ax, (n, _), letter in zip(axs[0, :2], NETS, 'ab'):
    curve(ax, tr[n], ['A', 'C', 'D']); ax.set_title(f'{SHORT[n]}, per-session charging', loc='left'); tag(ax, letter)
axs[0, 0].legend(frameon=False, loc='lower right', fontsize=6.2)
curve(axs[0, 2], rc, ['A', 'D', 'request']); axs[0, 2].set_title('ACN, sessions with a request', loc='left'); tag(axs[0, 2], 'c')
axs[0, 2].legend(frameon=False, loc='lower right', fontsize=6.2)
RATED = {'caltech': 150.0, 'jpl': 195.0}
pol = [(('llf', 'oracle', 'na'), 'Perfect information', C['black'], '-', 'o'),
       (('llf', 'deployment', '0.5'), 'Unseen driver, own history', C['sky'], '-', '^'),
       (('llf', 'cold', '0.5'), 'Unseen driver, cold start', C['verm'], '-', 's'),
       (('fcfs', 'none', 'na'), 'No forecast (first come, first served)', C['grey'], '--', 'D')]
for ax, site, title, letter in [(axs[1, 0], 'caltech', 'Caltech', 'd'), (axs[1, 1], 'jpl', 'JPL', 'e')]:
    for (sch, fc, tau), lab, col, ls, mk in pol:
        z = need[(need.site == site) & (need.scheduler == sch) & (need.forecast == fc) & (need.tau == tau)].iloc[0]
        d = {float(k): float(v) for k, v in (kv.split(':') for kv in z.undelivered_pct_by_cap.split(';'))}
        x = np.array(sorted(d)); y = np.array([d[k] for k in x]); keep = x >= 0.3
        ax.plot(100 * x[keep], np.clip(y[keep], 0.005, None), ls, marker=mk, color=col, ms=2.8, label=lab)
    ax.set_yscale('log'); ax.set_ylim(0.004, 60); ax.axhline(2, color=C['grey'], lw=0.6, ls=':')
    ax.text(101, 2.3, '2%', fontsize=6.3, color=C['black'], ha='right', va='bottom')
    ax.set_yticks([0.01, 0.1, 1, 10]); ax.set_yticklabels(['0.01', '0.1', '1', '10'])
    ax.set_xlim(27, 103); ax.set_xticks([30, 40, 50, 60, 70, 100])
    ax.set_xlabel(f'Transformer capacity (% of {RATED[site]:.0f} kW rating)'); ax.set_ylabel('Undelivered energy (%)')
    ax.set_title(f'{title}, ACN-Sim', loc='left'); tag(ax, letter)
h_, l_ = axs[1, 0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0.06, 0.035), fontsize=6.3, handlelength=2.4)
ax = axs[1, 2]
fcs = [('benchmark', 'Driver seen', C['blue']), ('deployment', 'Unseen driver, own history', C['sky']), ('cold', 'Cold start', C['verm'])]
pos = 0; ticks = []
for site in ['caltech', 'jpl']:
    for fc, _, col in fcs:
        for tau, mk, off, fill in [(0.5, 'o', -0.15, col), (0.1, 's', 0.15, 'white')]:
            z = con[(con.site == site) & (con.tau == tau) &
                    (con.contrast == f'share of the perfect-information capacity value captured by {fc} (%)')].iloc[0]
            ax.errorbar(pos + off, z.estimate_fraction, yerr=[[z.estimate_fraction - z.ci_low], [z.ci_high - z.estimate_fraction]],
                        fmt=mk, ms=3.6, color=col, mfc=fill, mec=col, mew=0.9, elinewidth=0.7, capsize=1.6)
        pos += 1
    ticks.append(pos - 2); pos += 0.7
ax.axhline(0, color=C['grey'], lw=0.6)
ax.set_xticks(ticks); ax.set_xticklabels(['Caltech', 'JPL']); ax.set_ylim(-12, 78); ax.set_xlim(-0.6, pos - 0.4)
ax.set_ylabel('Capacity value captured (% of oracle)')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=c_, ms=3.6, label=l_) for _, l_, c_ in fcs] +
                  [Line2D([], [], marker='o', ls='', color=C['grey'], ms=3.6, label='Raw forecast'),
                   Line2D([], [], marker='s', ls='', color=C['grey'], mfc='white', ms=3.6, label=r'Shifted, $\gamma=0.1$')],
          frameon=False, loc='upper left', fontsize=5.8, bbox_to_anchor=(-0.02, 1.02), handletextpad=0.2, labelspacing=0.25)
ax.set_title('Capacity value of the forecast', loc='left'); tag(ax, 'f')
save(fig, 'fig4_operation')
print('figures written to', OUT)
