"""Overview figure (graphical abstract): the problem, its cause and its cost, on both networks.

    python make_overview.py [BUILD_DIR]

Reads every number from resultsnumbers.tex (ACN) and externalnumbers.tex (Palo Alto) in BUILD_DIR (default: the
current directory), so the figure cannot disagree with the text. Writes overview.pdf, overview.svg and overview.png.
"""
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

B = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
M = {}
for f in ['resultsnumbers.tex', 'externalnumbers.tex']:
    for k, v in re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', (B / f).read_text()):
        M[k] = v.replace('\\ensuremath{-}', '-')
num = lambda k: float(M[k].replace(',', ''))

C = {'blue': '#0072B2', 'sky': '#56B4E9', 'verm': '#D55E00', 'grey': '#8C959F', 'ink': '#1F2933', 'sub': '#52606D',
     'light': '#F4F6F8', 'leak': '#B8C2CC', 'unseen': '#E69F00'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
NETS = [('ACN', 'ACN\nworkplace', ''), ('PA', 'Palo Alto\npublic', 'PA')]

fig = plt.figure(figsize=(13.28, 5.312))   # 2.5 : 1, the Elsevier graphical-abstract proportion (1328 x 531 px)
gs = fig.add_gridspec(1, 3, left=0.075, right=0.975, top=0.69, bottom=0.255, wspace=0.55)


def header(ax, n, title, sub):
    ax.annotate(n, xy=(-0.02, 1.0), xycoords='axes fraction', xytext=(0, 34), textcoords='offset points',
                fontsize=17, fontweight='bold', color='white', ha='center', va='center',
                bbox={'boxstyle': 'circle,pad=0.32', 'fc': C['ink'], 'ec': 'none'})
    ax.annotate(title, xy=(0.04, 1.0), xycoords='axes fraction', xytext=(4, 34), textcoords='offset points',
                fontsize=15.5, fontweight='bold', color=C['ink'], ha='left', va='center')
    ax.annotate(sub, xy=(0.04, 1.0), xycoords='axes fraction', xytext=(4, 12), textcoords='offset points',
                fontsize=11.5, color=C['sub'], ha='left', va='center')


def arrows(ax, rows, lo_lab, hi_lab, lo_col, hi_col, unit, xmax, legend_loc):
    for i, (lab, a, b, pct) in enumerate(rows):
        y = 1.15 - 1.3 * i
        ax.annotate('', xy=(b, y), xytext=(a, y), arrowprops={'arrowstyle': '-|>,head_width=0.45,head_length=0.8',
                                                              'color': C['ink'], 'lw': 2.2, 'shrinkA': 7, 'shrinkB': 7})
        ax.plot(a, y, 'o', ms=12, color=lo_col, zorder=3); ax.plot(b, y, 'o', ms=12, color=hi_col, zorder=3)
        ax.text(a, y - 0.30, f'{a:.1f}', ha='center', va='top', fontsize=11.5, color=lo_col)
        ax.text(b, y - 0.30, f'{b:.1f}', ha='center', va='top', fontsize=11.5, color=hi_col)
        ax.text((a + b) / 2, y + 0.24, pct, ha='center', va='bottom', fontsize=16, fontweight='bold', color=C['ink'])
    ax.set_yticks([1.15, -0.15]); ax.set_yticklabels([r[0] for r in rows], fontsize=12, color=C['ink'])
    ax.set_ylim(-0.75, 1.95); ax.set_xlim(0, xmax); ax.set_xlabel(unit, fontsize=11.5, color=C['sub'])
    ax.tick_params(axis='x', labelsize=10.5, colors=C['sub']); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False); ax.grid(axis='x', color='#E3E7EB', lw=0.6); ax.set_axisbelow(True)
    ax.plot([], [], 'o', ms=9, color=lo_col, label=lo_lab); ax.plot([], [], 'o', ms=9, color=hi_col, label=hi_lab)
    ax.legend(loc=legend_loc, ncol=1, frameon=False, fontsize=10.5, handletextpad=0.2, borderaxespad=0.1)


pct = lambda a, b: f'{100 * (b / a - 1):+.0f}%'.replace('-', '\u2212')

# 1 the problem: random splits overstate accuracy
ax = fig.add_subplot(gs[0, 0])
rows = [(lab, num(p + 'EnRand'), num(p + 'EnUser'), pct(num(p + 'EnRand'), num(p + 'EnUser'))) for _, lab, p in NETS]
arrows(ax, rows, 'Random split', 'Unseen driver', C['grey'], C['verm'], 'Energy error, MAE (kWh)', 9, 'upper left')
header(ax, '1', 'Random splits overstate accuracy', 'Error for drivers the model has not seen')

# 2 the cause: the gap is missing driver history
ax = fig.add_subplot(gs[0, 1])
for i, (_, lab, p) in enumerate(NETS):
    y = 1 - i
    tot = math.log1p(num(p + 'TotEnCb') / 100)
    parts = [(math.log1p(num(p + 'LeakEnCb') / 100) / tot, C['leak']), (math.log1p(num(p + 'UnseenEnCb') / 100) / tot, C['unseen']),
             (math.log1p(num(p + 'HistEnCb') / 100) / tot, C['blue'])]
    left = 0
    for w, col in parts:
        ax.barh(y, 100 * w, left=left, height=0.52, color=col, edgecolor='white', linewidth=1.2)
        left += 100 * w
    lo, hi = num(p + 'ShareEnMin'), num(p + 'ShareEnMax')
    ax.text(100 - 100 * parts[2][0] / 2, y, f'{lo:.0f}\u2013{hi:.0f}%', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
ax.set_yticks([1, 0]); ax.set_yticklabels([lab for _, lab, _ in NETS], fontsize=12, color=C['ink'])
ax.set_ylim(-0.75, 1.75); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xticklabels(['0', '50', '100%'])
ax.tick_params(axis='x', labelsize=10.5, colors=C['sub']); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_xlabel('Share of the new-driver energy gap', fontsize=11.5, color=C['sub'])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C['leak'], label='Leakage'), Patch(color=C['unseen'], label='Unseen driver'),
                   Patch(color=C['blue'], label="Missing own history")], loc='upper center', ncol=3,
          frameon=False, fontsize=10.5, handlelength=1.0, handletextpad=0.35, columnspacing=0.9, borderaxespad=0.1)
one = max(num('OneSessionEnMax'), num('PAOneSessionEnMax'))
header(ax, '2', 'The gap is missing driver history', f'One completed session closes it (within {math.ceil(one):.0f}%)')

# 3 the cost: a cold start removes much of the grid flexibility (values from the macros)
ax = fig.add_subplot(gs[0, 2])
rows = [(lab, num(p + 'FlexC'), num(p + 'FlexD'), pct(num(p + 'FlexC'), num(p + 'FlexD'))) for _, lab, p in NETS]
arrows(ax, rows, 'Own history known', 'Cold start', C['blue'], C['verm'], 'Flexibility kept (% of oracle)', 45, 'upper left')
header(ax, '3', 'A cold start costs flexibility', 'At 2% undelivered energy')

# bottom ribbon: data
fig.patches.append(FancyBboxPatch((0.012, 0.02), 0.976, 0.08, boxstyle='round,pad=0.004,rounding_size=0.012',
                                  transform=fig.transFigure, fc=C['light'], ec='none', zorder=0))
fig.text(0.5, 0.06, f"Two independent networks, same protocols and learners:   ACN workplace sites, 66,727 sessions   \u00b7   "
         f"City of Palo Alto public network, {M['PANSessions']} sessions", ha='center', va='center', fontsize=12, color=C['ink'])
fig.text(0.012, 0.965, 'Forecasts made at plug-in for EV smart charging: what survives deployment?', fontsize=18,
         fontweight='bold', color=C['ink'], va='top')

for ext, kw in [('pdf', {}), ('svg', {}), ('png', {'dpi': 400})]:
    fig.savefig(B / f'overview.{ext}', **kw)
print('overview.pdf, .svg and .png written')
