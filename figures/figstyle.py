"""Shared style of the paper figures (Figs. 2 to 4 and the Supplementary Figures).

Every figure is printed unscaled at its final width: the IEEE two-column width (7.16 in) for the main
figures and the A4 text width of the Supplementary Information (166 mm). All text is at least 8 pt, panel
letters are 10 pt bold and panel titles 8.5 pt, on one line at the top left of each panel, as in Fig. 1.
Colours follow the Okabe-Ito palette with one meaning each: learners (CatBoost, FT-Transformer) and
information settings (driver seen, unseen driver with own history, cold start, driver request).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MM = 1 / 25.4
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
                     'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
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


TEXTWIDTH = 7.16                  # IEEE two-column width in inches
LETTER, TITLE = 10, 8.5           # panel letter and panel title (pt), as in Fig. 1
MS, LW = 3.2, 1.2                 # marker size and line width shared by every panel
ERR = {'elinewidth': 0.6, 'capsize': 1.5, 'capthick': 0.6}
# one meaning per colour across the figures: learners (Figs. 2 and 3) and information settings (Figs. 3 and 4)
MODEL = {'catboost': ('CatBoost', C['blue']), 'ft': ('FT-Transformer', C['orange'])}
SETTING = {'seen': ('Driver seen', C['green'], 'o'), 'history': ('Unseen driver, own history', C['sky'], '^'),
           'cold': ('Unseen driver, cold start', C['verm'], 's'), 'request': ('Driver request', C['purple'], 'D'),
           'oracle': ('Perfect information', C['black'], 'o'), 'fcfs': ('No forecast (FCFS)', C['grey'], 'v')}
_HEADS = []


def head(ax, letter, title):
    """Panel letter and title on one line, starting at the left edge of the panel (axis labels included), as in Fig. 1."""
    _HEADS.append((ax, letter, title))


def _place_heads(fig):
    for t in getattr(fig, '_head_texts', []):
        t.remove()
    fig._head_texts = []
    r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height; pt = fig.dpi / 72
    for ax, letter, title in [h for h in _HEADS if h[0].figure is fig]:
        x0 = ax.get_tightbbox(r).x0
        y = ax.get_window_extent(r).y1 + 6 * pt
        t1 = fig.text(x0 / W, y / H, letter, fontsize=LETTER, fontweight='bold', va='baseline', ha='left')
        x1 = t1.get_window_extent(r).x1 + 4 * pt
        t2 = fig.text(x1 / W, y / H, title, fontsize=TITLE, va='baseline', ha='left')
        fig._head_texts += [t1, t2]


def save(fig, path, width=TEXTWIDTH):
    """Write the figure so that its tight bounding box is exactly `width` inches wide (printed unscaled)."""
    for _ in range(8):
        _place_heads(fig)
        bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(2 / 72)
        if abs(bb.width - width) < 0.002:
            break
        fig.set_size_inches(fig.get_figwidth() + width - bb.width, fig.get_figheight())
    _place_heads(fig)
    for ext, kw in [('pdf', {}), ('svg', {}), ('png', {'dpi': 1000})]:
        fig.savefig(f'{path}.{ext}', bbox_inches='tight', pad_inches=2 / 72, **kw)
    plt.close(fig)


def setting_line(ax, x, y, key, **kw):
    lab, col, mk = SETTING[key]
    ls = '--' if key == 'fcfs' else '-'
    return ax.plot(x, y, ls, marker=mk, color=col, ms=MS, lw=LW, label=lab, **kw)


SI_WIDTH = 472.32 / 72            # Supplementary text width (A4, 2.2 cm margins: 472.3 pt)


def pct_label(v):
    return f'{v:+.0f}%'.replace('-', '\u2212')


def setting_handles(keys):
    return [Line2D([], [], ls='--' if k == 'fcfs' else '-', marker=SETTING[k][2], color=SETTING[k][1], ms=MS, lw=LW,
                   label=SETTING[k][0]) for k in keys]
