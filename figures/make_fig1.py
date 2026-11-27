"""Figure 1: evaluation framework, printed at the full text width.

Canvas 1500 x 684 user units, printed at the IEEE text width (515.5 pt), so 1 unit = 0.344 pt: the smallest text
(23.3 units) prints at 8.0 pt, panel letters at 10.0 pt and panel titles at 8.5 pt, as in Figs. 2 to 4.
Panel letters are plain bold letters, as in the other figures.
Every text element is measured with the Liberation Sans metrics and checked against the space it may use.
Writes fig1_framework.svg, .pdf and .png in the current folder.
"""
from pathlib import Path
from PIL import ImageFont

W, H = 1500, 684
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"
INK, SUB, LINE = "#1F2933", "#323F4B", "#9AA5B1"
PANEL_FILL, PANEL_EDGE = "#F8FAFC", "#CBD2D9"
BLUE, SKY, ORANGE, GREEN, VERM, GREY = "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#8C959F"
MIN = 23.3                                   # smallest text size, 8.0 pt in print
TTF = {('normal', False): 'LiberationSans-Regular.ttf', ('bold', False): 'LiberationSans-Bold.ttf',
       ('normal', True): 'LiberationSans-Italic.ttf', ('bold', True): 'LiberationSans-BoldItalic.ttf'}
_fonts = {}
out = []


def width(s, size, weight='normal', italic=False):
    k = (weight, italic)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype('/usr/share/fonts/truetype/liberation/' + TTF[k], 200)
    return _fonts[k].getlength(s) * size / 200


def add(s):
    out.append(s)


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def rect(x, y, w, h, r=0, fill="none", stroke="none", sw=1.5):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" ry="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>')


def line(x1, y1, x2, y2, stroke=LINE, sw=2):
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=MIN, weight="normal", fill=INK, anchor="start", italic=False, maxw=None, raw=None, measure=None):
    """Plain text s (escaped) or raw SVG content; measure gives the plain string used to check the width."""
    assert size >= MIN, (s, size)
    w = width(measure if measure is not None else s, size, weight, italic)
    if maxw is not None and w > maxw + 0.5:
        raise ValueError(f'text too wide: {s or measure!r} {w:.0f} > {maxw:.0f}')
    st = ' font-style="italic"' if italic else ''
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{st}>{raw if raw is not None else esc(s)}</text>')
    return w


def panel(x, y, w, h, letter, title):
    rect(x, y, w, h, 12, PANEL_FILL, PANEL_EDGE, 1.6)
    head(x + 14, y + 34, letter, title, w - 28)


def head(x, y, letter, title, maxw):
    """Panel letter (bold, 10 pt) followed by the panel title (regular, 8.5 pt), as in Figs. 2 to 4."""
    lw = text(x, y, letter, 29.1, "bold")
    text(x + lw + 12, y, title, 24.7, "normal", maxw=maxw - lw - 12)


def chip(x, y, label, color, size=MIN, h=30, pad=9, maxright=None):
    w = width(label, size) + 2 * pad
    if maxright is not None and x + w > maxright + 0.5:
        raise ValueError(f'chip too wide: {label!r}')
    rect(x, y, w, h, h / 2, "#FFFFFF", color, 1.8)
    text(x + w / 2, y + h / 2 + size * 0.36, label, size, "normal", INK, "middle")
    return w


def card(x, y, w, h, title):
    rect(x, y, w, h, 9, "#FFFFFF", PANEL_EDGE, 1.3)
    text(x + 12, y + 27, title, MIN, "bold", maxw=w - 24)


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="181.9mm" height="{181.9 * H / W:.1f}mm">')
add('<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    '<rect width="7" height="7" fill="#E4E7EB"/><line x1="0" y1="0" x2="0" y2="7" stroke="#9AA5B1" stroke-width="3"/>'
    '</pattern></defs>')
rect(0, 0, W, H, 0, "#FFFFFF")

PY, PH, GAP = 8, 480, 26                     # panel row
PW = (W - 16 - 3 * GAP) / 4                  # four equal panels
XS = [8 + i * (PW + GAP) for i in range(4)]
IN = PW - 28                                 # usable width inside a panel

# ---------------------------------------------------------------- a: data and plug-in information
x = XS[0]
panel(x, PY, PW, PH, "a", "Data")
text(x + 14, PY + 76, "ACN-Data, three", MIN, "bold", SUB, maxw=IN)
text(x + 14, PY + 102, "workplace sites", MIN, "bold", SUB, maxw=IN)
text(x + 14, PY + 128, "66,727 sessions", MIN, "normal", INK, maxw=IN)
bx, by, bw, bh = x + 12, PY + 150, PW - 24, 200
rect(bx, by, bw, bh, 9, "#EAF4FB", BLUE, 1.8)
text(bx + 12, by + 30, "", MIN, "bold", BLUE, maxw=bw - 24, measure="Plug-in set Ft",
     raw='Plug-in set <tspan font-style="italic">F</tspan><tspan font-size="18.5" dy="5" font-style="italic">t</tspan>')
for i, s in enumerate(["calendar, site, station", "driver history", "station history", "driver request"]):
    add(f'<circle cx="{bx + 18}" cy="{by + 56 + i * 28}" r="4" fill="{INK}"/>')
    text(bx + 30, by + 63 + i * 28, s, MIN, maxw=bw - 42)
text(bx + 8, by + 182, "completed sessions only", MIN, "normal", SUB, italic=True, maxw=bw - 14)
ay = PY + 382
rect(x + 12, ay, PW - 24, 62, 9, "#E8F6F1", GREEN, 1.8)
add(f'<path d="M{x + 26},{ay + 31} l8,9 l15,-18" fill="none" stroke="{GREEN}" stroke-width="4"/>')
text(x + 58, ay + 27, "Leakage audit", MIN, "bold", maxw=PW - 80)
text(x + 58, ay + 51, "stops any violation", MIN, maxw=PW - 80)
# ---------------------------------------------------------------- b: protocols
x = XS[1]
panel(x, PY, PW, PH, "b", "Protocols")
# legend
lx, ly = x + 14, PY + 62
for col, lab in [(BLUE, "train"), (ORANGE, "test")]:
    rect(lx, ly, 18, 18, 3, col)
    lx += 24 + text(lx + 24, ly + 16, lab, MIN, "normal", SUB) + 14
rect(lx, ly, 18, 18, 3, "url(#hatch)", "#9AA5B1", 1)
text(lx + 24, ly + 16, "purged", MIN, "normal", SUB, maxw=x + PW - lx - 34)
CELL = (PW - 28) / 12
rows = [("Random", "BBOBBBOBBOBB", None),
        ("Chronological", "BBBBBBBHOOOO", None),
        ("Site-disjoint", "BBBBBBBBOOOO", "site"),
        ("Driver-disjoint", "BOBBOBBOBBBO", "driver")]
for r, (name, pat, note) in enumerate(rows):
    yy = PY + 108 + r * 88
    text(x + 14, yy, name, MIN, "bold", maxw=IN)
    for c, k in enumerate(pat):
        cx = x + 14 + c * CELL
        if k == 'H':
            rect(cx + 1, yy + 12, CELL - 2, 24, 3, "url(#hatch)", "#9AA5B1", 0.8)
        else:
            rect(cx + 1, yy + 12, CELL - 2, 24, 3, BLUE if k == "B" else ORANGE)
    sub_ = {"Random": "driver seen in training", "Chronological": "train before test",
            "Site-disjoint": "test site held out", "Driver-disjoint": "test driver unseen"}[name]
    text(x + 14, yy + 59, sub_, MIN, "normal", SUB, maxw=IN)
# arrow: optimistic -> deployment-like
ax_ = x + PW - 14
text(x + 14, PY + 466, "↓ closer to deployment", MIN, "normal", SUB, maxw=IN)

# ---------------------------------------------------------------- c: learners
x = XS[2]
panel(x, PY, PW, PH, "c", "Learners")
text(x + 14, PY + 76, "Main learners", MIN, "bold", SUB, maxw=IN)
cy = PY + 88
w1 = chip(x + 14, cy, "CatBoost", BLUE, maxright=x + PW - 14)
chip(x + 14, cy + 40, "FT-Transformer", BLUE, maxright=x + PW - 14)
text(x + 14, PY + 200, "Benchmark (12)", MIN, "bold", SUB, maxw=IN)
yy, xx = PY + 212, x + 14
for lab in ["LightGBM", "XGBoost", "TabPFN v2", "MLP", "Ridge", "+5 more"]:
    wch = width(lab, MIN) + 18
    if xx + wch > x + PW - 14:
        xx, yy = x + 14, yy + 40
    chip(xx, yy, lab, GREY, maxright=x + PW - 14)
    xx += wch + 8
for i, s in enumerate(["absolute-error loss", "8 tuning trials", "5 seeds"]):
    add(f'<circle cx="{x + 20}" cy="{PY + 382 + i * 30}" r="4" fill="{INK}"/>')
    text(x + 32, PY + 389 + i * 30, s, MIN, maxw=IN - 18)

# ---------------------------------------------------------------- d: analyses
x = XS[3]
panel(x, PY, PW, PH, "d", "Analyses")
cx, cw = x + 10, PW - 20
card(cx, PY + 50, cw, 146, "Gap decomposition")
# colours of the information settings, as in Figs. 3 and 4 (B, driver-temporal, appears only here)
lad = [("A", "random", 30, GREEN), ("B", "past only", 44, GREY), ("C", "unseen", 58, SKY), ("D", "cold start", 82, VERM)]
for i, (k, lab, L, col) in enumerate(lad):
    yy = PY + 92 + i * 25
    text(cx + 12, yy + 16, k, MIN, "bold")
    rect(cx + 34, yy + 1, L, 18, 3, col)
    text(cx + 42 + L, yy + 16, lab, MIN, "normal", SUB, maxw=cw - 52 - L)
card(cx, PY + 204, cw, 76, "Reliability bound")
text(cx + 12, PY + 266, "", MIN, maxw=cw - 24, measure="Pr(U > 0) ≤ 2γ",
     raw='Pr(<tspan font-style="italic">U</tspan> &gt; 0) ≤ 2<tspan font-style="italic">γ</tspan>')
card(cx, PY + 288, cw, 76, "Information-state rule")
text(cx + 12, PY + 350, "", MIN, maxw=cw - 24, measure="Pr(U > 0 | s) ≤ γ",
     raw='Pr(<tspan font-style="italic">U</tspan> &gt; 0 | <tspan font-style="italic">s</tspan>) ≤ <tspan font-style="italic">γ</tspan>')
card(cx, PY + 372, cw, 100, "Grid value")
text(cx + 12, PY + 428, "ACN-Sim capacity,", MIN, "normal", SUB, maxw=cw - 24)
text(cx + 12, PY + 454, "firm limits, feeders", MIN, "normal", SUB, maxw=cw - 24)

# flow arrows between panels
for i in range(3):
    x1, x2 = XS[i] + PW, XS[i + 1]
    ymid = PY + PH / 2
    add(f'<polygon points="{x1 + 3},{ymid - 7} {x2 - 10},{ymid - 7} {x2 - 10},{ymid - 14} {x2 - 2},{ymid} '
        f'{x2 - 10},{ymid + 14} {x2 - 10},{ymid + 7} {x1 + 3},{ymid + 7}" fill="{INK}" opacity="0.85"/>')

# ---------------------------------------------------------------- e: statistical inference
EY, EH = PY + PH + 12, 120
rect(8, EY, W - 16, EH, 12, PANEL_FILL, PANEL_EDGE, 1.6)
head(22, EY + 32, "e", "Statistical inference", W - 44)
items = [["Day-block bootstrap, B = 3000", "Driver-cluster bootstrap, B = 2000", "Holm correction within families"],
         ["Nadeau–Bengio corrected t-test", "Friedman test, Nemenyi CD", "p = 2(ν + 1)/(B + 1)"]]
cwid = (W - 16 - 24 - 2 * 12) / 3
for r, rowv in enumerate(items):
    for c, s in enumerate(rowv):
        xx = 20 + c * (cwid + 12)
        yy = EY + 44 + r * 36
        rect(xx, yy, cwid, 32, 16, "#FFFFFF", INK, 1.4)
        # math symbols in italics, as in the text
        it = lambda v: f'<tspan font-style="italic">{v}</tspan>'
        raw = (esc(s).replace('B =', it('B') + ' =').replace('(B', '(' + it('B')).replace('(ν', '(' + it('ν'))
               .replace('p =', it('p') + ' ='))
        # centred by measurement: text-anchor="middle" with tspans is not rendered reliably by cairosvg
        tw = width(s, MIN)
        text(xx + (cwid - tw) / 2, yy + 24, "", MIN, "normal", INK, "start", maxw=cwid - 16, measure=s, raw=raw)

# ---------------------------------------------------------------- external validation band
XV = EY + EH + 10
rect(8, XV, W - 16, H - XV - 8, 12, "#E8F6F1", GREEN, 1.8)
t1 = text(22, XV + 31, "External validation:", MIN, "bold", INK)
text(22 + t1 + 10, XV + 31, "City of Palo Alto public network, 2011 to 2020, 256,391 sessions, same pipeline",
     MIN, "normal", INK, maxw=W - 16 - 28 - t1 - 10)
add('</svg>')

assert XV + 40 <= H, 'external band does not fit'
Path('fig1_framework.svg').write_text('\n'.join(out), encoding='utf-8')
import cairosvg  # noqa: E402
cairosvg.svg2pdf(url='fig1_framework.svg', write_to='fig1_framework.pdf')
cairosvg.svg2png(url='fig1_framework.svg', write_to='fig1_framework.png', output_width=round(7.16 * 600))   # 600 dpi
from PIL import Image  # noqa: E402
_im = Image.open('fig1_framework.png').convert('RGBA'); _bg = Image.new('RGB', _im.size, 'white')
_bg.paste(_im, mask=_im.split()[-1]); _bg.save('fig1_framework.png', dpi=(600, 600))
print('fig1_framework.svg, .pdf and .png written; smallest text', MIN, 'units =',
      round(MIN * 7.16 * 72 / W, 2), 'pt at the full text width')
