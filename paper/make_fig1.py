"""Figure 1: evaluation framework. Generates fig1_framework.svg (and PDF/PNG via cairosvg).

Canvas 1800 x 1060 user units printed at 190 mm x 111.9 mm (1 unit = 0.1056 mm);
the smallest text (20 units) prints at 6.0 pt.
"""
from pathlib import Path

W, H = 1800, 1134
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"
INK = "#1F2933"
SUB = "#52606D"
LINE = "#9AA5B1"
PANEL_FILL = "#F8FAFC"
PANEL_EDGE = "#CBD2D9"
BLUE, SKY, ORANGE, GREEN, VERM, PURPLE, YELLOW, GREY = (
    "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#8C959F")

out = []


def add(s):
    out.append(s)


def rect(x, y, w, h, r=0, fill="none", stroke="none", sw=1.5, extra=""):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def text(x, y, s, size=22, weight="normal", fill=INK, anchor="start", italic=False, extra=""):
    size = max(size, 20)
    st = ' font-style="italic"' if italic else ''
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{st} {extra}>{s}</text>')


def line(x1, y1, x2, y2, stroke=LINE, sw=2, extra=""):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def sub(base, s, size):
    """italic symbol with subscript, as tspans."""
    return (f'<tspan font-style="italic">{base}</tspan>'
            f'<tspan font-size="{int(size * 0.72)}" dy="{int(size * 0.28)}">{s}</tspan>'
            f'<tspan dy="-{int(size * 0.28)}"> </tspan>')


def arrow_down(x, y1, y2, color=LINE):
    line(x, y1, x, y2 - 9, color, 2.5)
    add(f'<polygon points="{x - 8},{y2 - 11} {x + 8},{y2 - 11} {x},{y2}" fill="{color}"/>')


def panel(x, y, w, h, letter, title):
    rect(x, y, w, h, 14, PANEL_FILL, PANEL_EDGE, 1.6)
    add(f'<circle cx="{x + 30}" cy="{y + 34}" r="17" fill="{INK}"/>')
    text(x + 30, y + 42, letter, 22, "bold", "#FFFFFF", "middle")
    text(x + 58, y + 43, title, 27, "bold")


def chip(x, y, label, color, size=21, h=36, pad=14, w=None):
    w = w or int(len(label) * size * 0.53 + 2 * pad)
    rect(x, y, w, h, h / 2, "#FFFFFF", color, 2)
    text(x + w / 2, y + h / 2 + size * 0.36, label, size, "normal", INK, "middle")
    return w


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="190mm" height="{190 * H / W:.1f}mm">')
add('<defs>'
    f'<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    f'<rect width="7" height="7" fill="#E4E7EB"/><line x1="0" y1="0" x2="0" y2="7" stroke="#9AA5B1" stroke-width="3"/></pattern>'
    f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
    '</defs>')
rect(0, 0, W, H, 0, "#FFFFFF")

# ---------------------------------------------------------------- panel a
AX, AW = 20, 390
panel(AX, 20, AW, 800, "a", "Data and information")
text(AX + 24, 108, "ACN-Data, three workplace sites", 20, "bold", SUB)
for i, (site, n) in enumerate([("Caltech", "31,424"), ("JPL", "33,638"), ("Office001", "1,683")]):
    y = 122 + i * 46
    rect(AX + 24, y, AW - 48, 38, 8, "#FFFFFF", PANEL_EDGE, 1.4)
    text(AX + 40, y + 26, site, 21)
    text(AX + AW - 40, y + 26, f"{n} sessions", 21, "normal", SUB, "end")
arrow_down(AX + AW / 2, 262, 292)
rect(AX + 24, 294, AW - 48, 70, 10, "#FFFFFF", INK, 1.8)
text(AX + AW / 2, 324, "66,727 sessions after cleaning", 22, "bold", INK, "middle")
text(AX + AW / 2, 352, "2018 to 2021; 18 sessions removed", 19, "normal", SUB, "middle")
arrow_down(AX + AW / 2, 364, 394)

# plug-in information set
IB_TOP = 394
it = "<tspan font-style='italic'>t</tspan>"
tau_j = "<tspan font-style='italic'>τ</tspan><tspan font-size='16' dy='5' font-style='italic'>j</tspan><tspan dy='-5'> </tspan>"
items = [("Calendar, cyclic encodings", None),
         ("Driver history from completed", f"sessions only ({tau_j}&lt; {it})"),
         ("Station history from completed", "sessions only"),
         ("Driver request: last revision", f"made at or before {it}")]
yy = IB_TOP + 66
body = []
for head, tail in items:
    body.append(("sq", yy))
    body.append(("t", yy, head, INK))
    yy += 25
    if tail:
        body.append(("t", yy, tail, SUB))
        yy += 25
    yy += 5
IB_H = yy - IB_TOP + 2
rect(AX + 24, IB_TOP, AW - 48, IB_H, 10, "#EAF4FB", BLUE, 2)
add(f'<text x="{AX + 40}" y="{IB_TOP + 32}" font-family="{FONT}" font-size="22" font-weight="bold" fill="{INK}">'
    f'Plug-in information set <tspan font-style="italic">F</tspan><tspan font-size="20" dy="5" font-style="italic">t</tspan></text>')
for b in body:
    if b[0] == "sq":
        rect(AX + 42, b[1] - 13, 11, 11, 2, BLUE)
    else:
        text(AX + 62, b[1] - 2, b[2], 19.5, "normal", b[3])

# timeline: forecast at plug-in, labels revealed at departure
TL = IB_TOP + IB_H + 24
text(AX + 24, TL, "Timing of information", 20, "bold", SUB)
ty = TL + 42
x_p, x_c, x_d, x_end = AX + 70, AX + 200, AX + 322, AX + AW - 24
line(AX + 40, ty, x_end, ty, INK, 2.2, 'marker-end="url(#arr)"')
for xx, col in [(x_p, BLUE), (x_c, GREY), (x_d, ORANGE)]:
    add(f'<circle cx="{xx}" cy="{ty}" r="7" fill="{col}"/>')
text(x_p, ty - 15, "forecast", 18, "bold", BLUE, "middle")
text(x_c, ty - 15, "diagnostic", 20, "normal", GREY, "middle")
add(f'<text x="{x_d - 52}" y="{ty - 15}" font-family="{FONT}" font-size="20" font-weight="bold" fill="{ORANGE}">'
    f'<tspan font-style="italic">D</tspan>, <tspan font-style="italic">E</tspan> known</text>')
text(x_p, ty + 26, "plug-in", 20, "normal", SUB, "middle")
text(x_c, ty + 26, "charging ends", 20, "normal", SUB, "middle")
text(x_d, ty + 26, "departure", 20, "normal", SUB, "middle")

# leakage audit strip
AU = 756
rect(AX + 24, AU, AW - 48, 52, 10, "#E8F6F1", GREEN, 2)
add(f'<path d="M{AX + 42},{AU + 26} l9,10 l17,-21" fill="none" stroke="{GREEN}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
text(AX + 80, AU + 23, "Automated leakage audit", 19.5, "bold")
text(AX + 80, AU + 44, "the run stops on any violation", 20, "normal", SUB)

# ---------------------------------------------------------------- panel b
BX, BW = 450, 440
panel(BX, 20, BW, 800, "b", "Evaluation protocols")
cols = {"T": BLUE, "C": SKY, "Q": ORANGE}
lx = BX + 24
for lab, key in [("Train/validation", "T"), ("Calibration", "C"), ("Test", "Q")]:
    rect(lx, 86, 17, 17, 3, cols[key])
    text(lx + 23, 101, lab, 20, "normal", SUB)
    lx += 34 + len(lab) * 10.2
rect(BX + 24, 113, 17, 17, 3, "url(#hatch)", "#9AA5B1", 1)
text(BX + 47, 128, "Purged (immature label)", 20, "normal", SUB)
rect(BX + 290, 113, 17, 17, 3, "#FFFFFF", "#9AA5B1", 1.2)
text(BX + 313, 128, "Excluded", 20, "normal", SUB)

CELL, NC = 23, 16
X0 = BX + 24


def bar(y, pattern, groups=None, divider=None, labels=None):
    for i, c in enumerate(pattern):
        x = X0 + i * CELL
        if c == "G":
            rect(x + 1, y, CELL - 2, 26, 3, "url(#hatch)", "#9AA5B1", 0.8)
        elif c == "X":
            rect(x + 1, y, CELL - 2, 26, 3, "#FFFFFF", "#9AA5B1", 1)
        else:
            rect(x + 1, y, CELL - 2, 26, 3, cols[c])
    if groups:
        for g in groups:
            line(X0 + g * CELL, y - 4, X0 + g * CELL, y + 30, INK, 2.2)
    if divider is not None:
        xd = X0 + divider * CELL
        line(xd, y - 12, xd, y + 36, VERM, 2.4, 'stroke-dasharray="5,4"')
    if labels:
        for (a, b), s in labels:
            text(X0 + (a + b) / 2 * CELL, y + 51, s, 20, "normal", SUB, "middle")


rows = [
    ("Random", "uniform permutation", "TTQTCTTQTTCTTQTT", {}),
    ("Chronological", "maturation purge, 4 h embargo", "TTTTTTTTTGCCGQQQ", {}),
    ("Chronological, pre-2020", "rolling origin, before 1 Mar 2020", "TTTTTTTTGQQQXXXX", {"divider": 12, "sens": True}),
    ("User-disjoint", "no shared drivers; anonymous excluded", "TTTTTTTTCCQQQQXX",
     {"groups": [4, 8, 10, 14], "labels": [((0, 8), "drivers in training"), ((14, 16), "anon.")]}),
    ("Site-disjoint, online", "own history accrues at the target", "TTTTTTTTTTTCGQQQ",
     {"groups": [6, 11], "labels": [((0, 6), "source"), ((6, 11), "source"), ((11, 16), "target")]}),
    ("Site-disjoint, strict", "no target-site information", "TTTTTTTTTTTQQQQQ",
     {"groups": [6, 11], "sens": True}),
]
ry = 160
for name, desc, pat, opt in rows:
    text(X0, ry, name, 21, "bold")
    if opt.get("sens"):
        w = len(name) * 11.9
        rect(X0 + w + 10, ry - 20, 110, 26, 13, "#FFFFFF", PURPLE, 1.8)
        text(X0 + w + 65, ry - 1, "sensitivity", 20, "normal", PURPLE, "middle")
    text(X0, ry + 25, desc, 20, "normal", SUB)
    bar(ry + 38, pat, opt.get("groups"), opt.get("divider"), opt.get("labels"))
    ry += 118 if opt.get("labels") else 94

add(f'<line x1="{BX + BW - 16}" y1="150" x2="{BX + BW - 16}" y2="790" stroke="{SUB}" stroke-width="2.2" marker-end="url(#arr)"/>')
add(f'<text transform="translate({BX + BW - 24},470) rotate(-90)" font-family="{FONT}" font-size="20" '
    f'fill="{SUB}" text-anchor="middle">closer to deployment</text>')

# ---------------------------------------------------------------- panel c
CX, CW = 930, 390
panel(CX, 20, CW, 800, "c", "Learners")
fam = [("Reference forecasts", GREY, [["Historical mean", "Driver history"], ["Driver request"]]),
       ("Linear", PURPLE, [["Ridge regression"]]),
       ("Bagged trees", GREEN, [["Random forest", "Extra trees"]]),
       ("Gradient boosting", BLUE, [["HistGB", "LightGBM"], ["XGBoost", "CatBoost"]]),
       ("Deep tabular", VERM, [["Residual MLP", "FT-Transformer"]]),
       ("Foundation model", ORANGE, [["TabPFN v2"]])]
y = 92
for fname, col, chips in fam:
    rect(CX + 24, y + 2, 6, 20, 2, col)
    text(CX + 38, y + 19, fname, 19.5, "bold", SUB)
    y += 30
    for row in chips:
        x = CX + 24
        for c in row:
            x += chip(x, y, c, col, 20, 33, 13) + 10
        y += 40
    y += 4
rect(CX + 24, 640, CW - 48, 164, 10, "#FFFFFF", PANEL_EDGE, 1.4)
notes = ["Absolute-error objective", "Early stopping on validation", "Equal tuning budget: 8 trials",
         "5 seeds; 5 × 5 repeated CV", "Cold-start FT: missing token"]
for i, n in enumerate(notes):
    add(f'<circle cx="{CX + 44}" cy="{666 + i * 29}" r="4.5" fill="{INK}"/>')
    text(CX + 58, 672 + i * 29, n, 19)

# ---------------------------------------------------------------- panel d
DX, DW = 1360, 420
panel(DX, 20, DW, 800, "d", "Analyses")


def card(y, h, title):
    rect(DX + 20, y, DW - 40, h, 10, "#FFFFFF", PANEL_EDGE, 1.4)
    text(DX + 36, y + 30, title, 20.5, "bold")


# d1 uncertainty
card(84, 170, "Uncertainty and occupancy")
cx0, cy0 = DX + 46, 182
line(cx0, cy0 - 34, cx0, cy0 + 34, BLUE, 3)
line(cx0 - 9, cy0 - 34, cx0 + 9, cy0 - 34, BLUE, 3)
line(cx0 - 9, cy0 + 34, cx0 + 9, cy0 + 34, BLUE, 3)
add(f'<circle cx="{cx0}" cy="{cy0 + 6}" r="6" fill="{BLUE}"/>')
add(f'<circle cx="{cx0 + 15}" cy="{cy0 - 16}" r="4.5" fill="{ORANGE}"/>')
for i, s in enumerate(["Split conformal, CQR, Mondrian", "Coverage, width, interval score",
                       "Target-site recalibration", "Congested-query decisions"]):
    text(DX + 92, 146 + i * 27, s, 20, "normal", INK if i < 3 else SUB)

# d2 cold-start decomposition (schematic ladder; lengths are illustrative)
card(270, 180, "Cold-start decomposition")
ladder = [("A", "random split", 64, GREY), ("B", "past only", 92, SKY),
          ("C", "unseen + history", 120, BLUE), ("D", "cold start", 160, VERM)]
ly = 312
for i, (k, lab, L, col) in enumerate(ladder):
    y = ly + i * 24
    text(DX + 38, y + 16, k, 20, "bold", INK)
    rect(DX + 60, y + 1, L, 18, 3, col)
    text(DX + 68 + L, y + 16, lab, 20, "normal", SUB)
add(f'<text x="{DX + 36}" y="{ly + 128}" font-family="{FONT}" font-size="20" fill="{INK}">'
    f'MAE D/A<tspan> = </tspan><tspan font-style="italic">r</tspan><tspan font-size="20" dy="5">leak</tspan>'
    f'<tspan dy="-5"> · </tspan><tspan font-style="italic">r</tspan><tspan font-size="20" dy="5">unseen</tspan>'
    f'<tspan dy="-5"> · </tspan><tspan font-style="italic">r</tspan><tspan font-size="20" dy="5">history</tspan></text>')

# d3 information availability
card(466, 160, "Information availability")
nx, ny = DX + 30, 510
rect(nx, ny, 360, 100, 12, "#FDF3E7", ORANGE, 1.8)
rect(nx + 8, ny + 8, 244, 84, 10, "#FBE9DD", VERM, 1.8)
rect(nx + 16, ny + 16, 112, 68, 8, "#EAF4FB", BLUE, 2)
text(nx + 72, ny + 45, "plug-in", 20, "bold", BLUE, "middle")
text(nx + 72, ny + 70, "forecasts", 20, "normal", SUB, "middle")
text(nx + 190, ny + 45, "+ charging", 20, "normal", VERM, "middle")
text(nx + 190, ny + 70, "duration", 20, "normal", VERM, "middle")
text(nx + 306, ny + 45, "+ session", 20, "normal", ORANGE, "middle")
text(nx + 306, ny + 70, "duration", 20, "normal", ORANGE, "middle")

# d4 smart-charging impact
card(642, 158, "Smart-charging impact")
sy = 720
x_plug, x_s, x_full, x_dep = DX + 46, DX + 180, DX + 310, DX + 338
rect(x_plug, sy - 16, x_s - x_plug, 32, 3, "url(#hatch)", "#9AA5B1", 1)
rect(x_s, sy - 16, x_full - x_s, 32, 3, BLUE)
text((x_plug + x_s) / 2, sy + 7, "deferral Φ", 20, "bold", INK, "middle")
text((x_s + x_full) / 2, sy + 7, "charge at P", 20, "bold", "#FFFFFF", "middle")
line(x_plug, sy - 26, x_plug, sy + 26, INK, 2)
line(x_dep, sy - 26, x_dep, sy + 26, ORANGE, 2.5)
text(x_plug, sy + 44, "plug-in", 15.5, "normal", SUB, "middle")
text(x_dep, sy + 44, "departure", 15.5, "normal", ORANGE, "middle")
add(f'<text x="{DX + 36}" y="{sy + 70}" font-family="{FONT}" font-size="20" fill="{INK}">'
    f'Undelivered energy <tspan font-style="italic">U</tspan> vs flexibility Φ</text>')

# ---------------------------------------------------------------- flow arrows between panels
for x1, x2 in [(AX + AW, BX), (BX + BW, CX), (CX + CW, DX)]:
    add(f'<polygon points="{x1 + 6},404 {x2 - 14},404 {x2 - 14},392 {x2 - 3},420 {x2 - 14},448 {x2 - 14},436 {x1 + 6},436" '
        f'fill="{INK}" opacity="0.85"/>')

# ---------------------------------------------------------------- panel e
EY = 846
rect(20, EY, W - 40, 194, 14, PANEL_FILL, PANEL_EDGE, 1.6)
add(f'<circle cx="50" cy="{EY + 34}" r="17" fill="{INK}"/>')
text(50, EY + 42, "e", 22, "bold", "#FFFFFF", "middle")
text(78, EY + 43, "Statistical inference", 27, "bold")
inf = [["Day-block bootstrap, B = 3000", "Driver-cluster bootstrap, B = 2000",
        "Finite-sample p = 2(ν + 1)/(B + 1)", "Holm correction within families"],
       ["Nadeau–Bengio corrected t-test", "Friedman test, Nemenyi CD",
        "Out-of-sample R² and MAE skill", "Pre-specified seeds; no test-set tuning"]]
cw = (W - 40 - 48 - 3 * 16) / 4
for r, rowv in enumerate(inf):
    for c, s in enumerate(rowv):
        x = 44 + c * (cw + 16)
        y = EY + 70 + r * 58
        rect(x, y, cw, 44, 22, "#FFFFFF", INK if c != 3 else SUB, 1.6)
        text(x + cw / 2, y + 29, s, 19.5, "normal", INK, "middle")


# ---------------------------------------------------------------- external validation band
XV = EY + 194 + 16
rect(20, XV, W - 40, 62, 14, "#E8F6F1", GREEN, 2)
text(44, XV + 39, "External validation", 23, "bold", INK)
text(300, XV + 39, "City of Palo Alto public network, 2011 to 2020: 256,391 sessions after cleaning, four locations", 20.5, "normal", INK)
text(W - 44, XV + 39, "same protocols, learners, features, seeds and audit", 19, "normal", SUB, "end")
add('</svg>')
Path('fig1_framework.svg').write_text('\n'.join(out), encoding='utf-8')
try:
    import cairosvg
    cairosvg.svg2pdf(url='fig1_framework.svg', write_to='fig1_framework.pdf')
    cairosvg.svg2png(url='fig1_framework.svg', write_to='fig1_framework.png', output_width=7480, dpi=1000)
    print('fig1_framework.svg, .pdf and .png written')
except ImportError:
    print('fig1_framework.svg written (install cairosvg for PDF and PNG)')
