"""Compare regenerated paper numbers and tables with the published ones.

    python scripts/check_reproduction.py paper/build paper/expected

Every numeric macro (\\newcommand{\\Name}{value}) is compared with its published value. GPU training is not
bit-for-bit deterministic, so differences are reported rather than hidden: a value counts as reproduced when it
is identical, or when it differs by at most one unit in the last reported digit. Tables are compared line by line.
The exit status is non-zero if any macro differs by more than that tolerance or is missing, or if any table is missing or differs.
"""
import re
import sys
from pathlib import Path

build, expected = Path(sys.argv[1]), Path(sys.argv[2])
MACRO = re.compile(r'\\newcommand\{\\(\w+)\}\{(.*)\}\s*$')


def macros(path):
    out = {}
    for line in path.read_text().splitlines():
        m = MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).replace('\\ensuremath{-}', '-')
    return out


def number(v):
    try:
        return float(v.replace(',', ''))
    except ValueError:
        return None


fail = 0
for name in ['resultsnumbers.tex', 'externalnumbers.tex']:
    exp, got = macros(expected / name), macros(build / name)
    same = close = 0
    for k, ve in exp.items():
        vg = got.get(k)
        if vg is None:
            print(f'  MISSING  {name}: \\{k}'); fail += 1; continue
        if vg == ve:
            same += 1; continue
        ne, ng = number(ve), number(vg)
        if ne is not None and ng is not None:
            decimals = len(ve.split('.')[1]) if '.' in ve else 0
            if abs(ne - ng) <= 10 ** -decimals + 1e-12:
                close += 1; print(f'  last digit  \\{k}: published {ve}, reproduced {vg}'); continue
        print(f'  DIFFERS  \\{k}: published {ve}, reproduced {vg}'); fail += 1
    print(f'{name}: {same} identical, {close} within one unit of the last digit, {len(exp) - same - close} other')
for name in sorted(p.name for p in expected.glob('*.tex') if p.name not in ('resultsnumbers.tex', 'externalnumbers.tex')):
    if not (build / name).exists():
        print(f'  MISSING  {name}: not generated'); fail += 1; continue
    a, b = (expected / name).read_text().splitlines(), (build / name).read_text().splitlines()
    diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    print(f'{name}: {"identical" if diff == 0 else f"{diff} differing lines"}')
    fail += diff > 0
print('REPRODUCED' if fail == 0 else f'NOT REPRODUCED: {fail} items missing or beyond the tolerance')
sys.exit(1 if fail else 0)
