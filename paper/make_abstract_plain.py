"""Plain-text abstract for the submission form, from the LaTeX source with every number macro expanded.

    python make_abstract_plain.py [out=Abstract_plain.txt]

Reading the source rather than the PDF keeps hyphens at line breaks (for example label-maturation) and takes each
number from the same macros as the manuscript.
"""
import re
import sys
from pathlib import Path


def macros() -> dict:
    out = {}
    for f in ('resultsnumbers.tex', 'externalnumbers.tex', 'calibrationnumbers.tex'):
        if Path(f).exists():
            for name, body in re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', Path(f).read_text()):
                out[name] = body
    return out


def expand(text: str, m: dict) -> str:
    for _ in range(5):                                  # nested macros
        text = re.sub(r'\\(\w+)(?![A-Za-z])', lambda g: m.get(g.group(1), g.group(0)), text)
    return text


def plain(t: str) -> str:
    rules = [(r'\\ensuremath\{-\}', '\u2212'), (r'\$-\$', '\u2212'), (r'\\,\\%', ' %'), (r'\\%', '%'), (r'\\,', ' '),
             (r'---', '\u2014'), (r'--', '\u2013'), (r'~', ' '), (r'\\emph\{([^}]*)\}', r'\1'), (r'\\textit\{([^}]*)\}', r'\1'),
             (r'\\ ', ' '), (r'\{\}', ''), (r'\s+', ' ')]
    for a, b in rules:
        t = re.sub(a, b, t)
    return t.strip()


def main(out='Abstract_plain.txt'):
    src = Path('manuscript.tex').read_text()
    body = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', src, re.S).group(1)
    text = plain(expand(body, macros()))
    if '\\' in text or '{' in text:
        raise SystemExit(f'unexpanded LaTeX left in the abstract: {text}')
    Path(out).write_text(text + '\n')
    print(f'{len(text.split())} words written to {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
