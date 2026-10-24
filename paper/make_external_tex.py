"""Palo Alto external-validation numbers for the manuscript.

    python make_external_tex.py data/paloalto_FULL_REPORT.xlsx

Writes externalnumbers.tex (macros prefixed PA) and externaltable.tex (ACN against Palo Alto).
Every quantity uses the same definition as the corresponding ACN macro in make_results_tex.py.
"""
import math, re, sys
from pathlib import Path
import pandas as pd

X = sys.argv[1] if len(sys.argv) > 1 else 'data/paloalto_FULL_REPORT.xlsx'
_SHEETS = pd.ExcelFile(X).sheet_names
def S(name):
    hits = [s for s in _SHEETS if s == name or name.startswith(s) or s.startswith(name)]
    assert len(hits) == 1, (name, hits)
    return pd.read_excel(X, sheet_name=hits[0])
macros = {}
def m(name, value, fmt='{:.2f}'):
    assert name.isalpha(), name
    v = fmt.format(value) if not isinstance(value, str) else value
    # a typeset minus sign rather than a hyphen, valid in text and in math
    macros[name] = '\\ensuremath{-}' + v[1:] if v.startswith('-') else v
pct = lambda new, old: 100 * (new / old - 1)

flow = S('Data flow (cleaning)').set_index('stage').sessions
m('PANRaw', int(flow['raw_sessions']), '{:,d}'); m('PANSessions', int(flow.iloc[-1]), '{:,d}')

acc = S('RQ1 Accuracy by protocol')
prot = acc.groupby(['target', 'protocol', 'model'])[['mae_mean']].mean().reset_index()
get = lambda t, p, mod: float(prot.query('target==@t and protocol==@p and model==@mod').mae_mean.iloc[0])
names = {'random': 'Rand', 'chronological': 'Chrono', 'site_disjoint': 'Site', 'user_disjoint': 'User'}
for t, T in [('energy', 'En'), ('duration', 'Du')]:
    for p, P in names.items():
        m(f'PA{T}{P}', get(t, p, 'catboost'))
        if p != 'random':
            m(f'PAGap{T}{P}', pct(get(t, p, 'catboost'), get(t, 'random', 'catboost')), '{:.0f}')
seed = S('RQ1 Accuracy per seed'); cb = seed[seed.model == 'catboost']
ps = cb.groupby(['target', 'protocol', 'seed']).mae.mean().unstack('protocol')
g = 100 * (ps.loc['energy', 'user_disjoint'] / ps.loc['energy', 'random'] - 1)
m('PAGapEnUserMin', g.min(), '{:.0f}'); m('PAGapEnUserMax', g.max(), '{:.0f}')
cand = prot[~prot.model.isin(['driver_estimate', 'catboost_cqr'])]
best = cand.loc[cand.groupby(['target', 'protocol']).mae_mean.idxmin()]
rb = best[best.protocol == 'random'].set_index('target').model
m('PABestChanges', sum(1 for t, p, mod in best[['target', 'protocol', 'model']].values if p != 'random' and mod != rb[t]), '{:d}')
rk = S('RQ1 Model ranking stability')
m('PATauEnMin', rk[rk.target == 'energy'].kendall_tau.min()); m('PATauEnMax', rk[rk.target == 'energy'].kendall_tau.max())
# unseen location: learned models against the driver-history baseline (energy), and the historical-mean gain for unseen drivers
ue = {mod: get('energy', 'user_disjoint', mod) for mod in ['catboost', 'torch_ft', 'ridge', 'historical_mean']}
gains = [100 * (1 - ue[mod] / ue['historical_mean']) for mod in ['catboost', 'torch_ft', 'ridge']]
m('PAUserGainMin', min(gains), '{:.0f}'); m('PAUserGainMax', max(gains), '{:.0f}')
m('PARandGainEn', 100 * (1 - get('energy', 'random', 'catboost') / get('energy', 'random', 'historical_mean')), '{:.0f}')
site_best = cand[(cand.target == 'energy') & (cand.protocol == 'site_disjoint')].set_index('model').mae_mean
m('PASiteBestModel', {'user_history': 'the driver-history baseline', 'torch_ft': 'the FT-Transformer', 'catboost': 'CatBoost',
                      'ridge': 'ridge regression'}.get(site_best.idxmin(), site_best.idxmin()))
m('PASiteUH', site_best['user_history']); m('PASiteCb', site_best['catboost'])

dc = S('Cold-start decomposition')
dget = lambda t, l, comp, col='effect_pct': float(dc.query('target==@t and learner==@l and component==@comp')[col].iloc[0])
share = {}
for t, T in [('kwh_delivered', 'En'), ('duration_hours', 'Du')]:
    for l, L in [('catboost', 'Cb'), ('ft', 'Ft')]:
        tot, hist = dget(t, l, 'total gap'), dget(t, l, 'value of own history')
        share[(T, L)] = 100 * math.log1p(hist / 100) / math.log1p(tot / 100)
        for comp, C in [('total gap', 'Tot'), ('value of own history', 'Hist'), ('future-session leakage', 'Leak'), ('unseen-driver cost', 'Unseen')]:
            m(f'PA{C}{T}{L}', dget(t, l, comp), '{:.1f}')
            m(f'PA{C}{T}{L}Lo', dget(t, l, comp, 'ci_low'), '{:.1f}'); m(f'PA{C}{T}{L}Hi', dget(t, l, comp, 'ci_high'), '{:.1f}')
for T in ['En', 'Du']:
    m(f'PAShare{T}Min', min(share[(T, 'Cb')], share[(T, 'Ft')]), '{:.0f}'); m(f'PAShare{T}Max', max(share[(T, 'Cb')], share[(T, 'Ft')]), '{:.0f}')
m('PALeakMax', max(dget(t, l, 'future-session leakage') for t in ['kwh_delivered', 'duration_hours'] for l in ['catboost', 'ft']), '{:.1f}')
m('PANDrivers', int(dc.n_drivers.iloc[0]), '{:,d}'); m('PANRows', int(dc.n_rows.iloc[0]), '{:,d}')
m('PAPminHolm', float(dc.p_holm.max()), '{:.3f}'); m('PANBootFloor', float(dc.p_value.min()), '{:.4f}')

wu = S('Cold-start warm-up curve')
_k0 = wu[wu.k_bucket.astype(str) == '0']
m('PAFirstVisitMax', max((100 * (_k0.mae_warm / _k0.mae_random - 1)).abs().max(), (100 * (_k0.mae_cold / _k0.mae_random - 1)).abs().max()), '{:.1f}')
m('PAOneSessionEnMax', wu[(wu.k_bucket.astype(str) == '1') & (wu.target == 'kwh_delivered')].warm_vs_random_pct.max(), '{:.1f}')
w20 = wu[wu.k_bucket.astype(str) == '20+'].warm_vs_random_pct
m('PATwentyMin', w20.min(), '{:.0f}'); m('PATwentyMax', w20.max(), '{:.0f}')

av = S('Information availability')
aget = lambda sc, fs, col='mae': float(av.query('scope==@sc and feature_set==@fs')[col].iloc[0])
m('PAAvAllPlug', aget('all', 'plug_in')); m('PAAvAllCharge', aget('all', 'plus_charging'))
m('PAAvAllPct', -aget('all', 'plus_charging', 'change_vs_plug_in_pct'), '{:.0f}')
m('PAAvAllLo', -aget('all', 'plus_charging', 'ci_high'), '{:.0f}'); m('PAAvAllHi', -aget('all', 'plus_charging', 'ci_low'), '{:.0f}')
loc = av[(av.scope != 'all') & (av.feature_set == 'plus_charging')].change_vs_plug_in_pct
m('PAAvLocMin', -loc.max(), '{:.0f}'); m('PAAvLocMax', -loc.min(), '{:.0f}')
m('PAAvDepartExtra', aget('all', 'plus_charging', 'change_vs_plug_in_pct') - aget('all', 'plus_departure', 'change_vs_plug_in_pct'), '{:.1f}')

mr = S('Smart charging at matched').query("learner=='catboost' and unmet_target_pct==2.0").set_index('setting').flex_vs_oracle_pct
m('PAFlexA', mr['A'], '{:.1f}'); m('PAFlexC', mr['C'], '{:.1f}'); m('PAFlexD', mr['D'], '{:.1f}')
m('PAFlexLoss', 100 * (1 - mr['D'] / mr['C']), '{:.0f}')
acn = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', open('resultsnumbers.tex').read()))
m('FlexLoss', 100 * (1 - float(acn['FlexD']) / float(acn['FlexC'])), '{:.0f}')

cv = S('RQ2 Conformal coverage'); cv = cv[(cv.model == 'catboost') & (cv.alpha.round(2) == 0.10)]
cov = cv.groupby(['target', 'protocol']).coverage.mean()
m('PACovDuChrono', cov[('duration', 'chronological')]); m('PACovDuRand', cov[('duration', 'random')])
m('PACovEnChrono', cov[('energy', 'chronological')]); m('PACovEnSite', cov[('energy', 'site_disjoint')])

pdl = S('Pandemic delta')
pr = pdl[(pdl.protocol == 'chronological') & (pdl.model == 'catboost') & (pdl.target_site.astype(str) == 'all')].set_index('target').mae_relative_change_pct
m('PAPandDu', pr['duration'], '{:.0f}'); m('PAPandEn', pr['energy'], '{:.0f}')

with open('externalnumbers.tex', 'w') as f:
    f.write('% Generated by make_external_tex.py from the Palo Alto full report. Do not edit.\n')
    for k in sorted(macros):
        f.write(f'\\newcommand{{\\{k}}}{{{macros[k]}}}\n')

rows = [
    (r'\multicolumn{3}{l}{\textit{Protocol (CatBoost MAE, change relative to random splitting)}}', None, None),
    ('Energy, chronological', r'+\GapEnChrono\,\%', r'+\PAGapEnChrono\,\%'),
    ('Energy, unseen driver', r'+\GapEnUser\,\%', r'+\PAGapEnUser\,\%'),
    ('Energy, unseen site (ACN) or location (Palo Alto)', r'+\GapEnSite\,\%', r'+\PAGapEnSite\,\%'),
    ('Duration, chronological', r'+\GapDuChrono\,\%', r'\PAGapDuChrono\,\%'),
    ('Duration, unseen driver', r'+\GapDuUser\,\%', r'+\PAGapDuUser\,\%'),
    (r'Energy ranking agreement across protocol pairs (Kendall $\tau_b$)', r'\TauEnMin--\TauEnMax', r'\PATauEnMin--\PATauEnMax'),
    (r'\multicolumn{3}{l}{\textit{Cold-start decomposition (energy, CatBoost, change in MAE)}}', None, None),
    ('Random-split leakage', r'\LeakEnCb\,\%', r'\PALeakEnCb\,\%'),
    ('Unseen-driver cost', r'\UnseenEnCb\,\%', r'\PAUnseenEnCb\,\%'),
    ("Value of the driver's own history", r'\HistEnCb\,\%', r'\PAHistEnCb\,\%'),
    ('Total gap for unseen drivers', r'\TotEnCb\,\%', r'\PATotEnCb\,\%'),
    ('Share of the gap due to missing history, energy (both learners)', r'\ShareEnMin--\ShareEnMax\,\%', r'\PAShareEnMin--\PAShareEnMax\,\%'),
    ('Share of the gap due to missing history, duration (both learners)', r'\ShareDuMin--\ShareDuMax\,\%', r'\PAShareDuMin--\PAShareDuMax\,\%'),
    ('Largest setting difference on a first visit', r'\FirstVisitMax\,\%', r'\PAFirstVisitMax\,\%'),
    ('Energy, history versus random split after one session', r'\OneSessionEnMax\,\%', r'\PAOneSessionEnMax\,\%'),
    (r'\multicolumn{3}{l}{\textit{Information available only after plug-in (energy)}}', None, None),
    ('Error reduction from adding the charging duration', r'\AvAllPct\,\%', r'\PAAvAllPct\,\%'),
    (r'\multicolumn{3}{l}{\textit{Smart charging at 2\,\% undelivered energy (\% of oracle flexibility)}}', None, None),
    ('Random split / unseen driver with history / cold start', r'\FlexA\,/\,\FlexC\,/\,\FlexD', r'\PAFlexA\,/\,\PAFlexC\,/\,\PAFlexD'),
    ('Flexibility lost to the cold start', r'\FlexLoss\,\%', r'\PAFlexLoss\,\%'),
    (r'\multicolumn{3}{l}{\textit{Conformal coverage of 90\,\% intervals (duration)}}', None, None),
    ('Random / chronological', r'\CovDuRand\,/\,\CovDuChrono', r'\PACovDuRand\,/\,\PACovDuChrono'),
]
with open('externaltable.tex', 'w') as f:
    f.write('% Generated by make_external_tex.py. Do not edit.\n')
    f.write('\\begin{table}[!htbp]\n\\centering\n\\caption{Main findings on the ACN workplace sites and on the independent Palo Alto public network, '
            'obtained with the same protocols, learners, features and seeds. Palo Alto locations are clusters of addresses within one city, '
            'so the unseen-location protocol is a weaker shift than transfer between ACN sites. Shares of the gap are computed on a logarithmic scale and given as the range over CatBoost and the FT-Transformer.}\\label{tab:external}\n\\small\n'
            '\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{7.2cm}rr@{}}\n\\toprule\n & ACN & Palo Alto \\\\\n\\midrule\n')
    for a, b, c in rows:
        f.write((a + ' \\\\\n') if b is None else f'{a} & {b} & {c} \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
print(len(macros), 'macros;', 'externaltable.tex written')

# ------------------------------------------------------------------ Supplementary tables
import re as _re_cell
_NEGZERO = _re_cell.compile(r'(?<![\w.])-(0(?:\.0+)?)(?![.\d])')
_MINUS = _re_cell.compile(r'(?<![\w\\$.])-(?=\d)')
def fix_cell(v, col_index):
    # typeset a table cell: no negative zero, a true minus sign, and a dash for an empty data cell
    s = '' if v is None or (isinstance(v, float) and v != v) else str(v)
    if s.strip() == '':
        return '--' if col_index >= 2 else s
    parts = _re_cell.split(r'(\$[^$]*\$)', s)   # leave math segments untouched
    for k in range(0, len(parts), 2):
        parts[k] = _MINUS.sub('$-$', _NEGZERO.sub(r'\1', parts[k]))
    return ''.join(parts)


def tex_table(df, caption, label, colfmt, size='\\footnotesize'):
    out = ['\\begin{table}[htbp]', '\\centering', f'\\caption{{{caption}}}\\label{{{label}}}', size,
           f'\\begin{{tabular}}{{{colfmt}}}', '\\toprule', ' & '.join(df.columns) + ' \\\\', '\\midrule']
    rows_ = [[fix_cell(v, j) for j, v in enumerate(r)] for r in df.values]
    out += [' & '.join(row) + ' \\\\' for row in rows_]
    if any(x == '--' for row in rows_ for x in row):
        out[2] = out[2].replace('}\\label', ' A dash marks a cell that does not apply.}\\label')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


import json as _json
def _mref(label):
    """Number of a main-text label: manuscript.aux when present, else labels.json next to this script."""
    aux = Path('manuscript.aux')
    if aux.exists():
        m_ = re.search(r'\\newlabel\{' + re.escape(label) + r'\}\{\{([^}]*)\}', aux.read_text())
        if m_:
            return m_.group(1)
    lj = Path(__file__).with_name('labels.json')
    return _json.loads(lj.read_text()).get(label, '??') if lj.exists() else '??'

si = ['\\section{External validation on the Palo Alto network}',
      'These tables give the full results of the external validation summarised in Section ' + _mref('sec:res-external') + ' and Table ' + _mref('tab:external') + ' of the main text. '
      'The protocols, learners, features, seeds and statistics are those of the ACN analyses; Palo Alto locations play the role of sites.', '']
acc2 = acc.copy()
acc2['split'] = acc2.apply(lambda r: r.protocol.replace('_', ' ') if r.protocol != 'site_disjoint' else 'location: ' + str(r.target_site).replace('_', ' '), axis=1)
order = ['random', 'chronological', 'user disjoint', 'location: downtown', 'location: california ave', 'location: mitchell park', 'location: rinconada']
mods = [('catboost', 'CatBoost'), ('torch_ft', 'FT-Transformer'), ('ridge', 'Ridge'), ('user_history', 'Driver history'), ('historical_mean', 'Historical mean')]
rows = []
for t, T in [('energy', 'Energy (kWh)'), ('duration', 'Duration (h)')]:
    for sp in order:
        z = acc2[(acc2.target == t) & (acc2.split == sp)].set_index('model')
        if len(z) == 0:
            continue
        rows.append([T if sp == 'random' else '', sp] + [f'{z.loc[mm, "mae_mean"]:.3f} $\\pm$ {z.loc[mm, "mae_std"]:.3f}' if mm in z.index else '' for mm, _ in mods])
si.append(tex_table(pd.DataFrame(rows, columns=['Target', 'Protocol'] + [n for _, n in mods]),
    'Palo Alto main study: MAE (mean $\\pm$ s.d. over five seeds) by protocol and learner.', 'tab:si-pa-main', '@{}llrrrrr@{}', '\\scriptsize'))
d = dc.copy()
d['Effect (95\\% CI)'] = d.apply(lambda r: f'{r.effect_pct:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}]', axis=1)
d['Holm $p$'] = d.p_holm.map(lambda v: f'{v:.3f}')
d['target'] = d.target.map({'kwh_delivered': 'Energy', 'duration_hours': 'Duration'}); d['learner'] = d.learner.map({'catboost': 'CatBoost', 'ft': 'FT-Transformer'})
si.append(tex_table(d[['target', 'learner', 'component', 'Effect (95\\% CI)', 'Holm $p$']].rename(columns={'target': 'Target', 'learner': 'Learner', 'component': 'Component'}),
    f'Palo Alto cold-start decomposition (change in MAE, \\%, driver-cluster bootstrap; {int(dc.n_drivers.iloc[0]):,} drivers, {int(dc.n_rows.iloc[0]):,} evaluation sessions, five seeds). '
    f'Holm-adjusted $p$-values of {dc.p_holm.max():.3f} are the smallest attainable with 2,000 resamples and {len(dc)} tests.', 'tab:si-pa-decomp', '@{}lllrr@{}'))
w = wu.copy(); w['target'] = w.target.map({'kwh_delivered': 'Energy', 'duration_hours': 'Duration'}); w['learner'] = w.learner.map({'catboost': 'CatBoost', 'ft': 'FT-Transformer'})
w['A / C / D'] = w.apply(lambda r: f'{r.mae_random:.3f} / {r.mae_warm:.3f} / {r.mae_cold:.3f}', axis=1)
w['C vs A (95\\% CI)'] = w.apply(lambda r: f'{r.warm_vs_random_pct:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]', axis=1)
si.append(tex_table(w[['target', 'learner', 'k_bucket', 'n_rows', 'A / C / D', 'C vs A (95\\% CI)']].rename(columns={'target': 'Target', 'learner': 'Learner', 'k_bucket': 'Completed sessions', 'n_rows': 'Rows'}),
    'Palo Alto warm-up curve: MAE by number of the driver\'s completed sessions for settings A (random split), C (unseen driver with history) and D (cold start).', 'tab:si-pa-warmup', '@{}llrrrr@{}', '\\scriptsize'))
a = av.copy(); a['scope'] = a.scope.str.replace('_', ' '); a['feature_set'] = a.feature_set.map({'plug_in': 'plug-in', 'plus_charging': '+ charging duration', 'plus_departure': '+ session duration'})
a['MAE (kWh)'] = a.apply(lambda r: f'{r.mae:.3f} $\\pm$ {r.mae_sd:.4f}', axis=1)
a['Change (95\\% CI)'] = a.apply(lambda r: '' if pd.isna(r.change_vs_plug_in_pct) else f'{r.change_vs_plug_in_pct:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}]', axis=1)
si.append(tex_table(a[['scope', 'feature_set', 'MAE (kWh)', 'Change (95\\% CI)']].rename(columns={'scope': 'Scope', 'feature_set': 'Information'}),
    'Palo Alto information availability: energy MAE (CatBoost, five seeds) by the information available at prediction time, with the change relative to plug-in information (\\%).', 'tab:si-pa-avail', '@{}llrr@{}'))
c2 = S('RQ2 Conformal coverage'); c2 = c2[c2.alpha.round(2) == 0.10]
c2 = c2.groupby(['target', 'protocol', 'model']).coverage.mean().unstack('protocol').reset_index()
c2['model'] = c2.model.map({'catboost': 'CatBoost', 'ridge': 'Ridge', 'torch_ft': 'FT-Transformer'})
c2['target'] = c2.target.str.capitalize()
c2 = c2.rename(columns={'target': 'Target', 'model': 'Learner', 'chronological': 'Chronological', 'random': 'Random', 'site_disjoint': 'Unseen location', 'user_disjoint': 'Unseen driver'})
for col in ['Chronological', 'Random', 'Unseen location', 'Unseen driver']:
    c2[col] = c2[col].map(lambda v: f'{v:.3f}')
si.append(tex_table(c2[['Target', 'Learner', 'Random', 'Chronological', 'Unseen driver', 'Unseen location']],
    'Palo Alto split-conformal coverage of nominal 90\\,\\% intervals by protocol (mean over seeds and locations).', 'tab:si-pa-conformal', '@{}llrrrr@{}'))
open('si_external.tex', 'w').write('% Generated by make_external_tex.py. Do not edit.\n' + '\n'.join(si))
print('si_external.tex written')
