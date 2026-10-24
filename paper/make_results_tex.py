"""Write results_numbers.tex (macros) and results_tables.tex from FULL_REPORT.xlsx.

    python make_results_tex.py FULL_REPORT.xlsx [results/operational]

Every number quoted in results.tex comes from a macro defined here, so the text cannot drift
from the official results. Re-run after any rerun of the pipeline.
"""
import math
import sys
from pathlib import Path

import pandas as pd

X = pd.ExcelFile(sys.argv[1])
OPS = Path(sys.argv[2]) if len(sys.argv) > 2 else None
S = lambda name: X.parse(next(s for s in X.sheet_names if s.startswith(name)))
macros = {}


def m(name, value, fmt='{:.2f}'):
    assert name.isalpha(), name
    v = fmt.format(value) if not isinstance(value, str) else value
    # a typeset minus sign rather than a hyphen, valid in text and in math
    macros[name] = '\\ensuremath{-}' + v[1:] if v.startswith('-') else v


pct = lambda new, old: 100 * (new / old - 1)

# ---------------------------------------------------------------- RQ1 main study
acc = S('RQ1 Accuracy by protocol')
prot = acc.groupby(['target', 'protocol', 'model'])[['mae_mean', 'mae_std', 'r2_oos_mean']].mean().reset_index()
get = lambda t, p, mod, col='mae_mean': float(prot.query('target==@t and protocol==@p and model==@mod')[col].iloc[0])
names = {'random': 'Rand', 'chronological': 'Chrono', 'site_disjoint': 'Site', 'user_disjoint': 'User'}
for t, T in [('energy', 'En'), ('duration', 'Du')]:
    for p, P in names.items():
        m(f'{T}{P}', get(t, p, 'catboost'))
        if p != 'random':
            m(f'Gap{T}{P}', pct(get(t, p, 'catboost'), get(t, 'random', 'catboost')), '{:.0f}')
seed = S('RQ1 Accuracy per seed'); cb = seed[seed.model == 'catboost']
ps = cb.groupby(['target', 'protocol', 'seed']).mae.mean().unstack('protocol')
g = 100 * (ps.loc['energy', 'user_disjoint'] / ps.loc['energy', 'random'] - 1)
m('GapEnUserMin', g.min(), '{:.0f}'); m('GapEnUserMax', g.max(), '{:.0f}')
cand = prot[~prot.model.isin(['driver_estimate', 'catboost_cqr'])]
best = cand.loc[cand.groupby(['target', 'protocol']).mae_mean.idxmin()]
rb = best[best.protocol == 'random'].set_index('target').model
m('BestChanges', sum(1 for t, p, mod in best[['target', 'protocol', 'model']].values if p != 'random' and mod != rb[t]), '{:d}')
rk = S('RQ1 Model ranking stability')
m('TauEnMin', rk[rk.target == 'energy'].kendall_tau.min()); m('TauEnMax', rk[rk.target == 'energy'].kendall_tau.max())
st = S('Statistical comparisons'); hc = 'holm_adjusted_tail_probability'
site_ub = st[st.split_name.str.startswith('site') & (st.target == 'energy')
             & (st[['model_a', 'model_b']].isin(['user_history']).any(axis=1))
             & (st[['model_a', 'model_b']].isin(['catboost', 'torch_ft']).any(axis=1))]
m('SiteUHpMin', site_ub[hc].min()); m('SiteUHpMax', site_ub[hc].max())
m('SiteUHnsig', int((site_ub[hc] >= 0.05).sum()), '{:d}'); m('SiteUHn', len(site_ub), '{:d}')
# orientation: positive = learner has higher MAE than the driver-history baseline
sgn = site_ub.apply(lambda r: r.difference_a_minus_b if r.model_b == 'user_history' else -r.difference_a_minus_b, axis=1)
sig = site_ub[hc] < 0.05
m('SiteUHbaseBetter', int(((sgn > 0) & sig).sum()), '{:d}'); m('SiteUHtie', int((~sig).sum()), '{:d}'); m('SiteUHlearnBetter', int(((sgn < 0) & sig).sum()), '{:d}')
words = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}
m('SiteUHbaseBetterW', words[int(((sgn > 0) & sig).sum())]); m('SiteUHtieW', words[int((~sig).sum())]); m('SiteUHlearnBetterW', words[int(((sgn < 0) & sig).sum())]); m('SiteUHnW', words[len(site_ub)])
m('HolmSig', int((st[hc] < 0.05).sum()), '{:d}'); m('HolmN', len(st), '{:d}')

# ---------------------------------------------------------------- RQ1 benchmark
bs = S('Benchmark summary'); pw = S('Benchmark pairwise tests'); fr = S('Benchmark Friedman tests')
bget = lambda t, p, mod: float(bs.query('target==@t and protocol==@p and model==@mod').mae.iloc[0])
ties = lambda t, p: int((~pw.query('target==@t and protocol==@p').significantly_worse).sum())
total = lambda t, p: int(len(pw.query('target==@t and protocol==@p')))
m('BnEnChronoTies', ties('kwh_delivered', 'chronological'), '{:d}'); m('BnEnChronoN', total('kwh_delivered', 'chronological'), '{:d}')
m('BnEnSiteTies', ties('kwh_delivered', 'site_transfer'), '{:d}'); m('BnEnSiteN', total('kwh_delivered', 'site_transfer'), '{:d}')
m('BnRandTiesEn', ties('kwh_delivered', 'random'), '{:d}'); m('BnRandTiesDu', ties('duration_hours', 'random'), '{:d}')
rank = lambda t, p, mod: float(fr.query('target==@t and protocol==@p and model==@mod').average_rank.iloc[0])
m('RkLgbRand', rank('kwh_delivered', 'random', 'lightgbm')); m('RkLgbSite', rank('kwh_delivered', 'site_transfer', 'lightgbm'))
m('RkRidgeRand', rank('kwh_delivered', 'random', 'ridge')); m('RkRidgeSite', rank('kwh_delivered', 'site_transfer', 'ridge'))
m('BnLgbRand', bget('kwh_delivered', 'random', 'lightgbm')); m('BnLgbSite', bget('kwh_delivered', 'site_transfer', 'lightgbm'))
m('BnRidgeSite', bget('kwh_delivered', 'site_transfer', 'ridge'))
m('BnGapLgbSite', pct(bget('kwh_delivered', 'site_transfer', 'lightgbm'), bget('kwh_delivered', 'random', 'lightgbm')), '{:.0f}')
m('BnGapRidgeSite', pct(bget('kwh_delivered', 'site_transfer', 'ridge'), bget('kwh_delivered', 'random', 'ridge')), '{:.0f}')
strict = bs.query("target=='kwh_delivered' and protocol=='site_transfer_strict'")
bst = strict[strict.model != 'mean'].sort_values('mae').iloc[0]
m('StrictBest', bst.mae); m('StrictMean', float(strict[strict.model == 'mean'].mae.iloc[0]))
m('StrictGain', -pct(bst.mae, float(strict[strict.model == 'mean'].mae.iloc[0])), '{:.0f}')
m('StrictBestModel', {'catboost': 'CatBoost', 'ridge': 'ridge regression', 'ft': 'the FT-Transformer', 'lightgbm': 'LightGBM'}[bst.model])
for t, T in [('kwh_delivered', 'En'), ('duration_hours', 'Du')]:
    r0 = bget(t, 'random', 'lightgbm')
    m(f'PreGap{T}', pct(bget(t, 'chronological_prepandemic', 'lightgbm'), r0), '{:.1f}')
    m(f'ChrGap{T}', pct(bget(t, 'chronological', 'lightgbm'), r0), '{:.1f}')

# ---------------------------------------------------------------- RQ2 cold start
dc = S('Cold-start decomposition')
dget = lambda t, l, comp, col='effect_pct': float(dc.query('target==@t and learner==@l and component==@comp')[col].iloc[0])
share = {}
for t, T in [('kwh_delivered', 'En'), ('duration_hours', 'Du')]:
    for l, L in [('catboost', 'Cb'), ('ft', 'Ft')]:
        tot, hist = dget(t, l, 'total gap'), dget(t, l, 'value of own history')
        share[(T, L)] = 100 * math.log1p(hist / 100) / math.log1p(tot / 100)
        m(f'Hist{T}{L}', hist, '{:.1f}'); m(f'Tot{T}{L}', tot, '{:.1f}')
        m(f'Leak{T}{L}', dget(t, l, 'future-session leakage'), '{:.1f}'); m(f'Unseen{T}{L}', dget(t, l, 'unseen-driver cost'), '{:.1f}')
m('ShareEnMin', min(share[('En', 'Cb')], share[('En', 'Ft')]), '{:.0f}'); m('ShareEnMax', max(share[('En', 'Cb')], share[('En', 'Ft')]), '{:.0f}')
m('ShareDuMin', min(share[('Du', 'Cb')], share[('Du', 'Ft')]), '{:.0f}'); m('ShareDuMax', max(share[('Du', 'Cb')], share[('Du', 'Ft')]), '{:.0f}')
leaks = [dget(t, l, 'future-session leakage') for t in ['kwh_delivered', 'duration_hours'] for l in ['catboost', 'ft']]
m('LeakMax', max(leaks), '{:.1f}')
m('NDrivers', int(dc.n_drivers.iloc[0]), '{:d}')
wu = S('Cold-start warm-up curve')
_k0 = wu[wu.k_bucket.astype(str) == '0']
m('FirstVisitMax', max((100 * (_k0.mae_warm / _k0.mae_random - 1)).abs().max(), (100 * (_k0.mae_cold / _k0.mae_random - 1)).abs().max()), '{:.1f}')
k1 = wu[(wu.k_bucket.astype(str) == '1') & (wu.target == 'kwh_delivered')].warm_vs_random_pct
m('OneSessionEnMax', k1.max(), '{:.1f}')
w20 = wu[wu.k_bucket.astype(str) == '20+']
m('TwentyEnCb', float(w20.query("target=='kwh_delivered' and learner=='catboost'").warm_vs_random_pct.iloc[0]), '{:.1f}')
m('TwentyDuCb', float(w20.query("target=='duration_hours' and learner=='catboost'").warm_vs_random_pct.iloc[0]), '{:.1f}')
m('TwentyShare', 100 * float(w20.n_rows.iloc[0]) / wu[(wu.target == w20.target.iloc[0]) & (wu.learner == w20.learner.iloc[0])].n_rows.sum(), '{:.0f}')
hr = S('Remedy R1- gated ensemble'); u = hr[hr.protocol == 'user_disjoint'].pivot_table(index=['target', 'seed'], columns='model', values='mae')
for t, T in [('energy', 'En'), ('duration', 'Du')]:
    d = 100 * (u.loc[t, 'hrge'] / u.loc[t, 'catboost_full'] - 1)
    m(f'Hrge{T}', -d.mean(), '{:.1f}')                                  # reduction, as a positive number
    assert d.mean() < 0, 'history-gated ensemble no longer reduces error; rewrite the text'
m('HrgeWins', int((u.hrge < u.catboost_full).sum()), '{:d}'); m('HrgeUnits', len(u), '{:d}')
ab = S('Grouped ablation, five'); q = ab[(ab.protocol == 'user_disjoint') & (ab.ablation == 'no_user_history')].set_index('target')
for t, T in [('energy', 'En'), ('duration', 'Du')]:
    m(f'Abl{T}', q.loc[t, 'delta_pct_mean'], '{:.1f}'); m(f'Abl{T}SD', q.loc[t, 'delta_pct_sd'], '{:.1f}')

# ---------------------------------------------------------------- RQ3 availability
av = S('Information availability')
aget = lambda sc, fs, col='mae': float(av.query('scope==@sc and feature_set==@fs')[col].iloc[0])
m('AvAllPlug', aget('all', 'plug_in')); m('AvAllCharge', aget('all', 'plus_charging'))
m('AvAllPct', -aget('all', 'plus_charging', 'change_vs_plug_in_pct'), '{:.0f}')
m('AvAllLo', -aget('all', 'plus_charging', 'ci_high'), '{:.0f}'); m('AvAllHi', -aget('all', 'plus_charging', 'ci_low'), '{:.0f}')
m('AvBigMin', -max(aget('caltech', 'plus_charging', 'change_vs_plug_in_pct'), aget('jpl', 'plus_charging', 'change_vs_plug_in_pct')), '{:.0f}')
m('AvBigMax', -min(aget('caltech', 'plus_charging', 'change_vs_plug_in_pct'), aget('jpl', 'plus_charging', 'change_vs_plug_in_pct')), '{:.0f}')
m('AvCalPlug', aget('caltech', 'plug_in')); m('AvCalCharge', aget('caltech', 'plus_charging'))
m('AvOffPct', -aget('office001', 'plus_charging', 'change_vs_plug_in_pct'), '{:.0f}')
m('AvDepartExtra', aget('all', 'plus_charging', 'change_vs_plug_in_pct') - aget('all', 'plus_departure', 'change_vs_plug_in_pct'), '{:.1f}')

# ---------------------------------------------------------------- RQ4 uncertainty and decisions
cv = S('RQ2 Conformal coverage'); cv = cv[(cv.model == 'catboost') & (cv.alpha.round(2) == 0.10)]
cov = cv.groupby(['target', 'protocol']).coverage.mean()
m('CovDuChrono', cov[('duration', 'chronological')]); m('CovDuRand', cov[('duration', 'random')])
cc = S('Conditional coverage'); w = cc[(cc.model == 'catboost') & (cc.dimension == 'user_history_bin')]
worst = w.loc[w.coverage.idxmin()]
m('WorstCov', worst.coverage); m('WorstGroup', str(worst.group).replace('_', ' '))
m('WorstProtocol', {'chronological': 'chronological evaluation', 'site_disjoint': 'site transfer', 'user_disjoint': 'unseen drivers', 'random': 'random splitting'}[worst.protocol])
assert worst.target == 'energy' and str(worst.group) == '1-2', 'worst group changed; rewrite the text'
rc = S('RQ2 Target-site recalibrat').groupby(['target', 'target_site', 'budget'])[['coverage', 'interval_score']].mean()
m('OffCovZero', rc.loc[('energy', 'office001', 0), 'coverage']); m('OffIsZero', rc.loc[('energy', 'office001', 0), 'interval_score'], '{:.1f}')
m('OffIsHund', rc.loc[('energy', 'office001', 100), 'interval_score'], '{:.1f}')
occ = S('RQ4 Occupancy prediction'); occ = occ[occ.horizon == 1]                 # one-hour horizon
oc = occ.groupby(['protocol', 'target_site'], dropna=False)[['recommendation_hit_rate', 'confidently_wrong_rate']].mean()
m('OccHitChrono', float(oc.loc['chronological'].recommendation_hit_rate.iloc[0]))
m('OccCwChrono', float(oc.loc['chronological'].confidently_wrong_rate.iloc[0]))
m('OccHitCal', oc.loc[('site_disjoint', 'caltech'), 'recommendation_hit_rate']); m('OccHitJpl', oc.loc[('site_disjoint', 'jpl'), 'recommendation_hit_rate'])
m('OccCwJpl', oc.loc[('site_disjoint', 'jpl'), 'confidently_wrong_rate'])
m('OccCwCal', oc.loc[('site_disjoint', 'caltech'), 'confidently_wrong_rate'])
r2 = S('Remedy R2- history-condition'); r2 = r2[(r2.target == 'energy') & (r2.protocol == 'chronological') & (r2.group.astype(str) == '1-2')]
r2v = r2.groupby('variant').coverage.mean()
m('CondCovMarg', r2v['marginal']); m('CondCovCond', r2v['history_conditional'])
mr = S('Smart charging at matched').query("learner=='catboost' and unmet_target_pct==2.0").set_index('setting').flex_vs_oracle_pct
m('FlexA', mr['A'], '{:.1f}'); m('FlexC', mr['C'], '{:.1f}'); m('FlexD', mr['D'], '{:.1f}')
sq = S('R2 for the squared-error').set_index(['target', 'protocol'])
m('SqRtwoUser', sq.loc[('energy', 'user_disjoint'), 'r2_sqerr']); m('SqRtwoOosUser', sq.loc[('energy', 'user_disjoint'), 'r2_oos_sqerr'])
m('AbsRtwoUser', sq.loc[('energy', 'user_disjoint'), 'r2_abserr']); m('SqRtwoMin', sq.r2_sqerr.min()); m('SqRtwoMax', sq.r2_sqerr.max())

# ---------------------------------------------------------------- operational (optional)
if OPS and (OPS / 'request_differences.csv').exists():
    rd = pd.read_csv(OPS / 'request_differences.csv').query("learner=='catboost' and unmet_target_pct==2.0")
    r = rd.query("group=='all' and comparison=='request - D'").iloc[0]
    m('ReqMinusD', r.difference_pct_points, '{:.1f}'); m('ReqMinusDLo', r.ci_low, '{:.1f}'); m('ReqMinusDHi', r.ci_high, '{:.1f}')
    r = rd.query("group=='all' and comparison=='request - A'").iloc[0]
    m('ReqMinusA', -r.difference_pct_points, '{:.1f}'); m('ReqMinusAp', r.p_value, '{:.3f}')
    assert r.difference_pct_points < 0
if OPS and (OPS / 'site_capacity_needed.csv').exists():
    cn = pd.read_csv(OPS / 'site_capacity_needed.csv').query('undelivered_target_pct==1.0').set_index(['site', 'policy'])
    for site, Sx in [('caltech', 'Cal'), ('jpl', 'Jpl')]:
        if (site, 'oracle') not in cn.index:
            continue
        f = lambda pol: float(cn.loc[(site, pol), 'capacity_needed_fraction'])
        span = f('first come first served') - f('oracle')
        m(f'Cap{Sx}Oracle', f('oracle'), '{:.3f}'); m(f'Cap{Sx}A', f('A'), '{:.3f}'); m(f'Cap{Sx}D', f('D'), '{:.3f}'); m(f'Cap{Sx}Fcfs', f('first come first served'), '{:.3f}')
        m(f'Capture{Sx}A', 100 * (f('first come first served') - f('A')) / span, '{:.0f}')
        m(f'Capture{Sx}D', 100 * (f('first come first served') - f('D')) / span, '{:.0f}')

RED = lambda t: r'\textcolor{red}{[check: ' + t + ']}'
if 'ReqMinusD' in macros:
    rd = pd.read_csv(OPS / 'request_differences.csv').query("learner=='catboost' and unmet_target_pct==2.0 and group=='all'").set_index('comparison')
    tie_d = rd.loc['request - D', 'ci_low'] <= 0 <= rd.loc['request - D', 'ci_high']
    worse_a = rd.loc['request - A', 'ci_high'] < 0
    m('ReqLead', 'The driver\'s own request does not close this gap.' if (tie_d and worse_a)
      else RED('request result differs from the sandbox; rewrite this paragraph'))
if OPS and (OPS / 'bias_correction.csv').exists():
    bc = pd.read_csv(OPS / 'bias_correction.csv')
    bc = bc[bc.model.isin(['catboost', 'torch_ft'])]
    piv = bc.pivot_table(index=['target', 'target_site', 'model'], columns='budget', values='mae')
    chg = 100 * (piv.div(piv[0], axis=0) - 1)
    c100 = chg[100]
    m('BiasBest', -c100.min(), '{:.0f}'); m('BiasWorst', c100.max(), '{:.0f}')
    m('BiasSmallWorst', chg[[b for b in chg.columns if 0 < b < 50]].max().max(), '{:.0f}')
    r2 = bc.pivot_table(index=['target', 'target_site', 'model'], columns='budget', values='r2_oos')
    m('BiasOffRtwoZero', r2.loc[('energy', 'office001', 'catboost'), 0]); m('BiasOffRtwoHund', r2.loc[('energy', 'office001', 'catboost'), 100])
    m('BiasCalCb', -c100.loc[('energy', 'caltech', 'catboost')], '{:.0f}'); m('BiasCalFt', c100.loc[('energy', 'caltech', 'torch_ft')], '{:.0f}')
    ok = c100.min() < -2 and c100.max() > 2
    m('BiasVerdict', 'has mixed effects' if ok else RED('bias correction is no longer mixed; rewrite this paragraph'))
if OPS and (OPS / 'site_level.csv').exists():
    sl = pd.read_csv(OPS / 'site_level.csv').groupby('site')[['n_controlled', 'n_uncontrolled']].first()
    for site, Sx in [('caltech', 'Cal'), ('jpl', 'Jpl')]:
        if site in sl.index:
            m(f'Ctl{Sx}', 100 * sl.loc[site, 'n_controlled'] / sl.loc[site, ['n_controlled', 'n_uncontrolled']].sum(), '{:.0f}')
    rqm = pd.read_csv(OPS / 'request_matched.csv').query("learner=='catboost' and unmet_target_pct==2.0")
    m('NReqFirst', int(rqm[rqm.group == 'first visit'].n_sessions_per_seed.iloc[0]), '{:d}')
    m('NReqAll', int(rqm[rqm.group == 'all'].n_sessions_per_seed.iloc[0]), '{:,d}')
Path('resultsnumbers.tex').write_text('% generated by make_results_tex.py; do not edit by hand\n' +
                                       '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')

# ---------------------------------------------------------------- tables
L = []
L += [r'\begin{table}[!tbp]', r'\centering\footnotesize',
      r'\caption{Mean absolute error of CatBoost and the FT-Transformer under each evaluation protocol. Values are the mean $\pm$ one standard deviation over five seeds; unseen-site values average the three target sites with equal weight. All protocols use the frozen split definitions with label-maturation purging (\cref{sec:protocols}). Duration in hours, energy in kWh.}',
      r'\label{tab:accuracy}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{llcccc}', r'\toprule',
      r'Target & Learner & Random & Chronological & \makecell{Unseen\\site} & \makecell{Unseen\\driver} \\', r'\midrule']
for t, tn in [('duration', 'Duration'), ('energy', 'Energy')]:
    for mod, mn in [('catboost', 'CatBoost'), ('torch_ft', 'FT-Transformer')]:
        cells = [f"{get(t, p, mod):.2f} $\\pm$ {get(t, p, mod, 'mae_std'):.2f}" for p in names]
        L.append(f'{tn} & {mn} & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}', '']

L += [r'\begin{table}[!tbp]', r'\centering\footnotesize',
      r'\caption{Twelve-learner benchmark under six protocols. For each target and protocol: the learner with the lowest mean MAE; the number of other learners not significantly worse than it (Nadeau--Bengio corrected resampled $t$-test, Holm-adjusted, $\alpha=0.05$); and the mean Friedman rank over folds (1 = best) of LightGBM, the best learner under random splitting, and of ridge regression. Random, driver-grouped and chronological protocols use 25 folds; unseen-site protocols use 15. The two sensitivity protocols (pre-2020 and strict site transfer) compare six learners.}',
      r'\label{tab:benchmark}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{llcccc}', r'\toprule',
      r'Target & Protocol & Best learner & \makecell{Tied with\\best} & \makecell{Rank\\LightGBM} & \makecell{Rank\\ridge} \\', r'\midrule']
pn = {'random': 'Random', 'driver_grouped': 'Driver-grouped', 'chronological': 'Chronological',
      'chronological_prepandemic': 'Chronological, pre-2020', 'site_transfer': 'Unseen site', 'site_transfer_strict': 'Unseen site, strict'}
mn = {'lightgbm': 'LightGBM', 'catboost': 'CatBoost', 'ft': 'FT-Transformer', 'ridge': 'Ridge', 'xgboost': 'XGBoost', 'hgb': 'HistGB'}
for t, tn in [('duration_hours', 'Duration'), ('kwh_delivered', 'Energy')]:
    for p in pn:
        g = pw.query('target==@t and protocol==@p')
        L.append(f"{tn} & {pn[p]} & {mn.get(g.best_model.iloc[0], g.best_model.iloc[0])} & {ties(t, p)} of {total(t, p)} & "
                 f"{rank(t, p, 'lightgbm'):.2f} & {rank(t, p, 'ridge'):.2f}" + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}', '']

L += [r'\begin{table}[!tbp]', r'\centering\footnotesize',
      r'\caption{Cross-fitted decomposition of the gap for unseen drivers on the same 7,354 evaluation sessions of 549 drivers (\cref{eq:components,eq:decomp}). Each component is the relative change in MAE between two settings: random-split leakage (driver-temporal versus random split), unseen-driver cost (unseen driver with own history versus driver-temporal) and value of own history (cold start versus unseen driver with history). Values are means over five seeds with 95\,\% driver-cluster bootstrap intervals; all components are significant (Holm-adjusted $p\le' + f'{dc.p_holm.max():.3f}' + r'$, the smallest value attainable with 2,000 resamples and ' + f'{len(dc)}' + r' tests).}',
      r'\label{tab:decomp}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{llcccc}', r'\toprule',
      r'Target & Learner & \makecell{Random-split\\leakage} & \makecell{Unseen-driver\\cost} & \makecell{Value of\\own history} & Total gap \\', r'\midrule']
fmt = lambda t, l, c: f"{dget(t, l, c):.1f} [{dget(t, l, c, 'ci_low'):.1f}, {dget(t, l, c, 'ci_high'):.1f}]"
for t, tn in [('duration_hours', 'Duration'), ('kwh_delivered', 'Energy')]:
    for l, ln in [('catboost', 'CatBoost'), ('ft', 'FT-Transformer')]:
        L.append(f'{tn} & {ln} & ' + ' & '.join(fmt(t, l, c) for c in ['future-session leakage', 'unseen-driver cost', 'value of own history', 'total gap']) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}', '']

L += [r'\begin{table}[!tbp]', r'\centering\footnotesize',
      r'\caption{Energy MAE (kWh) under nested information sets, CatBoost, five-fold cross-fitting over five seeds; change relative to plug-in information with 95\% bootstrap intervals.}',
      r'\label{tab:availability}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{lccc}', r'\toprule',
      r'Sites & Plug-in & + charging duration & + session duration \\', r'\midrule']
for sc, sn in [('all', 'All'), ('caltech', 'Caltech'), ('jpl', 'JPL'), ('office001', 'Office001')]:
    c = lambda fs: f"{aget(sc, fs):.2f} ({aget(sc, fs, 'change_vs_plug_in_pct'):.0f}\\%)"
    L.append(f"{sn} & {aget(sc, 'plug_in'):.2f} & {c('plus_charging')} & {c('plus_departure')}" + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}', '']
blocks = '\n'.join(L).split(r'\end{table}')
for name, b in zip(['protocols', 'benchmark', 'decomp', 'availability'], blocks):
    Path(f'resultstable{name}.tex').write_text(b.strip() + '\n' + r'\end{table}' + '\n')
print(f'{len(macros)} macros, 4 tables written')
