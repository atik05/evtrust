"""Supplementary Information: the ACN tables and figures S1 to S4.

    python make_si.py ACN_RESULTS_DIR PALOALTO_FULL_REPORT.xlsx

Writes si_body.tex and figs/si_*.pdf. Every number comes from the pipeline outputs; section and table
numbers of the main text are read from manuscript.aux when present, otherwise from labels.json.
The Palo Alto tables are written by make_external_tex.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import re as _re
import json as _json
_AUX = Path('manuscript.aux').read_text() if Path('manuscript.aux').exists() else ''
_LABELS_FILE = Path(__file__).with_name('labels.json')
_LABELS = _json.loads(_LABELS_FILE.read_text()) if _LABELS_FILE.exists() else {}
def ref(label):
    """Number of a label in the manuscript: from manuscript.aux when present, else from labels.json."""
    m = _re.search(r'\\newlabel\{' + _re.escape(label) + r'\}\{\{([^}]*)\}', _AUX)
    if m:
        return m.group(1)
    if label in _LABELS:
        return _LABELS[label]
    raise SystemExit(f'label {label} not found in manuscript.aux or labels.json')

R, PAX = Path(sys.argv[1]), Path(sys.argv[2])   # ACN results directory, Palo Alto FULL_REPORT.xlsx
OUT = Path('figs'); OUT.mkdir(exist_ok=True)
L = []
add = L.append
MN = {'catboost': 'CatBoost', 'torch_ft': 'FT-Transformer', 'ft': 'FT-Transformer', 'ridge': 'Ridge', 'catboost_cqr': 'CatBoost CQR',
      'historical_mean': 'Historical mean', 'user_history': 'Driver history', 'driver_estimate': 'Driver request',
      'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'hgb': 'HistGB', 'random_forest': 'Random forest', 'extra_trees': 'Extra trees',
      'mlp': 'MLP', 'tabpfn': 'TabPFN v2', 'mean': 'Historical mean'}
PN = {'random': 'Random', 'chronological': 'Chronological', 'site_disjoint': 'Unseen site', 'user_disjoint': 'Unseen driver',
      'driver_grouped': 'Driver-grouped', 'chronological_prepandemic': 'Chronological, pre-2020', 'site_transfer': 'Unseen site',
      'site_transfer_strict': 'Unseen site, strict'}
SN = {'caltech': 'Caltech', 'jpl': 'JPL', 'office001': 'Office001'}
TN = {'duration': 'Duration', 'energy': 'Energy', 'duration_hours': 'Duration', 'kwh_delivered': 'Energy'}
esc = lambda s: str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def site_of(split):
    return SN.get(str(split).split(':')[-1], '') if ':' in str(split) else ''


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


def table(df, cols, header, caption, label, align, long=False, size=r'\small'):
    cells = [[fix_cell(esc(r[c]) if not isinstance(r[c], str) or '$' not in r[c] else r[c], j) for j, c in enumerate(cols)] for _, r in df.iterrows()]
    if any(x == '--' for row in cells for x in row):
        caption = caption.rstrip() + ' A dash marks a cell that does not apply.'
    body = [' & '.join(row) + r' \\' for row in cells]
    if long:
        add(r'{' + size + r'\begin{longtable}{' + align + '}')
        add(r'\caption{' + caption + r'}\label{' + label + r'}\\' + '\n\\toprule\n' + header + r' \\' + '\n\\midrule\n\\endfirsthead')
        add(r'\toprule' + '\n' + header + r' \\' + '\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot')
        L.extend(body); add(r'\end{longtable}}'); add('')
    else:
        add(r'\begin{table}[H]\centering' + size)
        add(r'\caption{' + caption + r'}\label{' + label + '}')
        add(r'\begin{tabular}{' + align + r'}\toprule' + '\n' + header + r' \\\midrule')
        L.extend(body); add(r'\bottomrule\end{tabular}\end{table}'); add('')


f2 = lambda v: '' if pd.isna(v) else f'{v:.2f}'
f3 = lambda v: '' if pd.isna(v) else f'{v:.3f}'
f1 = lambda v: '' if pd.isna(v) else f'{v:.1f}'
pm = lambda a, b: '' if pd.isna(a) else (f'{a:.2f} $\\pm$ {b:.2f}' if not pd.isna(b) else f'{a:.2f}')

# ===================================================================== S1 data and audit
add(r'\section{Data and leakage audit}')
add(r'This section documents the data used in the main text (Section~' + ref('sec:data') + r') and the automated audit of every split (Section~' + ref('sec:protocols') + r').')
df = pd.read_csv(R / 'metrics/data_flow.csv')
df['stage'] = df.stage.str.replace('_', ' ').str.capitalize(); df['removed'] = df.removed_from_previous.map(lambda v: '' if pd.isna(v) else f'{int(v):,}')
df['sessions'] = df.sessions.map(lambda v: f'{int(v):,}')
table(df, ['stage', 'sessions', 'removed'], 'Stage & Sessions & Removed', 'Data cleaning: sessions remaining after each rule.', 'tab:s-flow', 'lrr')
a = json.load(open(R / 'metrics/leakage_audit.json'))
rows = []
for sc in a['split_checks']:
    ov = sum(v for v in sc['index_overlap'].values())
    uo = sc.get('user_overlap'); uo = '' if uo is None else (sum(uo.values()) if isinstance(uo, dict) else uo)
    s = sc['sizes']
    rows.append({'split': sc['name'].replace('site_disjoint:', 'Unseen site, ').replace('_', ' '), 'train': f"{s['train']:,}", 'val': f"{s['validation']:,}",
                 'cal': f"{s['calibration']:,}", 'test': f"{s['test']:,}", 'ov': str(ov), 'uo': str(uo)})
sci = '; '.join(f"{c['split_name'].replace('_', ' ')}: {'passed' if c['ok'] else 'FAILED'}" for c in a['scientific_split_checks'])
table(pd.DataFrame(rows), ['split', 'train', 'val', 'cal', 'test', 'ov', 'uo'],
      'Split & Train & Validation & Calibration & Test & Shared rows & Shared drivers',
      'Split sizes (seed of the frozen definitions) and audit results. Shared rows counts sessions that appear in more than one set; '
      'shared drivers is reported for the driver-disjoint split. Forbidden features used: ' + (', '.join(a['forbidden_features_used']) or 'none')
      + '. Temporal and maturation checks: ' + esc(sci) + '.', 'tab:s-audit', 'lrrrrrr', size=r'\footnotesize')
xo = pd.read_csv(R / 'metrics/cross_site_user_overlap.csv')
xo['site'] = xo.target_site.map(SN); xo['frac'] = (100 * xo.fraction_user_history_transferable).map(f1)
for c in ['identified_sessions', 'sessions_with_user_seen_at_source', 'unique_target_users', 'unique_target_users_seen_at_source']:
    xo[c] = xo[c].map(lambda v: f'{int(v):,}')
table(xo, ['site', 'identified_sessions', 'sessions_with_user_seen_at_source', 'frac', 'unique_target_users', 'unique_target_users_seen_at_source'],
      r'Target site & Identified sessions & With history from a source site & Share (\%) & Drivers & Drivers seen at a source',
      'Driver history that can transfer from the source sites to each target site.', 'tab:s-xsite', 'lrrrrr', size=r'\footnotesize')

# ===================================================================== S2 main-study accuracy
add(r'\section{Main-study accuracy}')
add(r'Full results of the main study (Section~' + ref('sec:res-rq1') + r'): every model under every protocol and target site, mean and standard deviation over five seeds.')
s = pd.read_csv(R / 'metrics/regression_metrics_summary.csv')
s['Protocol'] = s.protocol.map(PN) + s.split_name.map(lambda x: (', ' + site_of(x)) if site_of(x) else '')
s['Target'] = s.target.map(TN); s['Model'] = s.model.map(MN)
s['MAE'] = [pm(a_, b_) for a_, b_ in zip(s.mae_mean, s.mae_std)]
s['RMSE'] = s.rmse_mean.map(f2); s['R2'] = s.r2_mean.map(f3); s['R2o'] = s.r2_oos_mean.map(f3); s['skill'] = s.mae_skill_mean.map(f3)
order = {'random': 0, 'chronological': 1, 'user_disjoint': 2, 'site_disjoint': 3}
s = s.assign(o=s.protocol.map(order)).sort_values(['target', 'o', 'split_name', 'mae_mean'])
table(s, ['Target', 'Protocol', 'Model', 'MAE', 'RMSE', 'R2', 'R2o', 'skill'],
      r'Target & Protocol & Model & MAE & RMSE & $R^2$ & $R^2_{\text{oos}}$ & MAE skill',
      r'Accuracy of all models in the main study. Duration in hours, energy in kWh. $R^2_{\text{oos}}$ and MAE skill are relative to the training-mean forecast. '
      r'Negative $R^2$ values mean a larger squared error than the reference mean; they occur for drivers\textquoteright{} own requests and for learners trained on absolute error under unseen drivers or sites (Methods). '
      r'The driver request is evaluated only on sessions where a request exists.', 'tab:s-acc', 'lllrrrrr', long=True, size=r'\footnotesize')
sq = pd.read_csv(R / 'metrics/squared_error_variant.csv')
ab = pd.read_csv(R / 'metrics/regression_metrics_by_seed.csv'); ab = ab[ab.model == 'catboost']
k = ['target', 'split_name', 'seed']
mm = sq[k + ['rmse', 'r2', 'r2_oos']].merge(ab[k + ['rmse', 'r2', 'r2_oos']], on=k, suffixes=('_sq', '_ab'))
g = mm.groupby(['target', 'split_name'])[['rmse_ab', 'rmse_sq', 'r2_ab', 'r2_sq', 'r2_oos_ab', 'r2_oos_sq']].mean().reset_index()
g['Target'] = g.target.map(TN); g['Split'] = g.split_name.map(lambda x: PN.get(x.split(':')[0]) + ((', ' + site_of(x)) if site_of(x) else ''))
for c in ['rmse_ab', 'rmse_sq']: g[c] = g[c].map(f2)
for c in ['r2_ab', 'r2_sq', 'r2_oos_ab', 'r2_oos_sq']: g[c] = g[c].map(f3)
table(g, ['Target', 'Split', 'rmse_ab', 'rmse_sq', 'r2_ab', 'r2_sq', 'r2_oos_ab', 'r2_oos_sq'],
      r'Target & Split & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{$R^2$} & \multicolumn{2}{c}{$R^2_{\text{oos}}$} \\ & & abs. & sq. & abs. & sq. & abs. & sq.',
      r'CatBoost trained on absolute error (abs.) and on squared error (sq.), by split and target site (Section~' + ref('sec:stats') + r'). At individual unseen sites $R^2$ can be negative '
      r'for either model because the site mean differs from the source sites.', 'tab:s-sq', 'llrrrrrr', size=r'\footnotesize')
po = pd.read_csv(R / 'metrics/protocol_optimism.csv'); po = po[po.model.isin(['catboost', 'torch_ft', 'ridge', 'user_history'])]
po['Target'] = po.target.map(TN); po['Model'] = po.model.map(MN); po['Protocol'] = po.protocol.map(PN); po['pct'] = po.relative_optimism_pct.map(f1)
po = po.pivot_table(index=['Target', 'Model'], columns='Protocol', values='pct', aggfunc='first').reset_index()
cols = [c for c in ['Chronological', 'Unseen driver', 'Unseen site'] if c in po.columns]
table(po, ['Target', 'Model'] + cols, 'Target & Model & ' + ' & '.join(cols),
      r'Relative increase in MAE over random splitting (\%), Eq.~(' + ref('eq:optimism') + r'); unseen-site values average the three target sites.', 'tab:s-opt', 'll' + 'r' * len(cols))
poi = pd.read_csv(R / 'metrics/protocol_optimism_identified_users.csv'); poi = poi[poi.model.isin(['catboost', 'torch_ft', 'ridge', 'user_history'])]
poi['Target'] = poi.target.map(TN); poi['Model'] = poi.model.map(MN); poi['Protocol'] = poi.protocol.map(PN); poi['pct'] = poi.relative_optimism_pct.map(f1)
poi = poi.pivot_table(index=['Target', 'Model'], columns='Protocol', values='pct', aggfunc='first').reset_index()
cols_i = [c for c in ['Chronological', 'Unseen driver', 'Unseen site'] if c in poi.columns]
table(poi, ['Target', 'Model'] + cols_i, 'Target & Model & ' + ' & '.join(cols_i),
      r'Sensitivity analysis: relative increase in MAE over random splitting (\%), Eq.~(' + ref('eq:optimism') + r'), with every protocol restricted to sessions of identified drivers (Section~' + ref('sec:measures') + r'). '
      r'Every deployment-like protocol remains worse than random splitting.', 'tab:s-optid', 'll' + 'r' * len(cols_i))
ds = pd.read_csv(R / 'metrics/distribution_shift.csv')
rows_ds = []
for pr in ['random', 'chronological', 'user_disjoint', 'site_disjoint']:
    z = ds[ds.protocol == pr]
    num_, cat_ = z[z.feature_type == 'numeric'], z[z.feature_type == 'categorical']
    top_ = num_.loc[num_.normalized_shift.idxmax()]
    rows_ds.append({'Protocol': PN[pr], 'nm': f3(num_.normalized_shift.mean()), 'nx': f3(num_.normalized_shift.max()),
                    'nf': str(top_.feature).replace('_', ' '), 'cm': f3(cat_.normalized_shift.mean()), 'cx': f3(cat_.normalized_shift.max())})
table(pd.DataFrame(rows_ds), ['Protocol', 'nm', 'nx', 'nf', 'cm', 'cx'],
      r'Protocol & \multicolumn{3}{c}{Numeric features} & \multicolumn{2}{c}{Categorical features} \\ & mean & max & most shifted & mean & max',
      r'Shift between training and test features (Section~' + ref('sec:measures') + r'): Wasserstein-1 distance divided by the training interquartile range for numeric features, '
      r'Jensen--Shannon divergence for categorical features; unseen-site values are pooled over the three target sites. Unseen drivers show little feature shift although their error rises most.',
      'tab:s-shift', 'lrrlrr', size=r'\footnotesize')

# ===================================================================== S3 benchmark
add(r'\section{Twelve-learner benchmark}')
add(r'Complete benchmark results (Sections~' + ref('sec:benchmark') + r' and~' + ref('sec:res-rq1') + r'). Every learner uses one fixed configuration; folds are five-fold cross-validation repeated five times, '
    r'five rolling-origin folds repeated five times for chronological protocols, and leave-one-site-out repeated five times for site transfer.')
b = pd.read_csv(R / 'benchmark/summary.csv')
for t in ['kwh_delivered', 'duration_hours']:
    z = b[b.target == t].copy(); z['v'] = [pm(m_, s_) for m_, s_ in zip(z.mae, z.mae_sd)]
    p = z.pivot_table(index='model', columns='protocol', values='v', aggfunc='first')
    pr = [c for c in ['random', 'driver_grouped', 'chronological', 'chronological_prepandemic', 'site_transfer', 'site_transfer_strict'] if c in p.columns]
    p = p[pr].reset_index(); p['Model'] = p.model.map(MN); p = p.fillna('')
    p = p.assign(o=b[(b.target == t) & (b.protocol == 'random')].set_index('model').mae.reindex(p.model).to_numpy()).sort_values('o')
    hdr = 'Learner & ' + ' & '.join(PN[c].replace(', ', r'\newline ') for c in pr)
    table(p, ['Model'] + pr, hdr, f'Benchmark MAE ({"kWh" if t == "kwh_delivered" else "h"}), {TN[t].lower()}: mean $\\pm$ standard deviation over folds. '
          'Sensitivity protocols use six learners.', f'tab:s-bench-{TN[t].lower()}', 'l' + 'p{2.0cm}' * len(pr), size=r'\scriptsize')
# ---------------------------------------------------------------- critical-difference diagrams (drawn from the ranks)
def cd_figure(target, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt_
    plt_.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'], 'font.size': 7,
                          'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fz_ = pd.read_csv(R / 'benchmark/friedman.csv'); fz_ = fz_[fz_.target == target]
    prots = ['random', 'driver_grouped', 'chronological', 'chronological_prepandemic', 'site_transfer', 'site_transfer_strict']
    fig_, axs_ = plt_.subplots(3, 2, figsize=(190 / 25.4, 168 / 25.4), gridspec_kw={'hspace': 0.24, 'wspace': 0.2})
    for i, (ax_, pr) in enumerate(zip(axs_.ravel(), prots)):
        z = fz_[fz_.protocol == pr].sort_values('average_rank')
        ranks = z.average_rank.to_numpy(); names = [MN.get(m_, m_) for m_ in z.model]; k = int(z.n_models.iloc[0]); cd = float(z.nemenyi_cd.iloc[0])
        mx = 0.66 * k
        ax_.set_xlim(1 - mx, k + mx); ax_.set_ylim(0.02, 1.5); ax_.axis('off')
        ax_.plot([1, k], [1, 1], color='#1F2933', lw=0.8)
        for r_ in range(1, k + 1):
            ax_.plot([r_, r_], [1, 1.035], color='#1F2933', lw=0.6); ax_.text(r_, 1.05, str(r_), ha='center', va='bottom', fontsize=5.8)
        ax_.plot([1, 1 + cd], [1.22, 1.22], color='#D55E00', lw=1.2)
        for xx in (1, 1 + cd):
            ax_.plot([xx, xx], [1.195, 1.245], color='#D55E00', lw=1.0)
        ax_.text(1 + cd / 2, 1.255, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=5.8, color='#D55E00')
        half = (len(ranks) + 1) // 2
        for j, (rk, nm) in enumerate(zip(ranks, names)):
            left = j < half; level = j if left else len(ranks) - 1 - j
            y_ = 0.78 - 0.118 * level
            xe = 1 - 0.08 * k if left else k + 0.08 * k
            ax_.plot([rk, rk, xe], [1, y_, y_], color='#5B6670', lw=0.55)
            ax_.text(xe - 0.03 * k if left else xe + 0.03 * k, y_, f'{nm} ({rk:.1f})' if left else f'({rk:.1f}) {nm}',
                     ha='right' if left else 'left', va='center', fontsize=5.9)
        cliques = []
        for a_ in range(len(ranks)):
            b_ = a_
            while b_ + 1 < len(ranks) and ranks[b_ + 1] - ranks[a_] < cd:
                b_ += 1
            if b_ > a_ and not any(c0 <= a_ and b_ <= c1 for c0, c1 in cliques):
                cliques.append((a_, b_))
        for q, (a_, b_) in enumerate(cliques):
            yq = 0.95 - 0.034 * (q % 5)
            ax_.plot([ranks[a_] - 0.04, ranks[b_] + 0.04], [yq, yq], color='#1F2933', lw=2.0, solid_capstyle='round')
        ax_.text(1 - mx, 1.49, f'({chr(97 + i)}) {PN[pr]}', ha='left', va='top', fontsize=7.5)
    fig_.savefig(path, bbox_inches='tight'); plt_.close(fig_)

pw = pd.read_csv(R / 'benchmark/pairwise_vs_best.csv')
pw['Target'] = pw.target.map(TN); pw['Protocol'] = pw.protocol.map(PN); pw['Best'] = pw.best_model.map(MN); pw['Learner'] = pw.model.map(MN)
pw['d'] = pw.mae_difference.map(f3); pw['t'] = pw.t_corrected.map(f2); pw['p'] = pw.p_value.map(f3); pw['ph'] = pw.p_holm.map(f3)
pw['sig'] = pw.significantly_worse.map({True: 'yes', False: 'no'})
table(pw, ['Target', 'Protocol', 'Best', 'Learner', 'd', 't', 'p', 'ph', 'sig'],
      r'Target & Protocol & Best & Learner & $\Delta$MAE & $t$ & $p$ & $p_{\text{Holm}}$ & Worse',
      r'Corrected resampled $t$-tests of every learner against the learner with the lowest MAE, Holm-adjusted within each target and protocol.',
      'tab:s-pairwise', 'llllrrrrl', long=True, size=r'\scriptsize')
for t, tl in [('kwh_delivered', 'energy'), ('duration_hours', 'duration')]:
    cd_figure(t, OUT / f'si_cd_{tl}.pdf')
    add(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{figs/si_cd_' + tl + r'.pdf}')
    fz = pd.read_csv(R / 'benchmark/friedman.csv'); fz = fz[fz.target == t].groupby('protocol').first()
    cdv = lambda pr: f"{fz.loc[pr, 'nemenyi_cd']:.2f}"
    pv = fz.friedman_p
    big = [pr for pr in pv.index if pv[pr] >= 1e-3]
    ptxt = ('the Friedman test rejects equal ranks in every panel ($p<0.001$' +
            (''.join(f', except $p={pv[pr]:.3f}$ in ({chr(97 + ["random","driver_grouped","chronological","chronological_prepandemic","site_transfer","site_transfer_strict"].index(pr))})' for pr in big)) + ')')
    best_rank = fz['model'] if 'model' in fz else None
    add(r'\caption{Critical-difference diagrams for ' + tl + r' MAE in the six benchmark protocols. Each diagram places the learners at their mean Friedman rank over folds '
        r'(1 = lowest MAE in a fold); learners joined by a thick horizontal bar do not differ significantly by the Nemenyi post-hoc test ($\alpha=0.05$), and the bar above each axis '
        r'shows the critical difference (CD). (a) Random splitting, (b) driver-grouped and (c) chronological rolling-origin evaluation use 25 folds and twelve learners (CD $=' + cdv('random') + r'$); '
        r'(d) chronological evaluation before March 2020 uses 25 folds and six learners (CD $=' + cdv('chronological_prepandemic') + r'$); (e) an unseen site uses 15 folds, three '
        r'left-out sites repeated five times, and twelve learners (CD $=' + cdv('site_transfer') + r'$); (f) a strict unseen site, with no target-site information, uses 15 folds and six '
        r'learners (CD $=' + cdv('site_transfer_strict') + r'$). ' + ptxt[0].upper() + ptxt[1:] + r'. Ranks are computed per fold, so the learner with the best mean rank need not have '
        r'the lowest mean MAE (Table~\ref{' + ('tab:s-bench-energy' if t == 'kwh_delivered' else 'tab:s-bench-duration') + r'}). Numbers in parentheses are the mean ranks.}\label{fig:s-cd-' + tl + r'}\end{figure}')

# ===================================================================== S4 cold start
add(r'\section{Cold start}')
add(r'Detail for Section~' + ref('sec:res-rq2') + r': the warm-up curve for both learners, the history-gated ensemble and the five-seed grouped ablation.')
wu = pd.read_csv(R / 'paper/coldstart_warmup.csv')
wu['Target'] = wu.target.map(TN); wu['Learner'] = wu.learner.map(MN); wu['k'] = wu.k_bucket.astype(str)
for c in ['mae_random', 'mae_warm', 'mae_cold']: wu[c] = wu[c].map(f3)
wu['gap'] = [f'{a_:.1f} [{b_:.1f}, {c_:.1f}]' for a_, b_, c_ in zip(wu.warm_vs_random_pct, wu.ci_low, wu.ci_high)]
wu['n'] = wu.n_rows.map(lambda v: f'{int(v):,}')
table(wu, ['Target', 'Learner', 'k', 'n', 'mae_random', 'mae_warm', 'mae_cold', 'gap'],
      r'Target & Learner & Sessions & Rows & Random & Unseen, history & Cold start & Unseen vs random (\%)',
      r'MAE by the driver\textquotesingle s number of completed sessions, with the relative gap of the unseen-driver setting with history over random splitting and its 95\% driver-cluster interval.',
      'tab:s-warm', 'lllrrrrr', long=True, size=r'\footnotesize')
ca = pd.read_csv(R / 'metrics/contribution_accuracy.csv')
ca = ca[ca.model.isin(['catboost_full', 'catboost_no_history', 'hrge'])]
grp = [c for c in ['mae', 'mae_0', 'mae_1-2', 'mae_3-9', 'mae_10+'] if c in ca.columns]
cg = ca.groupby(['target', 'protocol', 'model'])[grp].mean().reset_index()
cg['Target'] = cg.target.map(TN); cg['Protocol'] = cg.protocol.map(PN)
cg['Model'] = cg.model.map({'catboost_full': 'With history', 'catboost_no_history': 'Without history', 'hrge': 'Gated ensemble'})
for c in grp: cg[c] = cg[c].map(f3)
table(cg, ['Target', 'Protocol', 'Model'] + grp, r'Target & Protocol & Model & All & 0 & 1--2 & 3--9 & 10+',
      'History-gated ensemble: MAE overall and by the number of completed driver sessions, mean over five seeds.', 'tab:s-hrge', 'lll' + 'r' * len(grp), long=True, size=r'\footnotesize')
gw = pd.read_csv(R / 'metrics/contribution_gate.csv')
gw = gw.groupby(['target', 'protocol', 'group']).weight.mean().unstack('group').reset_index()
gcols = [c for c in ['0', '1-2', '3-9', '10+', 'pooled'] if c in gw.columns]
gw['Target'] = gw.target.map(TN); gw['Protocol'] = gw.protocol.map(PN)
for c in gcols: gw[c] = gw[c].map(f2)
table(gw, ['Target', 'Protocol'] + gcols, 'Target & Protocol & ' + ' & '.join(gcols),
      'Weight of the model with history chosen by the gate on validation data, by history group (1 = model with history only).', 'tab:s-gate', 'll' + 'r' * len(gcols), size=r'\footnotesize')
ab5 = pd.read_csv(R / 'metrics/grouped_ablation_seeds_summary.csv'); ab5 = ab5[ab5.ablation != 'full']
ab5['Target'] = ab5.target.map(TN); ab5['Protocol'] = ab5.protocol.map(PN)
ab5['Removed'] = ab5.ablation.map({'no_driver': 'Driver request', 'no_user_history': 'Driver history', 'no_station_history': 'Station history', 'calendar_only': 'All but calendar'})
ab5['d'] = [f'{a_:+.1f} $\\pm$ {b_:.1f}' for a_, b_ in zip(ab5.delta_pct_mean, ab5.delta_pct_sd)]
ab5 = ab5.pivot_table(index=['Target', 'Removed'], columns='Protocol', values='d', aggfunc='first').reset_index()
pc = [c for c in ['Random', 'Chronological', 'Unseen site', 'Unseen driver'] if c in ab5.columns]
table(ab5, ['Target', 'Removed'] + pc, 'Target & Removed group & ' + ' & '.join(pc),
      r'Grouped feature ablation over five seeds: change in CatBoost MAE (\%) when a feature group is removed, mean $\pm$ standard deviation.', 'tab:s-ablation', 'll' + 'r' * len(pc), size=r'\footnotesize')

# ===================================================================== S5 uncertainty
add(r'\section{Information availability}')
av_ = pd.read_csv(R / 'paper/feature_availability.csv')
av_['Scope'] = av_.scope.map(lambda v: {'all': 'All sites', 'caltech': 'Caltech', 'jpl': 'JPL', 'office001': 'Office001'}.get(v, v))
av_['Information'] = av_.feature_set.map({'plug_in': 'plug-in', 'plus_charging': '+ charging duration', 'plus_departure': '+ session duration'})
av_['MAE (kWh)'] = av_.apply(lambda r: f'{r.mae:.3f} $\\pm$ {r.mae_sd:.4f}', axis=1)
av_['chg'] = av_.apply(lambda r: '' if pd.isna(r.change_vs_plug_in_pct) else f'{r.change_vs_plug_in_pct:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}]', axis=1)
table(av_, ['Scope', 'Information', 'MAE (kWh)', 'chg'], r'Scope & Information & MAE (kWh) & Change (95\% CI)',
      r'Energy MAE at the ACN sites (CatBoost, five seeds) by the information available at prediction time, with the change relative to plug-in information (\%; Section~' + ref('sec:availability') + r').',
      'tab:s-avail', 'llrr')

add(r'\section{Uncertainty}')
add(r'Detail for Section~' + ref('sec:res-rq4') + r': coverage of every interval method, coverage by driver history, and recalibration on the target site.')
cr = pd.read_csv(R / 'metrics/conformal_reliability.csv'); cr = cr[cr.alpha.round(2) == 0.10]
cr['site'] = cr.split_name.map(site_of)
c2 = cr.groupby(['target', 'protocol', 'site', 'model'])[['coverage', 'mean_interval_width', 'interval_score']].mean().reset_index()
c2['Target'] = c2.target.map(TN); c2['Protocol'] = c2.protocol.map(PN) + c2.site.map(lambda x: (', ' + x) if x else ''); c2['Model'] = c2.model.map(MN)
c2['cov'] = c2.coverage.map(f3); c2['w'] = c2.mean_interval_width.map(f2); c2['is'] = c2.interval_score.map(f2)
c2 = c2.assign(o=c2.protocol.map(order)).sort_values(['target', 'o', 'site', 'model'])
table(c2, ['Target', 'Protocol', 'Model', 'cov', 'w', 'is'], 'Target & Protocol & Model & Coverage & Mean width & Interval score',
      'Conformal intervals at nominal coverage 0.90, mean over five seeds.', 'tab:s-cov', 'lllrrr', long=True, size=r'\footnotesize')
cc = pd.read_csv(R / 'metrics/conditional_coverage.csv'); cc = cc[(cc.model == 'catboost') & (cc.dimension == 'user_history_bin')]
cc['Target'] = cc.target.map(TN); cc['Protocol'] = cc.protocol.map(PN); cc['cov'] = cc.coverage.map(f3); cc['n'] = cc.n.map(lambda v: f'{int(v):,}')
cc = cc.pivot_table(index=['Target', 'Protocol'], columns='group', values='cov', aggfunc='first').reset_index()
hc = [c for c in ['0', '1-2', '3-9', '10+'] if c in cc.columns]
table(cc, ['Target', 'Protocol'] + hc, 'Target & Protocol & ' + ' & '.join(hc),
      'Coverage of CatBoost intervals by the number of the driver\\textquotesingle s completed sessions (nominal 0.90).', 'tab:s-condcov', 'll' + 'r' * len(hc))
cb = pd.read_csv(R / 'metrics/calibration_budget.csv')
cb = cb.groupby(['target', 'target_site', 'budget'])[['coverage', 'mean_interval_width', 'interval_score']].mean().reset_index()
cb['Target'] = cb.target.map(TN); cb['Site'] = cb.target_site.map(SN); cb['m'] = cb.budget.astype(int)
cb['cov'] = cb.coverage.map(f3); cb['w'] = cb.mean_interval_width.map(f2); cb['is'] = cb.interval_score.map(f2)
table(cb, ['Target', 'Site', 'm', 'cov', 'w', 'is'], r'Target & Target site & $m$ & Coverage & Mean width & Interval score',
      r'Recalibration of CatBoost intervals with the first $m$ target-site sessions ($m=0$: source-site validation residuals).', 'tab:s-budget', 'llrrrr', long=True, size=r'\footnotesize')

# ===================================================================== S6 occupancy
add(r'\section{Station occupancy}')
oc = pd.read_csv(R / 'metrics/occupancy_metrics.csv')
oc['site'] = oc.target_site.map(lambda x: SN.get(x, '') if isinstance(x, str) else '')
o2 = oc.groupby(['protocol', 'site', 'horizon'])[['auroc', 'brier', 'recommendation_hit_rate', 'confidently_wrong_rate', 'n_queries']].mean().reset_index()
o2['Protocol'] = o2.protocol.map(PN) + o2.site.map(lambda x: (', ' + x) if x else '')
for c in ['auroc', 'brier', 'recommendation_hit_rate', 'confidently_wrong_rate']: o2[c] = o2[c].map(f3)
o2['n'] = o2.n_queries.map(lambda v: f'{v:,.0f}')
table(o2, ['Protocol', 'horizon', 'auroc', 'brier', 'recommendation_hit_rate', 'confidently_wrong_rate', 'n'],
      'Protocol & Horizon (h) & AUROC & Brier & Hit rate & Confidently wrong & Congested queries',
      'Occupancy prediction and congested-query recommendations by protocol, target site and horizon, mean over five seeds.', 'tab:s-occ', 'lrrrrrr', size=r'\footnotesize')

# ===================================================================== S7 operational
add(r'\section{Operational analyses}')
add(r'Detail for Sections~' + ref('sec:operational') + r' and~' + ref('sec:res-rq4') + r': the driver request at matched reliability, site-level capacity and target-site bias correction.')
rm = pd.read_csv(R / 'operational/request_matched.csv')
rm['Learner'] = rm.learner.map(MN); rm['Setting'] = rm.setting.map({'A': 'Driver seen', 'C': 'Unseen, history', 'D': 'Cold start', 'request': 'Driver request'})
rm['f'] = [f'{a_:.1f} [{b_:.1f}, {c_:.1f}]' if not pd.isna(b_) else f1(a_) for a_, b_, c_ in zip(rm.flex_vs_oracle_pct, rm.ci_low, rm.ci_high)]
rm['u'] = rm.unmet_target_pct.map(lambda v: f'{v:.0f}')
rp = rm.pivot_table(index=['Learner', 'group', 'Setting'], columns='u', values='f', aggfunc='first').reset_index()
rp['Group'] = rp.group.map({'all': 'All sessions', 'first visit': 'First visit', '1-4 sessions': '1--4 sessions', '5+ sessions': '5+ sessions'})
rp['_g'] = rp.group.map({'all': 0, 'first visit': 1, '1-4 sessions': 2, '5+ sessions': 3})
rp['_s'] = rp.Setting.map({'Driver seen': 0, 'Unseen, history': 1, 'Cold start': 2, 'Driver request': 3})
rp = rp.sort_values(['Learner', '_g', '_s'])
uc = [c for c in ['1', '2', '5'] if c in rp.columns]
table(rp, ['Learner', 'Group', 'Setting'] + uc, r'Learner & Driver history & Forecast & 1\% & 2\% & 5\%',
      r'Flexibility retained (\% of oracle) at undelivered-energy shares of 1, 2 and 5\%, on the sessions where a driver request exists, with 95\% driver-cluster intervals.',
      'tab:s-request', 'lll' + 'r' * len(uc), long=True, size=r'\footnotesize')
cn = pd.read_csv(R / 'operational/site_capacity_needed.csv')
cell = {(r.site, r.policy, r.undelivered_target_pct): (r.capacity_needed_fraction, r.ci_low, r.ci_high) for r in cn.itertuples()}
cols = [('caltech', 1.0), ('caltech', 2.0), ('jpl', 1.0), ('jpl', 2.0)]
def _n(x):
    x = 0.0 if abs(x) < 5e-4 else x                      # no negative zero
    return f'{x:.3f}'.replace('-', '$-$')
fmt = lambda v: '' if v is None else (f'{_n(v[0])} [{_n(v[1])}, {_n(v[2])}]' if not pd.isna(v[1]) else _n(v[0]))
pols = [('oracle', 'Perfect information'), ('A', 'Forecast, driver seen'), ('C', 'Forecast, unseen driver with history'),
        ('D', 'Forecast, cold start'), ('request (D otherwise)', 'Driver request, else cold start'), ('first come first served', 'Arrival order, no forecast')]
diffs = [('D minus A', 'Cold start $-$ driver seen'), ('D minus oracle', 'Cold start $-$ perfect information'),
         ('request (D otherwise) minus D', 'Request $-$ cold start'), ('first come first served minus A', 'Arrival order $-$ driver seen')]
add(r'\begin{table}[H]\centering\scriptsize')
add(r'\caption{Site capacity needed to keep undelivered energy at or below 1\,\% or 2\,\% of the achievable energy, as a share of the peak load if every session '
    r'charged immediately (Section~' + ref('sec:operational') + r'). Point estimates with 95\,\% intervals from a day-block bootstrap ($B=1000$). The bottom rows give the share $\kappa$ of the capacity '
    r'value of perfect information captured by each forecast (Eq.~(' + ref('eq:capture') + r')), relative to serving vehicles in order of arrival.}\label{tab:s-site}')
add(r'\begin{tabular}{lcccc}\toprule')
add(r' & \multicolumn{2}{c}{Caltech} & \multicolumn{2}{c}{JPL} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}')
add(r'Policy & $\le$1\,\% & $\le$2\,\% & $\le$1\,\% & $\le$2\,\% \\ \midrule')
add(r'\multicolumn{5}{l}{\emph{Capacity needed}} \\')
for k, lab in pols:
    add(lab + ' & ' + ' & '.join(fmt(cell.get((site, k, u))) for site, u in cols) + r' \\')
add(r'\midrule\multicolumn{5}{l}{\emph{Difference in capacity needed}} \\')
for k, lab in diffs:
    add(lab + ' & ' + ' & '.join(fmt(cell.get((site, k, u))) for site, u in cols) + r' \\')
add(r'\midrule\multicolumn{5}{l}{\emph{Share of the value of perfect information captured, $\kappa$ (\%)}} \\')
for k, lab in pols[1:5]:
    vals = []
    for site, u in cols:
        f = lambda q: cell[(site, q, u)][0]
        span = f('first come first served') - f('oracle')
        vals.append(f"{100 * (f('first come first served') - f(k)) / span:.0f}".replace('-', '$-$'))
    add(lab + ' & ' + ' & '.join(vals) + r' \\')
add(r'\bottomrule\end{tabular}\end{table}')
add('')
bc = pd.read_csv(R / 'operational/bias_correction.csv'); bc = bc[bc.model.isin(['catboost', 'torch_ft'])]
bc['Target'] = bc.target.map(TN); bc['Site'] = bc.target_site.map(SN); bc['Model'] = bc.model.map(MN)
bp = bc.pivot_table(index=['Target', 'Site', 'Model'], columns='budget', values='mae').reset_index()
bcols = [c for c in [0, 10, 25, 50, 100, 250] if c in bp.columns]
for c in bcols: bp[c] = bp[c].map(f3)
bp.columns = [str(c) for c in bp.columns]
table(bp, ['Target', 'Site', 'Model'] + [str(c) for c in bcols], 'Target & Site & Model & ' + ' & '.join(f'$m={c}$' for c in bcols),
      'Unseen-site MAE after shifting forecasts by the median residual of the first $m$ target-site sessions, evaluated on common later sessions.', 'tab:s-bias', 'lll' + 'r' * len(bcols), size=r'\footnotesize')

# ===================================================================== S8 pandemic
add(r'\section{Pandemic period}')
pdl = pd.read_csv(R / 'metrics/pandemic_pre_post_delta.csv'); pdl = pdl[pdl.model.isin(['catboost', 'torch_ft'])]
pdl['Target'] = pdl.target.map(TN); pdl['Protocol'] = pdl.protocol.map(PN) + pdl.split_name.map(lambda x: (', ' + site_of(x)) if site_of(x) else '')
pdl['Model'] = pdl.model.map(MN)
for c in ['pre_break', 'post_break']: pdl[c] = pdl[c].map(f3)
pdl['chg'] = pdl.mae_relative_change_pct.map(f1)
table(pdl, ['Target', 'Protocol', 'Model', 'pre_break', 'post_break', 'chg'], r'Target & Protocol & Model & Before 1 March 2020 & After & Change (\%)',
      'MAE of test sessions before and after 1 March 2020 within each protocol (descriptive; test periods differ between protocols).', 'tab:s-pandemic', 'lllrrr', long=True, size=r'\footnotesize')

# ===================================================================== S9 reproducibility
add(r'\section{Tuned hyperparameters}')
hp = json.load(open(R / 'metrics/tuned_hyperparameters.json'))
rows = []
for key, v in hp.items():
    t, sp = key.split('|')
    c = v.get('catboost', {}); f = v.get('torch_ft', {})
    rows.append({'t': TN[t], 'sp': sp.replace('site_disjoint:', 'Unseen site, ').replace('_', ' '),
                 'cb': f"{c.get('iterations', '')}, {c.get('depth', '')}, {c.get('learning_rate', np.nan):.3f}, {c.get('l2_leaf_reg', np.nan):.1f}, {c.get('random_strength', np.nan):.2f}",
                 'ft': f"{f.get('d_token', '')}, {f.get('n_heads', '')}, {f.get('n_layers', '')}, {f.get('dropout', np.nan):.3f}, {f.get('lr', np.nan):.1e}",
                 'rd': f"{v.get('ridge', {}).get('alpha', np.nan):.2f}"})
table(pd.DataFrame(rows), ['t', 'sp', 'cb', 'ft', 'rd'],
      r'Target & Split & CatBoost (iterations, depth, learning rate, $\ell_2$, random strength) & FT-Transformer (width, heads, layers, dropout, learning rate) & Ridge $\alpha$',
      'Hyperparameters selected by eight Optuna trials per learner with seed 11 and frozen for all seeds.', 'tab:s-hp', 'llp{4.6cm}p{4.6cm}r', long=True, size=r'\scriptsize')

# ===================================================================== SI figures
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'], 'font.size': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
C = {'blue': '#0072B2', 'sky': '#56B4E9', 'verm': '#D55E00', 'orange': '#E69F00', 'grey': '#8C959F', 'black': '#1F2933'}
MM = 1 / 25.4
tag = lambda ax, l: ax.text(-0.16, 1.07, l, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
w = pd.read_csv(R / 'paper/coldstart_warmup.csv'); tr = pd.read_csv(R / 'paper/impact_tradeoff.csv')
_PS = pd.ExcelFile(PAX).sheet_names
def PAS(name):
    hits = [x for x in _PS if x == name or name.startswith(x) or x.startswith(name)]
    assert len(hits) == 1, (name, hits)
    return pd.read_excel(PAX, sheet_name=hits[0])
pa_w = PAS('Cold-start warm-up curve'); pa_tr = PAS('Smart charging trade-off'); pa_dc = PAS('Cold-start decomposition')
fig, axs = plt.subplots(2, 3, figsize=(190 * MM, 118 * MM), gridspec_kw={'wspace': 0.45, 'hspace': 0.75})
order = ['0', '1', '2', '3-4', '5-9', '10-19', '20+']
for row, (ww, tt, net) in enumerate([(w, tr, 'ACN workplace sites'), (pa_w, pa_tr, 'Palo Alto public network')]):
    for ax, (t, unit, title) in zip(axs[row, :2], [('kwh_delivered', 'kWh', 'Energy: warm-up'), ('duration_hours', 'h', 'Duration: warm-up')]):
        z = ww[(ww.target == t) & (ww.learner == 'ft')].copy(); z['k'] = z.k_bucket.astype(str); z = z.set_index('k').loc[order]; x = np.arange(7)
        ax.plot(x, z.mae_random, '--o', color=C['grey'], ms=3, label='Random split'); ax.plot(x, z.mae_warm, '-o', color=C['blue'], ms=3, label='Unseen, history')
        ax.plot(x, z.mae_cold, '-s', color=C['verm'], ms=3, label='Cold start'); ax.set_xticks(x); ax.set_xticklabels(order, fontsize=5.8)
        ax.set_xlabel("Driver's completed sessions before this one"); ax.set_ylabel(f'MAE ({unit})'); ax.set_title(title, loc='left', fontsize=8)
    z = tt[tt.learner == 'ft']
    for st, lab, col in [('A', 'Driver seen', C['blue']), ('C', 'Unseen, history', C['sky']), ('D', 'Cold start', C['verm'])]:
        q = z[z.setting == st].sort_values('unmet_share_pct'); axs[row, 2].plot(q.unmet_share_pct, q.flex_vs_oracle_pct, '-o', ms=2.6, lw=1.0, color=col, label=lab, markevery=max(1, len(q) // 10))
    axs[row, 2].set_xlim(0, 30); axs[row, 2].set_ylim(0, 102); axs[row, 2].axvline(2, color=C['grey'], lw=0.6, ls=':')
    axs[row, 2].set_xlabel('Undelivered energy (% of achievable)'); axs[row, 2].set_ylabel('Flexibility retained (% of oracle)')
    axs[row, 2].set_title('Smart charging', loc='left', fontsize=8)
    for ax_ in axs[row]:
        ax_.annotate(net + ', FT-Transformer', xy=(0, 1), xycoords='axes fraction', xytext=(0, 17), textcoords='offset points', fontsize=6.8, color=C['grey'])
for ax_, l_ in zip(axs.ravel(), 'abcdef'):
    ax_.annotate(l_, xy=(-0.16, 1), xycoords='axes fraction', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight='bold', va='bottom')
h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.03), fontsize=6.8)
axs[0, 2].legend(frameon=False, fontsize=6.3, loc='lower right')
fig.savefig(OUT / 'si_ft.pdf', bbox_inches='tight'); plt.close(fig)
pa_nft = int(pa_w[(pa_w.learner == 'ft') & (pa_w.target == 'kwh_delivered')].n_rows.sum()); pa_nd = int(pa_dc.n_drivers.iloc[0])
sl = pd.read_csv(R / 'operational/site_level.csv')
fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM), gridspec_kw={'wspace': 0.25})
pols = [('oracle', 'Perfect information', C['black'], '-'), ('A', 'Driver seen', C['blue'], '-'), ('C', 'Unseen, history', C['sky'], '-'),
        ('D', 'Cold start', C['verm'], '-'), ('request (D otherwise)', 'Request, else cold start', C['orange'], '--'), ('first come first served', 'Arrival order', C['grey'], '--')]
for ax, site in zip(axs, ['caltech', 'jpl']):
    for p_, lab, col, ls in pols:
        q = sl[(sl.site == site) & (sl.policy == p_)].sort_values('cap_fraction')
        ax.plot(100 * q.cap_fraction, q.undelivered_share_pct.clip(lower=1e-3), ls, color=col, label=lab, lw=1.1)
    ax.set_yscale('log'); ax.set_ylim(0.01, 60); ax.axhline(1, color=C['grey'], lw=0.6, ls=':')
    ax.set_xlabel('Site capacity limit (% of uncontrolled peak)'); ax.set_ylabel('Undelivered energy (%)'); ax.set_title(SN[site] + ': shared capacity limit', loc='left', fontsize=8)
axs[0].legend(frameon=False, fontsize=6.5)
for ax_, l_ in zip(axs, 'ab'): tag(ax_, l_)
fig.savefig(OUT / 'si_site.pdf', bbox_inches='tight'); plt.close(fig)
add(r'\section{Supplementary figures}')
sc = sl.groupby('site')[['n_controlled', 'n_uncontrolled', 'uncontrolled_peak_kw']].first()
unc = lambda st: 100 * sc.loc[st, 'n_uncontrolled'] / (sc.loc[st, 'n_controlled'] + sc.loc[st, 'n_uncontrolled'])
nft = int(w[(w.learner == 'ft') & (w.target == 'kwh_delivered')].n_rows.sum())
add(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{figs/si_ft.pdf}\caption{Cold-start dynamics and smart-charging trade-off for the FT-Transformer, '
    r'the counterparts of the CatBoost warm-up curves (Fig.~' + ref('fig:coldstart') + r' (e, f) and Table~\ref{tab:s-warm}) and of Fig.~' + ref('fig:operation') + r' (a, b). (a) Energy and (b) duration MAE as a function of the number of the driver\textquotesingle s completed sessions before the '
    r'predicted one, for the random split (driver seen), an unseen driver with own history and a cold start, over all ' + f'{nft:,}' + r' sessions of the 549 drivers in the '
    r'cold-start cohort of the ACN sites (means over five seeds). (c) Undelivered energy against flexibility retained under as-late-as-possible charging at 6.6\,kW, traced by shifting the '
    r'departure and energy forecasts by cross-conformal residual quantiles from $\gamma=0.5$ (the raw point forecast) to $\gamma=0.01$; the dotted line marks 2\,\% undelivered energy. '
    r'(d--f) The same for the Palo Alto public network, over all ' + f'{pa_nft:,}' + r' sessions of the ' + f'{pa_nd:,}' + r' drivers in its cold-start cohort. '
    r'On both networks the FT-Transformer reproduces the pattern found for CatBoost.}\label{fig:s-ft}\end{figure}')
add(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{figs/si_site.pdf}\caption{Site-level undelivered energy for all six scheduling policies at (a) Caltech '
    r'and (b) JPL, sessions before March 2020. Undelivered energy, as a share of the energy achievable during each stay and on a logarithmic scale, is plotted against the site '
    r'capacity limit, expressed as a share of the peak load if every session charged immediately (' + f"{sc.loc['caltech', 'uncontrolled_peak_kw']:.0f}" + r'\,kW at Caltech, '
    + f"{sc.loc['jpl', 'uncontrolled_peak_kw']:.0f}" + r'\,kW at JPL). Sessions of drivers in the cold-start cohort are served by least laxity (Eq.~(' + ref('eq:laxity') + r')) '
    r'using perfect information, seed-averaged CatBoost forecasts for a seen driver, an unseen driver with history or a cold start, or the driver request where available '
    r'with the cold-start forecast otherwise; the last policy serves them in order of arrival without any forecast. All other sessions (' + f"{unc('caltech'):.0f}" + r'\,\% at '
    r'Caltech, ' + f"{unc('jpl'):.0f}" + r'\,\% at JPL) charge immediately and share the limit. The dotted line marks 1\,\% undelivered energy; Table~\ref{tab:s-site} gives the capacity '
    r'needed at 1\,\% and 2\,\%.}\label{fig:s-site}\end{figure}')
Path('si_body.tex').write_text('\n'.join(L))
print('SI body written:', sum(1 for x in L if 'caption{' in x), 'captions')
