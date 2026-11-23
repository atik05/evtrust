"""Robustness of the cold-start decomposition.

Order of the decomposition. The paper's path is A -> B -> C -> D (leakage, unseen driver, history).
    The alternative path hides history first: A -> A' -> B' -> D, where A' and B' are the A and B models
    with the driver's history hidden at prediction (no new training settings). Each factor's share is
    reported on both paths and averaged over the two orders (a two-order Shapley value; a full Shapley
    value over three factors is undefined because an unseen driver cannot have leaked sessions).
Driver identity. A target-encoded driver identifier (smoothed mean target of the driver in training,
    out of fold within training) is added; an unseen driver gets the training mean. Shows how much larger the random-split
    optimism becomes with identity, while the cold-start level is unchanged.
Training size. The random-split model is retrained on 75 % and 50 % of its training rows: if accuracy
    barely moves, the unseen-driver gap is not an artefact of training-set size.
Optimism ladder: the top rung is a literature-style random split that adds a driver identifier
(target encoding), driver aggregates over the whole file (including the session itself and the future) and
the charging duration known only after plug-in; the paper's setting E (whole-file aggregates) is the rung below.
All on identical sessions; driver-cluster bootstrap; seeds averaged per session as in the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from evtrust.coldstart import published_style

from .common import TARGETS, Ctx, cluster_ratio, mask_history


def _folds(d, seed, n=5):
    groups = d['user_id'].astype(str).to_numpy()
    uniq = np.unique(groups)
    perm = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq))))
    gid = np.array([perm[g] for g in groups])
    return list(KFold(n, shuffle=True, random_state=seed).split(d)), list(GroupKFold(n).split(d, groups=gid))


def _base(d, target, learner, seed):
    return pd.DataFrame({'session_key': d['session_id'].astype(str).to_numpy(), 'user': d['user_id'].astype(str).to_numpy(),
                         'session_index': np.arange(len(d)), 'k': d['k'].to_numpy(), 'is_T': d['is_T'].to_numpy(),
                         'y': d[target].to_numpy(float), 'learner': learner, 'target': target, 'seed': seed})


# ----------------------------------------------------------------------------------------------- 2c
def alt_unit(ctx: Ctx, learner, target, seed):
    path = ctx.p('shards', f'alt_{learner}_{target}_{seed}.parquet')
    if path.exists():
        return pd.read_parquet(path)
    d = ctx.cohort(target)
    cols = ctx.numeric + ['station_code']
    X, Xm, y, T = d[cols], mask_history(d[cols]), d[target].to_numpy(float), d['is_T'].to_numpy()
    fit = ctx.fitter(learner, seed)
    kf, _ = _folds(d, seed)
    pA, pAm = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for tr, te in kf:
        m = fit(X.iloc[tr], y[tr])
        pA[te], pAm[te] = m.predict(X.iloc[te]), m.predict(Xm.iloc[te])
    pB, pBm = np.full(len(d), np.nan), np.full(len(d), np.nan)
    m = fit(X[~T], y[~T]); pB[T], pBm[T] = m.predict(X[T]), m.predict(Xm[T])
    out = _base(d, target, learner, seed)
    out['pred_A_rerun'], out['pred_Ap'], out['pred_B_rerun'], out['pred_Bp'] = pA, pAm, pB, pBm
    out.to_parquet(path, index=False)
    return out


def _paired(ctx, learner, target, extra_prefix):
    """Merge new-unit predictions with the manuscript's shard (A to E) on identical sessions; seed-average.

    extra_prefix may be a list: the predictions of several unit types (e.g. idsize, alt, csb) are merged, so
    that refitted variants can be compared with references refitted in the same run."""
    import glob
    sh = ctx.shards(learner, target)
    z = sh
    for pre in ([extra_prefix] if isinstance(extra_prefix, str) else extra_prefix):
        files = sorted(glob.glob(str(ctx.p('shards') / f'{pre}_{learner}_{target}_*.parquet')))
        if not files:
            continue
        new = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        keep = ['seed', 'session_key'] + [c for c in new.columns if c.startswith('pred_') and c not in z.columns]
        z = z.merge(new[keep], on=['seed', 'session_key'], how='inner')
    for c in [c for c in z.columns if c.startswith('pred_')]:
        z['err_' + c[5:]] = np.abs(z['y'] - z[c])
    errs = [c for c in z.columns if c.startswith('err_')]
    return z.groupby(['user', 'session_index', 'k', 'is_T'], as_index=False)[errs].mean(), z['seed'].nunique()


def _ratios(T, pairs, B):
    rows = []
    for a, b, comp in pairs:
        if a not in T or b not in T:
            continue
        z = T.dropna(subset=[a, b])
        if len(z) == 0:
            continue
        est, lo, hi, p = cluster_ratio(z[a].to_numpy(), z[b].to_numpy(), z['user'].to_numpy(), B)
        rows.append({'component': comp, 'from': a[4:], 'to': b[4:], 'effect_pct': est, 'ci_low': lo, 'ci_high': hi,
                     'p_value': p, 'log_ratio': np.log(z[b].sum() / z[a].sum()), 'n_rows': len(z), 'n_drivers': z['user'].nunique()})
    return pd.DataFrame(rows)


def summarise_alt(ctx: Ctx, learner):
    out = []
    for target in TARGETS:
        m, ns = _paired(ctx, learner, target, 'alt')
        T = m[m['is_T']]
        pairs = [('err_A', 'err_B', 'paper path: leakage (A->B)'), ('err_B', 'err_C', 'paper path: unseen driver (B->C)'),
                 ('err_C', 'err_D', 'paper path: history (C->D)'), ('err_A', 'err_D', 'total (A->D)'),
                 ('err_A_rerun', 'err_Ap', 'alternative path: history (A->A\')'), ('err_Ap', 'err_Bp', 'alternative path: leakage (A\'->B\')'),
                 ('err_Bp', 'err_D', 'alternative path: unseen driver (B\'->D)'), ('err_A_rerun', 'err_D', 'alternative path total (A->D)'),
                 ('err_A', 'err_A_rerun', 'check: rerun A vs manuscript A'), ('err_B', 'err_B_rerun', 'check: rerun B vs manuscript B')]
        r = _ratios(T, pairs, ctx.n_boot)
        lr = dict(zip(r['component'], r['log_ratio']))
        tot, tot_alt = lr['total (A->D)'], lr['alternative path total (A->D)']
        r['log_share_pct'] = [100 * v / (tot_alt if c.startswith('alternative') else tot) for c, v in zip(r['component'], r['log_ratio'])]
        sh = {'history': (lr['paper path: history (C->D)'] / tot, lr["alternative path: history (A->A')"] / tot_alt),
              'leakage': (lr['paper path: leakage (A->B)'] / tot, lr["alternative path: leakage (A'->B')"] / tot_alt),
              'unseen driver': (lr['paper path: unseen driver (B->C)'] / tot, lr["alternative path: unseen driver (B'->D)"] / tot_alt)}
        # bootstrap of the two-order shares
        rng = np.random.default_rng(7)
        users, uid = np.unique(T['user'].to_numpy(), return_inverse=True)
        sums = {c: np.bincount(uid, weights=T[c].fillna(0).to_numpy()) for c in ['err_A', 'err_B', 'err_C', 'err_D', 'err_A_rerun', 'err_Ap', 'err_Bp']}
        draws = {k: [] for k in sh}
        for _ in range(ctx.n_boot):
            w = np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users))
            s = {c: v @ w for c, v in sums.items()}
            L = lambda a, b: np.log(s[b] / s[a])
            t1, t2 = L('err_A', 'err_D'), L('err_A_rerun', 'err_D')
            draws['history'].append(50 * (L('err_C', 'err_D') / t1 + L('err_A_rerun', 'err_Ap') / t2))
            draws['leakage'].append(50 * (L('err_A', 'err_B') / t1 + L('err_Ap', 'err_Bp') / t2))
            draws['unseen driver'].append(50 * (L('err_B', 'err_C') / t1 + L('err_Bp', 'err_D') / t2))
        for k, (p1, p2) in sh.items():
            x = np.array(draws[k])
            r = pd.concat([r, pd.DataFrame([{'component': f'two-order share: {k}', 'effect_pct': np.nan, 'log_share_pct': 50 * (p1 + p2),
                                             'ci_low': np.percentile(x, 2.5), 'ci_high': np.percentile(x, 97.5),
                                             'n_rows': len(T), 'n_drivers': len(users)}])], ignore_index=True)
        r.insert(0, 'target', target); r.insert(1, 'learner', learner); r['n_seeds'] = ns
        out.append(r)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(ctx.p(f'decomp_alternative_path_{learner}.csv'), index=False)
    ctx.log('2c ' + learner + ': ' + '; '.join(f"{t} {c}: {v:.1f}" for t, c, v in zip(res['target'], res['component'], res['log_share_pct'])
                                               if c.startswith('two-order')))


# ------------------------------------------------------------------------------------------- identity, size
def target_encode(users_tr, y_tr, users_q, seed, m=5.0, folds=5, unseen='nan'):
    """Smoothed driver mean of the target: out of fold for training rows, full training for query rows.

    unseen='mean' gives a driver without training sessions the training mean, the value the smoothed encoding
    takes for a driver with no sessions; it is used where every query driver is unseen (settings C and D).
    """
    gm = float(np.mean(y_tr))
    def enc(u_fit, y_fit, u_apply):
        s = pd.DataFrame({'u': u_fit, 'y': y_fit}).groupby('u')['y'].agg(['sum', 'count'])
        te = (s['sum'] + m * gm) / (s['count'] + m)
        v = pd.Series(u_apply).map(te)
        return (v.fillna(gm) if unseen == 'mean' else v).to_numpy(float)
    te_tr = np.full(len(y_tr), np.nan)
    for a, b in KFold(folds, shuffle=True, random_state=seed).split(y_tr):
        te_tr[b] = enc(users_tr[a], y_tr[a], users_tr[b])
    return te_tr, enc(users_tr, y_tr, users_q)


def id_unit(ctx: Ctx, target, seed):
    learner = ctx.cfg['driver_id']['learner']
    path = ctx.p('shards', f'idsize_{learner}_{target}_{seed}.parquet')
    if path.exists():
        return pd.read_parquet(path)
    d = ctx.cohort(target)
    cols = ctx.numeric + ['station_code']
    y, T = d[target].to_numpy(float), d['is_T'].to_numpy()
    users = d['user_id'].astype(str).to_numpy()
    num_id = ctx.numeric + ['driver_te']
    fit_id = ctx.fitter(learner, seed, numeric=num_id)
    fit = ctx.fitter(learner, seed)
    kf, gf = _folds(d, seed)
    out = _base(d, target, learner, seed)
    pred = {k: np.full(len(d), np.nan) for k in ['A_id', 'B_id', 'C_id', 'D_id', 'A_lit'] + [f'A_size{int(100 * f)}' for f in ctx.cfg['driver_id']['size_fractions']]}
    num_lit = ctx.numeric + ['driver_te', 'charging_duration_hours']
    fit_lit = ctx.fitter(learner, seed, numeric=num_lit)
    lit = published_style(d)                                   # whole-file driver aggregates (leaky by design)

    def with_te(idx_tr, idx_q, unseen='nan'):
        te_tr, te_q = target_encode(users[idx_tr], y[idx_tr], users[idx_q], seed, unseen=unseen)
        Xtr = d.iloc[idx_tr][cols].copy(); Xtr['driver_te'] = te_tr
        Xq = d.iloc[idx_q][cols].copy(); Xq['driver_te'] = te_q
        return Xtr[num_id + ['station_code']], Xq[num_id + ['station_code']]

    for tr, te in kf:
        Xtr, Xq = with_te(tr, te)
        pred['A_id'][te] = fit_id(Xtr, y[tr]).predict(Xq)
        Ltr = lit.iloc[tr][cols + ['charging_duration_hours']].copy(); Ltr['driver_te'] = Xtr['driver_te'].to_numpy()
        Lq = lit.iloc[te][cols + ['charging_duration_hours']].copy(); Lq['driver_te'] = Xq['driver_te'].to_numpy()
        pred['A_lit'][te] = fit_lit(Ltr[num_lit + ['station_code']], y[tr]).predict(Lq[num_lit + ['station_code']])
        rng = np.random.default_rng(seed + 1)
        for f in ctx.cfg['driver_id']['size_fractions']:
            sub = rng.choice(tr, size=int(round(f * len(tr))), replace=False)
            pred[f'A_size{int(100 * f)}'][te] = fit(d.iloc[sub][cols], y[sub]).predict(d.iloc[te][cols])
    trT, teT = np.flatnonzero(~T), np.flatnonzero(T)
    Xtr, Xq = with_te(trT, teT)
    pred['B_id'][teT] = fit_id(Xtr, y[trT]).predict(Xq)
    for tr, te in gf:
        Xtr, Xq = with_te(tr, te, unseen='mean')                     # test drivers are unseen: training mean
        mdl = fit_id(Xtr, y[tr])
        pred['C_id'][te] = mdl.predict(Xq)
        pred['D_id'][te] = mdl.predict(mask_history(Xq))
    for k, v in pred.items():
        out[f'pred_{k}'] = v
    out['n_train_A'] = int(np.mean([len(tr) for tr, _ in kf])); out['n_train_B'] = int((~T).sum())
    out['n_train_C'] = int(np.mean([len(tr) for tr, _ in gf]))
    out.to_parquet(path, index=False)
    return out


def summarise_id(ctx: Ctx):
    learner = ctx.cfg['driver_id']['learner']
    res = []
    for target in TARGETS:
        m, ns = _paired(ctx, learner, target, ['idsize', 'alt', 'csb'])
        T = m[m['is_T']]
        # refitted variants are compared with references refitted in this run (same code, settings and GPU),
        # so that run-to-run differences cannot pass for effects; the manuscript's A to E are kept for the paper path
        Ar = 'err_A_rerun' if 'err_A_rerun' in T else 'err_A'
        Dr = 'err_D_rerun' if 'err_D_rerun' in T else 'err_D'
        pairs = [('err_A', 'err_B', 'without identity: leakage (A->B)'), ('err_B', 'err_C', 'without identity: unseen driver (B->C)'),
                 ('err_C', 'err_D', 'without identity: history (C->D)'), ('err_A', 'err_D', 'without identity: total (A->D)'),
                 ('err_A_id', 'err_B_id', 'with identity: leakage'), ('err_B_id', 'err_C_id', 'with identity: unseen driver'),
                 ('err_C_id', 'err_D_id', 'with identity: history'), ('err_A_id', 'err_D_id', 'with identity: total'),
                 (Ar, 'err_A_id', 'identity effect on random split (A)'), (Dr, 'err_D_id', 'identity effect on cold start (D)')]
        pairs += [(Ar, f'err_A_size{int(100 * f)}', f'training size {int(100 * f)} % of rows (A)') for f in ctx.cfg['driver_id']['size_fractions']]
        pairs += [('err_A', 'err_E', 'ladder: + whole-file driver aggregates (setting E)'),
                  (Ar, 'err_A_lit', 'ladder: literature-style random split (identity + whole-file aggregates + charging duration)'),
                  ('err_A_lit', Dr, 'cold-start gap against the literature-style random split'),
                  ('err_E', 'err_D', 'cold-start gap against setting E'),
                  ('err_A_id', Dr, 'cold-start gap against random split with identity')]
        if Ar != 'err_A':
            pairs.append(('err_A', Ar, 'check: random split refitted in this run vs manuscript'))
        if Dr != 'err_D':
            pairs.append(('err_D', Dr, 'check: cold start refitted in this run vs manuscript'))
        r = _ratios(T, pairs, ctx.n_boot)
        r['mae_from'] = [T[f'err_{a}'].mean() for a in r['from']]; r['mae_to'] = [T[f'err_{b}'].mean() for b in r['to']]
        r.insert(0, 'target', target); r['n_seeds'] = ns
        res.append(r)
    out = pd.concat(res, ignore_index=True)
    import glob
    n = pd.read_parquet(sorted(glob.glob(str(ctx.p('shards') / 'idsize_*.parquet')))[0])[['n_train_A', 'n_train_B', 'n_train_C']].iloc[0]
    for k, v in n.items():
        out[k] = int(v)
    out.to_csv(ctx.p('decomp_driver_identity_and_size.csv'), index=False)
    ctx.log('2d/2e: ' + '; '.join(f"{t} {c}: {v:+.1f}" for t, c, v in zip(out['target'], out['component'], out['effect_pct'])))


def run_alt(ctx: Ctx):
    for learner in ctx.cfg['decomp']['learners']:
        for target in TARGETS:
            for seed in ctx.seeds:
                alt_unit(ctx, learner, target, seed)
                ctx.log(f'2c {learner} {target} seed {seed} done')
        summarise_alt(ctx, learner)


def run_id(ctx: Ctx):
    for target in TARGETS:
        for seed in ctx.seeds:
            id_unit(ctx, target, seed)
            ctx.log(f'2d/2e {target} seed {seed} done')
    summarise_id(ctx)
