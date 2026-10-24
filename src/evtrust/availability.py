"""How much reported accuracy depends on information that only exists after plug-in.

Evaluated with 5-fold cross-fitting repeated once per seed (5 x 5 with five seeds).

Feature sets, same learner, same rows, random split:
  plug_in            information available when the vehicle is plugged in
  plus_charging      + charging duration (observed when charging ends)
  plus_departure     + charging duration + session duration (observed at departure)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .coldstart import cluster_ratio_ci, make_learner

POST_CHARGING = ['charging_duration_hours', 'charging_duration_log']
POST_DEPARTURE = POST_CHARGING + ['duration_hours']


def run(df: pd.DataFrame, numeric: list[str], learner: str, seeds: list[int], cfg: dict,
        sites: list[str] | None = None) -> pd.DataFrame:
    d = df[df['kwh_delivered'].notna()].copy().reset_index(drop=True)
    d['station_code'] = d['station_id'].astype('category').cat.codes
    d['charging_duration_log'] = np.log1p(d['charging_duration_hours'].clip(lower=0))
    rows = []
    scopes = [('all', d)] + [(s, d[d['site'] == s].reset_index(drop=True)) for s in (sites or sorted(d['site'].unique()))]
    for scope, z in scopes:
        if len(z) < 500:
            continue
        y = z['kwh_delivered'].to_numpy(float)
        user = z['user_id'].astype(str).fillna('anon').to_numpy()
        for name, extra in [('plug_in', []), ('plus_charging', POST_CHARGING), ('plus_departure', POST_DEPARTURE)]:
            feats = numeric + extra
            X = z[feats + ['station_code']]
            for seed in seeds:                                   # repeats
                fit = make_learner(learner, feats, seed, cfg)
                for tr, te in KFold(5, shuffle=True, random_state=seed).split(z):   # 5 folds, every row predicted once
                    err = np.abs(y[te] - fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
                    rows.append(pd.DataFrame({'scope': scope, 'feature_set': name, 'seed': seed, 'learner': learner,
                                              'user': user[te], 'row': te, 'err': err}))
    return pd.concat(rows, ignore_index=True)


def summarise(rows: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    out = []
    for (scope, learner), g in rows.groupby(['scope', 'learner']):
        piv = g.pivot_table(index=['seed', 'row', 'user'], columns='feature_set', values='err').reset_index()
        per_seed = g.groupby(['feature_set', 'seed'])['err'].mean().unstack(0)
        for fs in ['plug_in', 'plus_charging', 'plus_departure']:
            if fs not in piv:
                continue
            rec = {'scope': scope, 'learner': learner, 'feature_set': fs,
                   'mae': per_seed[fs].mean(), 'mae_sd': per_seed[fs].std(), 'n_seeds': per_seed[fs].count()}
            if fs != 'plug_in':
                z = piv.rename(columns={'plug_in': 'a', fs: 'b'})
                est, lo, hi, p = cluster_ratio_ci(z, 'a', 'b', n_boot)
                rec.update({'change_vs_plug_in_pct': est, 'ci_low': lo, 'ci_high': hi, 'p_value': p})
            out.append(rec)
    return pd.DataFrame(out)
