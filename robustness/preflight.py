"""Preflight: the inputs are the manuscript's, the environment works. Stops the run on any mismatch."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .common import TARGETS, Ctx, CurveSet, curves, stable_group, write_json


def value_of_history(ctx: Ctx, learner='catboost', target='kwh_delivered') -> float:
    r = ctx.shards(learner, target)
    m = r.groupby(['user', 'session_index', 'k', 'is_T'], as_index=False)[['err_C', 'err_D']].mean()
    T = m[m['is_T']]
    return 100 * (T['err_D'].sum() / T['err_C'].sum() - 1)


def flex_point(ctx: Ctx, learner='catboost') -> dict:
    out = {}
    js = ctx.joined(learner, 'ACD')
    for s in 'ACD':
        cs = CurveSet(1)
        for seed, j in js.items():
            D, E = j['duration_true'].to_numpy(float), j['energy_true'].to_numpy(float)
            grp = stable_group(j['user'].astype(str))
            U, EA, F, Fo = curves(D, E, j[f'pred_{s}'].to_numpy(float), j[f'pred_{s}_e'].to_numpy(float), grp, ctx.power)
            cs.add(U, EA, F, Fo, np.zeros(len(D), int))
        out[s] = cs.value()
    return out


def run(ctx: Ctx):
    rep, problems = {'network': ctx.net, 'smoke': ctx.smoke}, []
    df = ctx.sessions()
    need = ['session_id', 'site', 'station_id', 'user_id', 'connection_time', 'disconnect_time', 'done_charging_time',
            'duration_hours', 'kwh_delivered', 'charging_duration_hours'] + ctx.numeric
    miss = [c for c in need if c not in df.columns]
    if miss:
        problems.append(f'sessions.parquet lacks columns {miss}')
    rep['sessions'] = len(df)
    rep['anonymous_share'] = float((df['user_id'].isna() | (df['user_id'].astype(str).str.strip() == '')).mean())
    for learner in ('catboost', 'ft'):
        for target in TARGETS:
            r = ctx.shards(learner, target)
            seeds = sorted(r['seed'].unique())
            rep[f'{learner}_{target}_seeds'] = [int(s) for s in seeds]
            if seeds != sorted(ctx.seeds):
                problems.append(f'{learner} {target}: seeds {seeds} != {ctx.seeds}')
    r = ctx.shards('catboost', 'kwh_delivered')
    rep['rows_per_seed'] = int(len(r) // r['seed'].nunique())
    rep['drivers'] = int(r['user'].nunique())
    # the cohort rebuilt from sessions.parquet must be exactly the shards' sessions, with identical labels
    for target in TARGETS:
        d = ctx.cohort(target)
        sh = ctx.shards('catboost', target)
        sh = sh[sh['seed'] == ctx.seeds[0]].set_index('session_key')
        keys_d, keys_s = set(d['session_id'].astype(str)), set(sh.index.astype(str))
        rep[f'cohort_{target}'] = {'rebuilt': len(keys_d), 'shards': len(keys_s), 'only_rebuilt': len(keys_d - keys_s),
                                   'only_shards': len(keys_s - keys_d)}
        if keys_d != keys_s:
            problems.append(f'cohort for {target} differs from the shards: {rep[f"cohort_{target}"]}')
            continue
        z = d.set_index(d['session_id'].astype(str)).loc[sh.index]
        dy = float(np.nanmax(np.abs(z[target].to_numpy(float) - sh['y'].to_numpy(float))))
        dk = float(np.nanmax(np.abs(z['k'].to_numpy(float) - sh['k'].to_numpy(float))))
        rep[f'max_abs_diff_{target}'] = {'label': dy, 'history_count': dk}
        if dy > 1e-6 or dk > 1e-9:
            problems.append(f'labels or history counts differ for {target}: {dy}, {dk}')
    exp = ctx.ncfg.get('expected', {})
    rep['value_of_history_energy_catboost'] = value_of_history(ctx)
    rep['flex_at_2pct_catboost'] = flex_point(ctx)
    if not ctx.smoke:
        if 'rows_per_seed' in exp and rep['rows_per_seed'] != exp['rows_per_seed']:
            problems.append(f"rows per seed {rep['rows_per_seed']} != manuscript {exp['rows_per_seed']}")
        if 'drivers' in exp and rep['drivers'] != exp['drivers']:
            problems.append(f"drivers {rep['drivers']} != manuscript {exp['drivers']}")
        if 'value_of_history_energy_catboost' in exp and abs(rep['value_of_history_energy_catboost'] - exp['value_of_history_energy_catboost']) > 0.15:
            problems.append(f"value of history {rep['value_of_history_energy_catboost']:.2f} != manuscript {exp['value_of_history_energy_catboost']}")
        for s, v in exp.get('flex_at_2pct_catboost', {}).items():
            if abs(rep['flex_at_2pct_catboost'][s] - v) > 0.05:
                problems.append(f"flexibility {s} {rep['flex_at_2pct_catboost'][s]:.2f} != manuscript {v}")
    # environment
    try:
        import torch
        rep['torch_cuda'] = bool(torch.cuda.is_available())
        rep['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:                                   # noqa: BLE001
        problems.append(f'torch import failed: {e}')
    try:
        from catboost import CatBoostRegressor
        p = ctx.learner_cfg()['catboost']; p = {**p, 'iterations': 5}
        CatBoostRegressor(**p).fit(np.random.rand(200, 3), np.random.rand(200))
        rep['catboost_task_type'] = p.get('task_type')
    except Exception as e:                                   # noqa: BLE001
        problems.append(f'CatBoost ({ctx.learner_cfg()["catboost"].get("task_type")}) failed: {e}')
    if 'acnsim' in ctx.ncfg.get('stages', []):
        try:
            import acnportal, adacharge, cvxpy               # noqa: F401
            rep['cvxpy_solvers'] = cvxpy.installed_solvers()
        except Exception as e:                               # noqa: BLE001
            problems.append(f'ACN-Sim packages missing: {e}')
    rep['cpu_count'] = os.cpu_count()
    rep['problems'] = problems
    write_json(ctx.p('preflight.json'), rep)
    for k, v in rep.items():
        ctx.log(f'preflight {k}: {v}')
    if problems:
        raise SystemExit('PREFLIGHT FAILED:\n  ' + '\n  '.join(problems))
    ctx.log('preflight passed')
