"""Sensitivity of the protocol results to the hyperparameter tuning budget.

The main study tunes CatBoost and the FT-Transformer with eight trials of a TPE search on the canonical seed (11),
once per target and evaluation split, and freezes the selected configuration across the five seeds
(src/evtrust/experiments.py). Here the same search, with the same space, sampler and sampler seed, runs for a larger
budget (40 trials by default). Because the first trials of a seeded TPE search are drawn before any result is seen,
the first eight trials of the longer search are exactly the eight trials of the main study's search, so the eight-trial
and the long-budget selections come from one search and differ only in the budget:

  * config "8 trials"  = the best of the first eight trials (validation MAE), as in the main study
  * config "N trials"  = the best of all N trials

Both configurations are refitted for every seed on that seed's training and validation data and evaluated on its test
set, with the user and station history masked exactly as in the main study (experiments._split_frames). Outputs:

  tuning_budget_runs.csv      test MAE per learner, target, split, seed and budget (and the selected parameters)
  tuning_budget_summary.csv   MAE per protocol (site-disjoint splits weighted equally) under both budgets, its change,
                              the protocol gap against random splitting under both budgets, and whether the better of
                              CatBoost and the FT-Transformer changes with the budget
  tuning_budget_published.csv the re-run eight-trial MAE beside the main study's (a check; CatBoost on the GPU is not
                              bitwise deterministic, so small differences are expected)

Every study is stored in an SQLite file and every refit in its own file, so an interrupted run resumes.

  python scripts/tuning_budget.py --config configs/acn.yaml --sessions <processed sessions.parquet> --out <dir>
         [--trials 40] [--check 8] [--learners catboost torch_ft] [--published <regression_predictions.parquet>]
  python scripts/tuning_budget.py --merge <dir1> <dir2> --out <dir> [--published ...]   (combine per-learner runs)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evtrust.config import load_config  # noqa: E402
from evtrust.experiments import _split_frames, make_splits  # noqa: E402
from evtrust.utils import seed_everything  # noqa: E402

TARGETS = {'duration': 'duration_hours', 'energy': 'kwh_delivered'}
NAMES = {'catboost': 'CatBoost', 'torch_ft': 'FT-Transformer'}


def log(msg: str):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def frames(df, cfg, seed, split_name, ycol):
    """Training, validation and test rows of one split, with history masked and missing targets dropped exactly as in
    experiments.run_regression; None when the main study skips the unit (too few rows)."""
    for sp in make_splits(df, cfg, seed):
        if sp.name == split_name:
            tr, ca, va, te = (x[x[ycol].notna()] for x in _split_frames(df, sp))
            if min(len(tr), len(va), len(te)) < 10 or len(ca) < 5:
                return None
            return tr, va, te
    raise KeyError(split_name)


def study_for(storage: str, name: str, seed: int):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(study_name=name, storage=storage, load_if_exists=True, direction='minimize',
                               sampler=optuna.samplers.TPESampler(seed=seed))


def complete_trials(st):
    import optuna
    return [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]


def best_of(trials):
    """Best trial by validation MAE; ties go to the earlier trial, as in Optuna."""
    return min(trials, key=lambda t: (t.value, t.number))


def fit_predict(learner, tr, va, te, numeric, categorical, params, seed):
    ycol = params.pop('_ycol')
    ytr, yva, yte = tr[ycol].to_numpy(float), va[ycol].to_numpy(float), te[ycol].to_numpy(float)
    if learner == 'catboost':
        from evtrust.models import fit_catboost
        m = fit_catboost(tr, ytr, va, yva, numeric, categorical, params, seed)
    else:
        from evtrust.torch_tabular import fit_torch_ft
        m = fit_torch_ft(tr, ytr, va, yva, numeric, categorical, params, seed)
    pred = np.asarray(m.predict(te), float)
    return float(np.mean(np.abs(yte - pred))), len(yte)


def identity_row(pub: dict, learner: str, target: str, split_name: str, trials, check: int) -> dict:
    """Is the main study's selected configuration one of the first `check` trials of this search, and the one
    selected here? (Trial parameters are drawn before any result is seen, so they must coincide; the selection among
    them can differ only through ties or non-deterministic GPU training.)"""
    sel = pub.get(f'{target}|{split_name}', {}).get(learner)
    first = sorted(trials, key=lambda t: t.number)[:check]
    hit = None
    if sel is not None:
        for t in first:
            if all(k in sel and (sel[k] == v or (isinstance(v, float) and abs(float(sel[k]) - v) <= 1e-9 * max(1.0, abs(v))))
                   for k, v in t.params.items()):
                hit = t.number; break
    return {'learner': learner, 'target': target, 'split': split_name, 'published_available': sel is not None,
            'published_is_trial': hit, 'published_among_first': hit is not None,
            'selected_here': best_of(first).number, 'same_selection': hit is not None and hit == best_of(first).number}


def protocol_of(split_name: str) -> str:
    return split_name.split(':')[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'acn.yaml'))
    ap.add_argument('--sessions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--check', type=int, default=8, help='budget of the main study')
    ap.add_argument('--learners', nargs='+', default=['catboost', 'torch_ft'])
    ap.add_argument('--targets', nargs='+', default=['energy', 'duration'])
    ap.add_argument('--seeds', nargs='+', type=int, default=None, help='evaluation seeds (default: the config five_seeds)')
    ap.add_argument('--published', default=None, help='regression_predictions.parquet of the main study (optional check)')
    ap.add_argument('--published-params', default=None,
                    help="tuned_hyperparameters.json of the main study: checks that its selections are among the first trials")
    ap.add_argument('--smoke', action='store_true', help='tiny budget for testing the pipeline')
    ap.add_argument('--merge', nargs='+', default=None,
                    help='only combine the tuning_budget_runs.csv of these output folders (e.g. one per learner) into --out')
    a = ap.parse_args()
    if a.merge:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        runs = pd.concat([pd.read_csv(Path(d) / 'tuning_budget_runs.csv') for d in a.merge], ignore_index=True)
        runs.to_csv(out / 'tuning_budget_runs.csv', index=False)
        ids = [Path(d) / 'tuning_budget_identity.csv' for d in a.merge]
        if any(f.exists() for f in ids):
            pd.concat([pd.read_csv(f) for f in ids if f.exists()], ignore_index=True).to_csv(out / 'tuning_budget_identity.csv', index=False)
        summarise(runs, out, a.check, a.trials)
        if a.published and Path(a.published).exists():
            published_check(runs, Path(a.published), out, a.check)
        log(f'merged {len(runs)} refits from {len(a.merge)} folders'); return

    cfg = load_config(a.config)
    out = Path(a.out); (out / 'refits').mkdir(parents=True, exist_ok=True)
    numeric, categorical = cfg['features']['numeric'], cfg['features']['categorical']
    tune_seed = int(cfg['tuning']['canonical_seed'])
    if int(cfg['tuning']['trials_per_model']) != a.check:
        log(f'note: the config says {cfg["tuning"]["trials_per_model"]} trials for the main study; --check is {a.check}')
    seeds = a.seeds or [int(s) for s in cfg['project']['five_seeds']]
    trials, check = a.trials, a.check
    if a.smoke:
        trials, check = 3, 2
        for k in ('catboost',):
            cfg['models'][k] = {**cfg['models'][k], 'iterations': 60}
    # Two searches may share the GPU (one per learner). CatBoost reserves 95 % of the free GPU memory by default, which
    # would starve the other process; EVTRUST_CATBOOST_GPU_RAM_PART caps its share (memory only, not the algorithm).
    ram = os.environ.get('EVTRUST_CATBOOST_GPU_RAM_PART')
    if ram and str(cfg['models']['catboost'].get('task_type', '')).upper() == 'GPU' and not os.environ.get('EVTRUST_CATBOOST_TASK_TYPE'):
        cfg['models']['catboost'] = {**cfg['models']['catboost'], 'gpu_ram_part': float(ram)}
        log(f'CatBoost gpu_ram_part = {float(ram)}')
    torch_epochs = int(cfg['tuning']['torch_tuning_epochs'])
    if a.smoke:
        torch_epochs = 2
        cfg['models']['torch_ft'] = {**cfg['models']['torch_ft'], 'epochs': 2, 'patience': 1}
    df = pd.read_parquet(a.sessions)
    storage = f'sqlite:///{(out / "optuna_studies.db").resolve()}'
    json.dump({'trials': trials, 'check': check, 'learners': a.learners, 'targets': a.targets, 'seeds': seeds,
               'tune_seed': tune_seed, 'config': str(a.config), 'sessions': str(a.sessions)},
              open(out / 'tuning_budget_settings.json', 'w'), indent=1)

    from evtrust.tuning import catboost_objective, torch_objective
    split_names = [sp.name for sp in make_splits(df, cfg, tune_seed)]
    log(f'splits: {split_names}; learners {a.learners}; targets {a.targets}; budget {check} and {trials} trials')
    rows, identity = [], []
    pub_params = json.load(open(a.published_params)) if a.published_params and Path(a.published_params).exists() else None
    for learner in a.learners:
        for target in a.targets:
            ycol = TARGETS[target]
            for split_name in split_names:
                # 1. the search, on the canonical seed's training and validation data (as in the main study)
                seed_everything(tune_seed)
                fr = frames(df, cfg, tune_seed, split_name, ycol)
                if fr is None:
                    log(f'skip {learner} {target} {split_name}: too few rows (also skipped by the main study)'); continue
                tr, va, _ = fr
                ytr, yva = tr[ycol].to_numpy(float), va[ycol].to_numpy(float)
                name = f'{learner}|{target}|{split_name}'
                st = study_for(storage, name, tune_seed)
                if 0 < len(st.trials) and len(complete_trials(st)) < trials:
                    # A seeded sampler re-created after an interruption would not continue the original sequence of
                    # trials, so a partly finished search is discarded and run again from its first trial.
                    import optuna
                    log(f'{name}: discarding an interrupted search ({len(st.trials)} trials) and starting it again')
                    optuna.delete_study(study_name=name, storage=storage)
                    st = study_for(storage, name, tune_seed)
                if len(st.trials) == 0:
                    base = cfg['models'][learner]
                    obj = (catboost_objective(tr, ytr, va, yva, numeric, categorical, base, tune_seed) if learner == 'catboost'
                           else torch_objective(tr, ytr, va, yva, numeric, categorical, base, tune_seed, torch_epochs))
                    t0 = time.time()
                    st.optimize(obj, n_trials=trials)            # one call, one sampler: the sequence of the main study
                    log(f'{name}: {trials} trials in {(time.time() - t0) / 60:.1f} min')
                ct = complete_trials(st)
                if len(ct) < trials:
                    raise RuntimeError(f'{name}: only {len(ct)} of {trials} trials completed')
                if pub_params is not None:
                    identity.append(identity_row(pub_params, learner, target, split_name, ct, check))
                first = sorted(ct, key=lambda t: t.number)[:check]
                configs = {f'{check} trials': best_of(first), f'{trials} trials': best_of(ct)}
                # 2. refit both selections for every seed and score them on that seed's test set
                for seed in seeds:
                    seed_everything(seed)
                    fr = frames(df, cfg, seed, split_name, ycol)
                    if fr is None:
                        continue
                    trs, vas, tes = fr
                    cache = {}
                    for budget, trial in configs.items():
                        f = out / 'refits' / f'{learner}__{target}__{split_name.replace(":", "__")}__seed{seed}__{budget.replace(" ", "")}.json'
                        if f.exists():
                            rows.append(json.load(open(f))); continue
                        if trial.number in cache:
                            mae, n = cache[trial.number]
                        else:
                            p = copy.deepcopy(cfg['models'][learner]); p.update(trial.params); p['_ycol'] = ycol
                            mae, n = fit_predict(learner, trs, vas, tes, numeric, categorical, p, seed)
                            cache[trial.number] = (mae, n)
                        r = {'learner': learner, 'target': target, 'split': split_name, 'protocol': protocol_of(split_name),
                             'seed': seed, 'budget': budget, 'trial_number': trial.number, 'validation_mae_canonical': trial.value,
                             'test_mae': mae, 'n_test': n, 'params': json.dumps(trial.params, sort_keys=True)}
                        json.dump(r, open(f, 'w')); rows.append(r)
                log(f'{name}: refits done (selected trials {[c.number for c in configs.values()]})')
    runs = pd.DataFrame(rows)
    runs.to_csv(out / 'tuning_budget_runs.csv', index=False)
    if identity:
        pd.DataFrame(identity).to_csv(out / 'tuning_budget_identity.csv', index=False)
    summarise(runs, out, check, trials)
    if a.published and Path(a.published).exists():
        published_check(runs, Path(a.published), out, check)
    log('tuning budget written')


def summarise(runs: pd.DataFrame, out: Path, check: int, trials: int):
    """Protocol MAE: mean over seeds, with the site-disjoint splits averaged first (sites weighted equally), as in the
    paper's protocol table."""
    b8, bN = f'{check} trials', f'{trials} trials'
    per = (runs.groupby(['learner', 'target', 'protocol', 'budget', 'seed'])['test_mae'].mean()      # sites weighted equally
               .groupby(['learner', 'target', 'protocol', 'budget']).agg(['mean', 'min', 'max']).reset_index())
    rows = []
    for (learner, target, protocol), g in per.groupby(['learner', 'target', 'protocol']):
        g = g.set_index('budget')
        if b8 not in g.index or bN not in g.index:
            continue
        m8, mN = g.loc[b8, 'mean'], g.loc[bN, 'mean']
        rows.append({'learner': learner, 'target': target, 'protocol': protocol, f'mae_{check}': m8, f'mae_{trials}': mN,
                     'change_pct': 100 * (mN / m8 - 1)})
    s = pd.DataFrame(rows)
    # protocol gap against random splitting, under both budgets
    for budget, col in ((check, f'mae_{check}'), (trials, f'mae_{trials}')):
        rnd = s[s['protocol'] == 'random'].set_index(['learner', 'target'])[col]
        s[f'gap_vs_random_pct_{budget}'] = [100 * (r[col] / rnd.get((r['learner'], r['target']), np.nan) - 1) for _, r in s.iterrows()]
    # does the better learner change with the budget?
    if {'catboost', 'torch_ft'} <= set(s['learner']):
        w = []
        for (target, protocol), g in s.groupby(['target', 'protocol']):
            g = g.set_index('learner')
            if {'catboost', 'torch_ft'} <= set(g.index):
                w8 = 'catboost' if g.loc['catboost', f'mae_{check}'] <= g.loc['torch_ft', f'mae_{check}'] else 'torch_ft'
                wN = 'catboost' if g.loc['catboost', f'mae_{trials}'] <= g.loc['torch_ft', f'mae_{trials}'] else 'torch_ft'
                w.append({'target': target, 'protocol': protocol, f'better_{check}': w8, f'better_{trials}': wN, 'changed': w8 != wN})
        s = s.merge(pd.DataFrame(w), on=['target', 'protocol'], how='left')
    s.to_csv(out / 'tuning_budget_summary.csv', index=False)


def published_check(runs: pd.DataFrame, path: Path, out: Path, check: int):
    p = pd.read_parquet(path, columns=['seed', 'split_name', 'target', 'model', 'y_true', 'y_pred'])
    p = p[p['model'].isin(runs['learner'].unique())]
    p['ae'] = (p['y_true'] - p['y_pred']).abs()
    pub = p.groupby(['model', 'target', 'split_name', 'seed'])['ae'].mean().rename('published_mae').reset_index()
    r = runs[runs['budget'] == f'{check} trials'].rename(columns={'learner': 'model', 'split': 'split_name'})
    m = r.merge(pub, on=['model', 'target', 'split_name', 'seed'], how='left')
    m['difference_pct'] = 100 * (m['test_mae'] / m['published_mae'] - 1)
    m[['model', 'target', 'split_name', 'seed', 'test_mae', 'published_mae', 'difference_pct']].to_csv(
        out / 'tuning_budget_published.csv', index=False)


if __name__ == '__main__':
    main()
