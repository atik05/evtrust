"""Tuning-budget sensitivity (scripts/tuning_budget.py): the paired design and the protocol summary."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
optuna = pytest.importorskip('optuna')
spec = importlib.util.spec_from_file_location('tuning_budget', ROOT / 'scripts' / 'tuning_budget.py')
tb = importlib.util.module_from_spec(spec)
sys.modules['tuning_budget'] = tb
spec.loader.exec_module(tb)


def _space(trial):
    return (trial.suggest_int('iterations', 800, 3000, step=200) + trial.suggest_int('depth', 6, 10) * 100
            + trial.suggest_float('learning_rate', .015, .10, log=True) * 1e4 + trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True)
            + trial.suggest_float('random_strength', .2, 3.0, log=True)) % 997


def test_first_trials_of_a_longer_search_are_the_short_search(tmp_path):
    """The eight trials of the main study are the first eight trials of the longer search (same sampler and seed)."""
    a = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=11))
    a.optimize(_space, n_trials=8)
    b = tb.study_for(f'sqlite:///{tmp_path / "s.db"}', 'x', 11)
    b.optimize(_space, n_trials=20)
    assert [t.params for t in a.trials] == [t.params for t in b.trials[:8]]
    assert tb.best_of(tb.complete_trials(b)[:8]).params == a.best_params
    assert tb.best_of(tb.complete_trials(b)).value <= a.best_value


def test_summary_weights_sites_equally_and_flags_learner_changes(tmp_path):
    rows = []
    for learner, base in (('catboost', 1.0), ('torch_ft', 1.1)):
        for budget, f in (('8 trials', 1.0), ('40 trials', 0.95 if learner == 'torch_ft' else 1.0)):
            for seed in (11, 23):
                for split, m in (('random', 1.0), ('user_disjoint', 1.5), ('site_disjoint:caltech', 2.0), ('site_disjoint:jpl', 3.0)):
                    v = base * m * f * (0.5 if (learner == 'torch_ft' and budget == '40 trials' and split == 'site_disjoint:jpl') else 1.0)
                    rows.append({'learner': learner, 'target': 'energy', 'split': split, 'protocol': tb.protocol_of(split),
                                 'seed': seed, 'budget': budget, 'test_mae': v})
    tb.summarise(pd.DataFrame(rows), tmp_path, 8, 40)
    s = pd.read_csv(tmp_path / 'tuning_budget_summary.csv').set_index(['learner', 'protocol'])
    assert s.loc[('catboost', 'site_disjoint'), 'mae_8'] == pytest.approx(2.5)          # (2 + 3) / 2, sites equal
    assert s.loc[('catboost', 'random'), 'change_pct'] == pytest.approx(0.0)
    assert s.loc[('torch_ft', 'random'), 'change_pct'] == pytest.approx(-5.0)
    assert s.loc[('catboost', 'user_disjoint'), 'gap_vs_random_pct_8'] == pytest.approx(50.0)
    # FT at 40 trials: site-disjoint 1.1 * 0.95 * (2 + 1.5) / 2 = 1.828 < 2.5, so the better learner changes there
    assert bool(s.loc[('torch_ft', 'site_disjoint'), 'changed'])
    assert not bool(s.loc[('torch_ft', 'random'), 'changed'])
    assert s.loc[('torch_ft', 'site_disjoint'), 'better_40'] == 'torch_ft'
