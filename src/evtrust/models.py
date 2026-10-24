from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os
import platform
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge


@dataclass
class FittedModel:
    name: str
    model: Any
    numeric: list[str]
    categorical: list[str]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.name == 'catboost':
            x = _catboost_frame(df, self.numeric, self.categorical)
            return np.asarray(self.model.predict(x), dtype=float)
        return np.asarray(self.model.predict(df[self.numeric + self.categorical]), dtype=float)


def _sk_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    cat = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=3)),
    ])
    return ColumnTransformer([('num', num, numeric), ('cat', cat, categorical)], remainder='drop')


def fit_ridge(train: pd.DataFrame, y: np.ndarray, numeric: list[str], categorical: list[str], alpha: float = 10.0) -> FittedModel:
    pipe = Pipeline([
        ('prep', _sk_preprocessor(numeric, categorical)),
        ('model', Ridge(alpha=alpha)),
    ])
    pipe.fit(train[numeric + categorical], y)
    return FittedModel('ridge', pipe, numeric, categorical)


def _catboost_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    x = df[numeric + categorical].copy()
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    for c in categorical:
        x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
    return x


def fit_catboost(train: pd.DataFrame, y: np.ndarray, validation: pd.DataFrame, y_val: np.ndarray,
                 numeric: list[str], categorical: list[str], params: dict, seed: int) -> FittedModel:
    from catboost import CatBoostRegressor
    p = dict(params)
    p['random_seed'] = seed
    p.setdefault('allow_writing_files', False)
    # CatBoost GPU uses NVIDIA CUDA; macOS Intel/AMD must use CPU.
    forced = os.environ.get('EVTRUST_CATBOOST_TASK_TYPE')
    if forced:
        p['task_type'] = forced.upper()
    if platform.system() == 'Darwin' or str(p.get('task_type', '')).upper() == 'CPU':
        p['task_type'] = 'CPU'
        p.pop('devices', None)
        p.setdefault('thread_count', int(os.environ.get('EVTRUST_CPU_THREADS', '8')))
    model = CatBoostRegressor(**p)
    xtr = _catboost_frame(train, numeric, categorical)
    xva = _catboost_frame(validation, numeric, categorical)
    cat_idx = [xtr.columns.get_loc(c) for c in categorical]
    model.fit(xtr, y, cat_features=cat_idx, eval_set=(xva, y_val), use_best_model=True, early_stopping_rounds=150)
    return FittedModel('catboost', model, numeric, categorical)


def fit_catboost_quantile(train: pd.DataFrame, y: np.ndarray, validation: pd.DataFrame, y_val: np.ndarray,
                          numeric: list[str], categorical: list[str], params: dict, seed: int, alpha: float):
    from catboost import CatBoostRegressor
    p = dict(params)
    p['loss_function'] = f'Quantile:alpha={alpha}'
    p['random_seed'] = seed
    p.setdefault('allow_writing_files', False)
    # CatBoost GPU uses NVIDIA CUDA; macOS Intel/AMD must use CPU.
    forced = os.environ.get('EVTRUST_CATBOOST_TASK_TYPE')
    if forced:
        p['task_type'] = forced.upper()
    if platform.system() == 'Darwin' or str(p.get('task_type', '')).upper() == 'CPU':
        p['task_type'] = 'CPU'
        p.pop('devices', None)
        p.setdefault('thread_count', int(os.environ.get('EVTRUST_CPU_THREADS', '8')))
    model = CatBoostRegressor(**p)
    xtr = _catboost_frame(train, numeric, categorical)
    xva = _catboost_frame(validation, numeric, categorical)
    cat_idx = [xtr.columns.get_loc(c) for c in categorical]
    model.fit(xtr, y, cat_features=cat_idx, eval_set=(xva, y_val), use_best_model=True, early_stopping_rounds=150)
    return FittedModel(f'catboost_q{alpha}', model, numeric, categorical)
