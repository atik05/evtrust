from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .splits import random_split, chronological_split, user_disjoint_split, site_disjoint_splits, validate_split

LEAKAGE_FORBIDDEN = {
    'disconnect_time': 'Known only after the stay ends.',
    'done_charging_time': 'Known after charging completion.',
    'duration_hours': 'Regression target, not an arrival-time feature.',
    'kwh_delivered': 'Regression target, not an arrival-time feature.',
}


def _strict_before(df: pd.DataFrame, left, right, embargo_hours: float) -> bool:
    if len(left) == 0 or len(right) == 0:
        return False
    boundary = df.loc[right, 'connection_time'].min()
    return bool((df.loc[left, 'disconnect_time'] < boundary - pd.Timedelta(hours=embargo_hours)).all())


def _scientific_split_checks(df: pd.DataFrame, splits, embargo_hours: float) -> list[dict]:
    rows = []
    for sp in splits:
        if sp.name == 'chronological':
            temporal_order = (
                len(sp.train) and len(sp.validation) and len(sp.calibration) and len(sp.test)
                and df.loc[sp.train, 'connection_time'].max() < df.loc[sp.validation, 'connection_time'].min()
                and df.loc[sp.validation, 'connection_time'].max() < df.loc[sp.calibration, 'connection_time'].min()
                and df.loc[sp.calibration, 'connection_time'].max() < df.loc[sp.test, 'connection_time'].min()
            )
            maturation = (
                _strict_before(df, sp.train, sp.validation, embargo_hours)
                and _strict_before(df, sp.validation, sp.calibration, embargo_hours)
                and _strict_before(df, sp.calibration, sp.test, embargo_hours)
            )
            rows.append({'name':'chronological_order_and_maturation','split_name':sp.name,
                         'ok':bool(temporal_order and maturation),
                         'temporal_order_ok':bool(temporal_order),'label_maturation_ok':bool(maturation)})
        elif sp.name == 'user_disjoint':
            used = list(sp.train) + list(sp.validation) + list(sp.calibration) + list(sp.test)
            identified = bool(df.loc[used, 'user_id'].notna().all()) if used else False
            rows.append({'name':'anonymous_user_policy','split_name':sp.name,'ok':identified,
                         'anonymous_rows_excluded':int(sp.metadata.get('excluded_anonymous_rows',0))})
        elif sp.name.startswith('site_disjoint'):
            target_cal_ok = _strict_before(df, sp.calibration, sp.test, embargo_hours)
            source_train_ok = _strict_before(df, sp.train, sp.validation, embargo_hours)
            rows.append({'name':'site_disjoint_maturation','split_name':sp.name,
                         'ok':bool(target_cal_ok and source_train_ok),
                         'source_train_before_validation_ok':bool(source_train_ok),
                         'target_calibration_before_test_ok':bool(target_cal_ok)})
    return rows


def leakage_audit(df: pd.DataFrame, cfg: dict) -> dict:
    numeric = cfg['features']['numeric']
    categorical = cfg['features']['categorical']
    configured = set(numeric + categorical)
    forbidden_used = sorted(configured & set(LEAKAGE_FORBIDDEN))
    missing = sorted([c for c in configured if c not in df.columns])

    s = cfg['splits']
    seed = cfg['project']['seed']
    splits = [
        random_split(df, seed, s['validation_fraction'], s['calibration_fraction'], s['test_fraction']),
        chronological_split(df, s['validation_fraction'], s['calibration_fraction'], s['test_fraction'], s['embargo_hours']),
        user_disjoint_split(df, seed, s['validation_fraction'], s['calibration_fraction'], s['test_fraction']),
        *site_disjoint_splits(df, s['validation_fraction'], s['calibration_fraction'], s['embargo_hours'])
    ]
    checks = [validate_split(df, sp) for sp in splits]
    scientific = _scientific_split_checks(df, splits, float(s['embargo_hours']))
    ok = (
        not forbidden_used
        and not missing
        and all(all(v == 0 for v in x['index_overlap'].values()) for x in checks)
        and all(x['ok'] for x in scientific)
    )
    return {
        'ok': ok,
        'forbidden_features_used': forbidden_used,
        'missing_features': missing,
        'split_checks': checks,
        'scientific_split_checks': scientific,
    }


def write_audit(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
