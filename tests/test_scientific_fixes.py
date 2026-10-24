from pathlib import Path
import numpy as np
import pandas as pd

from evtrust.splits import chronological_split, site_disjoint_splits, user_disjoint_split
from evtrust.occupancy_experiment import _chronological_parts
from evtrust.experiments import _write_rq3_driver_matched_metrics, _write_pandemic_metrics


def session_frame(n=200):
    t = pd.date_range('2020-01-01', periods=n, freq='6h', tz='America/Los_Angeles')
    return pd.DataFrame({
        'connection_time': t,
        'disconnect_time': t + pd.Timedelta(hours=2),
        'user_id': [f'u{i%40}' for i in range(n)],
        'site': np.where(np.arange(n) % 3 == 0, 'a', np.where(np.arange(n) % 3 == 1, 'b', 'c')),
    })


def test_chronological_order_is_train_validation_calibration_test():
    df = session_frame()
    sp = chronological_split(df, embargo_hours=0)
    assert df.loc[sp.train, 'connection_time'].max() < df.loc[sp.validation, 'connection_time'].min()
    assert df.loc[sp.validation, 'connection_time'].max() < df.loc[sp.calibration, 'connection_time'].min()
    assert df.loc[sp.calibration, 'connection_time'].max() < df.loc[sp.test, 'connection_time'].min()


def test_user_disjoint_excludes_anonymous_rows():
    df = session_frame(120)
    df.loc[[0, 7, 11], 'user_id'] = pd.NA
    sp = user_disjoint_split(df, seed=7)
    used = set(sp.train) | set(sp.validation) | set(sp.calibration) | set(sp.test)
    assert not ({0, 7, 11} & used)
    assert sp.metadata['excluded_anonymous_rows'] == 3


def test_site_disjoint_target_calibration_is_purged_before_test():
    df = session_frame(300)
    sp = next(x for x in site_disjoint_splits(df, embargo_hours=4) if x.metadata['target_site'] == 'a')
    boundary = df.loc[sp.test, 'connection_time'].min()
    assert (df.loc[sp.calibration, 'disconnect_time'] < boundary - pd.Timedelta(hours=4)).all()


def occupancy_panel():
    q = pd.date_range('2020-01-01', periods=120, freq='1h', tz='UTC')
    return pd.DataFrame({
        'query_time': np.repeat(q, 2),
        'site': ['a','a'] * len(q),
        'station_id': ['s1','s2'] * len(q),
        'occupied_now': [0, 1] * len(q),
        'query_id': [f'a|{t}' for t in q for _ in range(2)],
        'free_1h': 1,
        'free_2h': 1,
        'free_4h': 1,
    })


def test_occupancy_longest_horizon_embargo():
    p = occupancy_panel()
    tr, va, ca, te = _chronological_parts(p, 4)
    va_start = p.loc[va, 'query_time'].min()
    ca_start = p.loc[ca, 'query_time'].min()
    te_start = p.loc[te, 'query_time'].min()
    assert (p.loc[tr, 'query_time'] + pd.Timedelta(hours=4) < va_start).all()
    assert (p.loc[va, 'query_time'] + pd.Timedelta(hours=4) < ca_start).all()
    assert (p.loc[ca, 'query_time'] + pd.Timedelta(hours=4) < te_start).all()


def test_rq3_metrics_use_identical_driver_available_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('results/metrics').mkdir(parents=True)
    rows = []
    for model, ids in [('driver_estimate', [1, 2]), ('ridge', [1, 2, 3]), ('catboost', [1, 2, 3])]:
        for rid in ids:
            rows.append({
                'target':'duration','protocol':'chronological','split_name':'chronological',
                'target_site':np.nan,'seed':11,'model':model,'row_id':rid,
                'y_true':float(rid),'y_pred':float(rid)+0.1,
            })
    _write_rq3_driver_matched_metrics(pd.DataFrame(rows))
    out = pd.read_csv('results/metrics/rq3_driver_matched_metrics_by_seed.csv')
    assert set(out['n_matched']) == {2}
    assert set(out['n_test_population']) == {3}
    assert set(out['model']) == {'driver_estimate','ridge','catboost'}


def test_pandemic_report_contains_pre_and_post(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('results/metrics').mkdir(parents=True)
    t_pre = pd.date_range('2020-01-01', periods=25, freq='1D', tz='America/Los_Angeles')
    t_post = pd.date_range('2020-03-02', periods=25, freq='1D', tz='America/Los_Angeles')
    t = t_pre.append(t_post)
    p = pd.DataFrame({
        'target':'duration','protocol':'random','split_name':'random','target_site':np.nan,
        'seed':11,'model':'ridge','connection_time':t,'y_true':1.0,'y_pred':1.1,
    })
    cfg={'data':{'timezone':'America/Los_Angeles'},'splits':{'pandemic_break':'2020-03-01'}}
    _write_pandemic_metrics(p,cfg)
    out = pd.read_csv('results/metrics/pandemic_stratified_metrics_by_seed.csv')
    assert set(out['pandemic_period']) == {'pre_break','post_break'}
