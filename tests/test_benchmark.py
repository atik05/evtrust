"""Invariants of the repeated cross-validated benchmark."""
import numpy as np
import pandas as pd

from evtrust import benchmark


def _sessions(n_users=40, per_user=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp('2019-01-01', tz='UTC')
    for u in range(n_users):
        for i in range(per_user):
            start = t0 + pd.Timedelta(hours=int(rng.integers(0, 24 * 300)))
            dur = float(rng.uniform(1, 10))
            rows.append({'user_id': f'u{u}' if u % 7 else None, 'site': ['a', 'b', 'c'][u % 3],
                         'station_id': f'st{u % 3}_{rng.integers(0, 4)}', 'connection_time': start,
                         'disconnect_time': start + pd.Timedelta(hours=dur), 'duration_hours': dur,
                         'kwh_delivered': dur * 2 + rng.normal(0, 1), 'x1': dur + rng.normal(0, 0.3),
                         'user_prior_mean_energy': rng.normal(10, 2), 'user_prior_median_duration': rng.normal(5, 1),
                         'station_prior_count': rng.integers(0, 50)})
    return pd.DataFrame(rows)


def test_driver_grouped_folds_share_no_driver_and_exclude_anonymous():
    d = benchmark._frame(_sessions(), 'kwh_delivered')
    for r, f, tr, te, _ in benchmark.splits(d, 'driver_grouped', repeats=2, folds=5):
        a, b = set(d.loc[tr, 'user_id']), set(d.loc[te, 'user_id'])
        assert not (a & b)
        assert d.loc[np.r_[tr, te], 'known_driver'].all()


def test_chronological_training_labels_mature_before_test_block():
    d = benchmark._frame(_sessions(), 'kwh_delivered')
    for r, f, tr, te, _ in benchmark.splits(d, 'chronological', repeats=1, folds=5):
        assert (d.loc[tr, 't_end'] < d.loc[te, 't'].min()).all()


def test_site_transfer_holds_out_whole_sites_and_drops_station_information():
    d = benchmark._frame(_sessions(), 'kwh_delivered')
    for r, f, tr, te, drop in benchmark.splits(d, 'site_transfer', repeats=1):
        assert not (set(d.loc[tr, 'site']) & set(d.loc[te, 'site']))
        assert 'station_code' in drop and 'station_prior_count' in drop


def test_random_protocol_predicts_every_row_once_per_repeat():
    d = benchmark._frame(_sessions(), 'kwh_delivered')
    seen = np.zeros(len(d), int)
    for r, f, tr, te, _ in benchmark.splits(d, 'random', repeats=1, folds=5):
        seen[te] += 1
    assert (seen == 1).all()


def test_corrected_ttest_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.05, 0.1, 25)
    t_corr, p_corr = benchmark.corrected_ttest(diff, n_train=800, n_test=200)
    from scipy import stats
    p_naive = stats.ttest_1samp(diff, 0).pvalue
    assert p_corr > p_naive


def test_run_and_summarise_end_to_end_with_failed_optional_model():
    df = _sessions()
    scores = benchmark.run(df, ['x1', 'user_prior_mean_energy', 'user_prior_median_duration', 'station_prior_count'],
                           'kwh_delivered', 'random', ['mean', 'ridge', 'hgb', 'tabpfn_missing'], {},
                           repeats=2, folds=3, log=lambda m: None)
    assert set(scores['model']) == {'mean', 'ridge', 'hgb', 'tabpfn_missing'}
    assert scores.loc[scores['model'] == 'tabpfn_missing', 'status'].str.startswith('failed').all()
    summ, pairs, fried = benchmark.summarise(scores)
    assert set(summ['model']) == {'mean', 'ridge', 'hgb'}
    assert summ.loc[summ['model'] == 'mean', 'mae'].iloc[0] > summ.loc[summ['model'] == 'ridge', 'mae'].iloc[0]
    assert not fried.empty and (fried['nemenyi_cd'] > 0).all()


def test_prepandemic_protocol_uses_only_pre_break_sessions():
    df = _sessions()
    df['connection_time'] = pd.Timestamp('2019-06-01', tz='UTC') + pd.to_timedelta(np.arange(len(df)) * 3, unit='h')
    df['disconnect_time'] = df['connection_time'] + pd.Timedelta(hours=2)
    d = benchmark._frame(df, 'kwh_delivered')
    brk = pd.Timestamp(benchmark.PANDEMIC_BREAK, tz='UTC')
    n = 0
    for r, f, tr, te, _ in benchmark.splits(d, 'chronological_prepandemic', repeats=1, folds=5):
        assert (d.loc[np.r_[tr, te], 't'] < brk).all()
        assert (d.loc[tr, 't_end'] < d.loc[te, 't'].min()).all()
        n += 1
    assert n == 5


def test_strict_site_transfer_removes_user_history_too():
    d = benchmark._frame(_sessions(), 'kwh_delivered')
    for r, f, tr, te, drop in benchmark.splits(d, 'site_transfer_strict', repeats=1):
        assert {'user_prior_count', 'user_prior_mean_energy', 'station_code'} <= set(drop)
        assert not (set(d.loc[tr, 'site']) & set(d.loc[te, 'site']))
    scores = benchmark.run(_sessions(), ['x1', 'user_prior_mean_energy', 'user_prior_median_duration', 'station_prior_count'],
                           'kwh_delivered', 'site_transfer_strict', ['mean', 'user_history', 'ridge'], {},
                           repeats=1, log=lambda m: None)
    assert (scores['status'] == 'ok').all()
