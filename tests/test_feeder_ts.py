"""Time-series hosting capacity (robustness/feeder_ts.py) and its power flow (robustness/fastpf.py)."""
import json

import numpy as np
import pandas as pd
import pytest

pp = pytest.importorskip('pandapower')
sb = pytest.importorskip('simbench')

from robustness import feeder_ts as fts  # noqa: E402
from robustness.acnsim import run_id, site_runs  # noqa: E402
from robustness.fastpf import FastPF  # noqa: E402


@pytest.fixture(scope='module')
def tsf():
    return fts.TSFeeder()


def test_fastpf_matches_pandapower(tsf):
    v = tsf.F.validate(n=12, seed=3)
    assert v['max_abs_vm_pu'] < 1e-7
    assert v['max_abs_line_loading_pct'] < 1e-4
    assert v['max_abs_trafo_loading_pct'] < 1e-4


def test_fastpf_ieee33_matches_pandapower():
    import pandapower.networks as pn
    F = FastPF(pn.case33bw())
    v = F.validate(n=8, seed=0, extra_mw=0.5)
    assert v['max_abs_vm_pu'] < 1e-7


def test_profile_steps_and_base_case(tsf):
    t = 20000
    base = tsf.base_case([t])[0]
    net = sb.get_simbench_net(fts.SIMBENCH)
    prof = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    net.load['p_mw'] = prof[('load', 'p_mw')].iloc[t].to_numpy(); net.load['q_mvar'] = prof[('load', 'q_mvar')].iloc[t].to_numpy()
    net.sgen['p_mw'] = prof[('sgen', 'p_mw')].iloc[t].to_numpy()
    pp.runpp(net, numba=True)
    mv = net.bus.index[net.bus['vn_kv'] < 100]
    assert abs(base[0] - net.res_bus.loc[mv, 'vm_pu'].min()) < 1e-7
    assert abs(base[1] - net.res_bus.loc[mv, 'vm_pu'].max()) < 1e-7
    assert abs(base[2] - net.res_line['loading_percent'].max()) < 1e-4
    assert abs(base[3] - net.res_trafo['loading_percent'].max()) < 1e-4


def test_no_worsening_rule_and_monotone_search(tsf):
    steps = [20000, 20001, 20002]
    tsf.set_base(steps, tsf.base_case(steps))
    kw = np.zeros(fts.STEPS); kw[steps] = 100.0
    zero = np.zeros(len(tsf.cand))
    assert tsf.ok(zero, kw, steps, [])[0]
    many = zero.copy(); many[:] = 50.0                                   # 50 hubs of 100 kW on every load bus
    good, why = tsf.ok(many, kw, steps, [])
    assert not good and why in ('voltage', 'thermal', 'power flow did not converge')
    order = np.random.default_rng(0).integers(0, len(tsf.cand), 1000)
    h, why, calls = tsf.hosting(order, kw, steps, 1000)
    assert 0 < h < 1000 and why in ('voltage', 'thermal')
    c = np.bincount(order[:int(h)], minlength=len(tsf.cand)).astype(float)
    assert tsf.ok(c, kw, steps, [])[0]
    c2 = np.bincount(order[:int(h) + 2], minlength=len(tsf.cand)).astype(float)
    assert not tsf.ok(c2, kw, steps, [])[0]


def test_hub_profile_calendar(tmp_path):
    origin = pd.Timestamp('2018-04-23')                                  # a Monday, 5-minute steps
    n = 7 * 24 * 12 * 100
    t = origin + pd.to_timedelta(np.arange(n) * 5, unit='min')
    load = np.where((t.weekday < 5) & (t.hour >= 8) & (t.hour < 17), 30.0, 0.0)
    load[(t.normalize() == pd.Timestamp('2019-03-12'))] = 99.0           # a marked Tuesday
    np.savez_compressed(tmp_path / 'r.power.npz', offsets=np.array([0]), p0=load.astype(np.float32))
    h = fts.hub_profile(tmp_path / 'r.power.npz', origin, 5)
    assert h.shape == (fts.STEPS,)
    cal = pd.date_range(fts.YEAR0, periods=fts.STEPS, freq='15min')
    assert h[(cal.weekday >= 5)].max() == 0                               # weekends stay weekends
    src = fts.SOURCE0 + (cal - fts.YEAR0)
    other = (cal.weekday < 5) & (cal.hour == 10) & (src.normalize() != pd.Timestamp('2019-03-12'))
    assert np.allclose(h[other], 30.0)                                   # weekday working hours carry the load
    assert np.allclose(h[(cal.weekday < 5) & (cal.hour == 7) & (src.normalize() != pd.Timestamp('2019-03-12'))], 0.0)
    assert h[src.normalize() == pd.Timestamp('2019-03-12')].max() == pytest.approx(99.0)
    assert (cal[src.normalize() == pd.Timestamp('2019-03-12')].weekday == 1).all()   # Tuesday onto a Tuesday


def test_orders_are_feeder_py_orders():
    from robustness.feeder import Feeder
    cfg = {'feeders': ['ieee33', fts.SIMBENCH], 'orders': 4, 'nmax': 50, 'simbench_critical_steps': 2}
    rng = np.random.default_rng(20261124)
    ref = {}
    for fname in cfg['feeders']:                                         # the loop of feeder.run
        F = Feeder(fname, cfg)
        ref[fname] = [rng.integers(0, len(F.cand), cfg['nmax']) for _ in range(cfg['orders'])]
    got = fts.simbench_orders(cfg, 3)
    assert all(np.array_equal(a, b) for a, b in zip(got, ref[fts.SIMBENCH][:3]))


def test_site_runs_from_capacity_needed(tmp_path):
    n = pd.DataFrame({'site': ['caltech'] * 6, 'scheduler': ['fcfs', 'llf', 'llf', 'llf', 'llf', 'llf'],
                      'forecast': ['none', 'oracle', 'benchmark', 'deployment', 'cold', 'cold'],
                      'tau': ['na', 'na', 0.5, 0.5, 0.5, 0.1],
                      'capacity_needed_fraction': [0.483081, 0.378334, 0.451443, 0.458183, 0.476784, 0.48]})
    n.to_csv(tmp_path / 'c.csv', index=False)
    r = site_runs({'e6_capacity_needed': str(tmp_path / 'c.csv')}, 'caltech')
    assert len(r) == 5 and r[0] == ('E6', 'fcfs', 0.4831, 'none', None) and r[4] == ('E6', 'llf', 0.4768, 'cold', 0.5)
    ids = [run_id('caltech', x) for x in r]
    assert len(set(ids)) == 5 and ids[3] == 'caltech__E6__llf__cap0.4582__deployment__tau0.5'
    assert site_runs({}, 'caltech') == []
    assert run_id('caltech', ('E2', 'llf', 0.5, 'oracle', None)) == 'caltech__E2__llf__cap0.50__oracle__taucna'.replace('taucna', 'tauna')


def test_diverse_profiles_keep_weekdays_and_season(tmp_path):
    origin = pd.Timestamp('2018-04-23')
    n = 7 * 24 * 12 * 100
    t = origin + pd.to_timedelta(np.arange(n) * 5, unit='min')
    wk = ((t - origin).days // 7).to_numpy()
    load = np.where((t.weekday < 5) & (t.hour >= 8) & (t.hour < 17), 10.0 + wk, 0.0)    # level marks the source week
    np.savez_compressed(tmp_path / 'r.power.npz', offsets=np.array([0]), p0=load.astype(np.float32))
    s15 = fts.site_series(tmp_path / 'r.power.npz', origin, 5)
    sh = fts.week_shifts(6, 4)
    assert sh.shape == (6, 53) and sh.min() >= -4 and sh.max() <= 4 and (sh != 0).any()
    P = fts.year_profiles(s15, sh)
    P0 = fts.year_profiles(s15, None)
    assert P.shape == (6, fts.STEPS) and P0.shape == (1, fts.STEPS)
    assert np.allclose(P0[0], fts.hub_profile(tmp_path / 'r.power.npz', origin, 5))
    cal = pd.date_range(fts.YEAR0, periods=fts.STEPS, freq='15min')
    assert P[:, cal.weekday >= 5].max() == 0 and P[:, (cal.hour < 8) | (cal.hour >= 17)].max() == 0
    # in target week j, variant m carries the source week shifted by sh[m, j]
    j = 20; day = (cal - fts.YEAR0).days
    at = np.flatnonzero((day // 7 == j) & (cal.weekday < 5) & (cal.hour == 10))[0]
    for m in range(6):
        assert P[m, at] == pytest.approx(P0[0, at] + sh[m, j])
    cf = fts.coincidence_factor(P)
    assert 0 < cf <= 1 and fts.coincidence_factor(np.repeat(P0, 3, axis=0)) == pytest.approx(1.0)


def test_identical_variants_give_the_coincident_hosting(tsf):
    steps = [20000, 20001, 20002]
    tsf.set_base(steps, tsf.base_case(steps))
    kw = np.zeros(fts.STEPS); kw[steps] = 100.0
    order = np.random.default_rng(1).integers(0, len(tsf.cand), 1000)
    h1 = tsf.hosting(order, kw, steps, 1000)
    h3 = tsf.hosting(order, np.repeat(kw[None, :], 3, axis=0), steps, 1000)
    assert h1[0] == pytest.approx(h3[0]) and h1[1] == h3[1]
    # variants that never coincide: half of the hubs load the first step, half the second
    a = np.zeros(fts.STEPS); a[20000] = 100.0
    b = np.zeros(fts.STEPS); b[20001] = 100.0
    hd = tsf.hosting(order, np.stack([a, b]), [20000, 20001], 1000)
    assert hd[0] >= h1[0]
