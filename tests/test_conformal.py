import numpy as np
from evtrust.conformal import split_conformal_calibrate, split_conformal_interval, cqr_calibrate, cqr_interval


def test_split_conformal_shapes_and_monotonicity():
    y=np.array([0.,1.,2.,3.,4.]); p=np.array([0.,1.2,1.8,2.5,4.5])
    q=split_conformal_calibrate(y,p,.2)
    lo,hi=split_conformal_interval(p,q)
    assert q>=0 and lo.shape==y.shape and np.all(hi>=lo)


def test_cqr_calibration_nonnegative_when_interval_misses():
    y=np.array([0.,2.,4.]); lo=np.array([.5,1.5,3.5]); hi=np.array([1.,1.8,3.8])
    q=cqr_calibrate(y,lo,hi,.1)
    l,u=cqr_interval(lo,hi,q)
    assert q>=0 and np.all(u>=l)
