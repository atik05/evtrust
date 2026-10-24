import pandas as pd
from evtrust.splits import random_split, chronological_split, user_disjoint_split, validate_split


def frame(n=120):
    t=pd.date_range('2020-01-01',periods=n,freq='12h',tz='UTC')
    return pd.DataFrame({
        'connection_time':t,
        'disconnect_time':t+pd.Timedelta(hours=4),
        'user_id':[f'u{i%20}' for i in range(n)],
        'site':['a']*n,
    })


def test_random_no_index_overlap():
    df=frame(); sp=random_split(df,7); r=validate_split(df,sp)
    assert all(v==0 for v in r['index_overlap'].values())


def test_user_disjoint_zero_user_overlap():
    df=frame(); sp=user_disjoint_split(df,7); r=validate_split(df,sp)
    assert r['user_overlap']==0


def test_chronological_purge_matures_labels():
    df=frame(); sp=chronological_split(df,embargo_hours=4)
    if len(sp.train) and len(sp.calibration):
        boundary=df.loc[sp.calibration,'connection_time'].min()
        assert (df.loc[sp.train,'disconnect_time'] < boundary-pd.Timedelta(hours=4)).all()
