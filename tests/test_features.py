import pandas as pd
from evtrust.features import add_driver_features, add_history_features


def test_later_user_input_is_excluded():
    t=pd.Timestamp('2020-01-01 08:00',tz='UTC')
    df=pd.DataFrame({
        'connection_time':[t],
        'user_inputs':[[
            {'modifiedAt':t.isoformat(),'requestedDeparture':(t+pd.Timedelta(hours=2)).isoformat(),'kWhRequested':10},
            {'modifiedAt':(t+pd.Timedelta(hours=1)).isoformat(),'requestedDeparture':(t+pd.Timedelta(hours=9)).isoformat(),'kWhRequested':99},
        ]]
    })
    out=add_driver_features(df)
    assert out.loc[0,'requested_energy_kwh']==10
    assert abs(out.loc[0,'requested_duration_hours']-2)<1e-9


def test_history_uses_completed_sessions_only():
    t=pd.Timestamp('2020-01-01 08:00',tz='UTC')
    df=pd.DataFrame({
        'connection_time':[t,t+pd.Timedelta(hours=1),t+pd.Timedelta(hours=5)],
        'disconnect_time':[t+pd.Timedelta(hours=4),t+pd.Timedelta(hours=2),t+pd.Timedelta(hours=6)],
        'duration_hours':[4.,1.,1.], 'kwh_delivered':[10.,3.,4.],
        'user_id':['u','u','u'],'site':['s']*3,'station_id':['a']*3
    })
    out=add_history_features(df)
    # At second connection, first session is not finished, so no user history is allowed.
    assert out.loc[1,'user_prior_count']==0
    # By third connection, both earlier sessions are complete.
    assert out.loc[2,'user_prior_count']==2
