from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .experiments import make_splits, _split_frames, USER_HISTORY, STATION_HISTORY
from .models import fit_catboost
from .metrics import regression_metrics


def run_grouped_ablation(df: pd.DataFrame, cfg: dict, out='results/metrics/grouped_ablation.csv') -> pd.DataFrame:
    seed=int(cfg['tuning']['canonical_seed'])
    numeric_base=list(cfg['features']['numeric']); cat_base=list(cfg['features']['categorical'])
    tuned_path=Path('results/metrics/tuned_hyperparameters.json')
    tuned=json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    groups={
        'full': set(),
        'no_driver': set(cfg['features']['driver_features']),
        'no_user_history': set(USER_HISTORY),
        'no_station_history': set(STATION_HISTORY),
        'calendar_only': set(numeric_base+cat_base) - {'arrival_hour_sin','arrival_hour_cos','weekday_sin','weekday_cos','month_sin','month_cos','is_weekend'},
    }
    targets={'duration':'duration_hours','energy':'kwh_delivered'}
    rows=[]
    for sp in make_splits(df,cfg,seed):
        tr0,_,va0,te0=_split_frames(df,sp)
        for target,ycol in targets.items():
            tr=tr0[tr0[ycol].notna()]; va=va0[va0[ycol].notna()]; te=te0[te0[ycol].notna()]
            if min(len(tr),len(va),len(te))<20: continue
            key=f'{target}|{sp.name}'
            params=tuned.get(key,{}).get('catboost',cfg['models']['catboost'])
            for name,drop in groups.items():
                num=[c for c in numeric_base if c not in drop]
                cat=[c for c in cat_base if c not in drop]
                if not num and not cat: continue
                m=fit_catboost(tr,tr[ycol].to_numpy(float),va,va[ycol].to_numpy(float),num,cat,params,seed)
                pred=m.predict(te)
                met=regression_metrics(te[ycol],pred)
                rows.append({'target':target,'protocol':sp.name.split(':')[0],'split_name':sp.name,
                             'target_site':sp.metadata.get('target_site'),'ablation':name,'n':len(te),**met})
    outdf=pd.DataFrame(rows)
    if not outdf.empty:
        full=outdf[outdf.ablation=='full'][['target','split_name','mae']].rename(columns={'mae':'full_mae'})
        outdf=outdf.merge(full,on=['target','split_name'],how='left')
        outdf['delta_mae_vs_full']=outdf.mae-outdf.full_mae
        Path(out).parent.mkdir(parents=True,exist_ok=True); outdf.to_csv(out,index=False)
    return outdf
