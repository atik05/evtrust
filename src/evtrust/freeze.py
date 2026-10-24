from __future__ import annotations
from pathlib import Path
import pandas as pd
from .experiments import make_splits


def freeze_session_splits(df: pd.DataFrame, cfg: dict, out_dir='results/splits') -> list[Path]:
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    paths=[]
    for seed in cfg['project']['five_seeds']:
        for sp in make_splits(df,cfg,int(seed)):
            rows=[]
            for part,idx in [('train',sp.train),('calibration',sp.calibration),('validation',sp.validation),('test',sp.test)]:
                z=df.loc[idx,['session_id','site','user_id','connection_time','disconnect_time']].copy()
                z['partition']=part; rows.append(z)
            table=pd.concat(rows,ignore_index=True)
            safe=sp.name.replace(':','__')
            path=out_dir/f'seed_{seed}__{safe}.csv'
            table.to_csv(path,index=False)
            paths.append(path)
    return paths
