from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd

from .experiments import make_splits
from .stats import continuous_shift, js_divergence, paired_day_block_bootstrap, holm_adjust


def distribution_shift_report(df: pd.DataFrame, cfg: dict, seed: int | None = None) -> pd.DataFrame:
    seed = int(seed if seed is not None else cfg['tuning']['canonical_seed'])
    num = cfg['features']['numeric']
    cat = cfg['features']['categorical']
    rows=[]
    for sp in make_splits(df,cfg,seed):
        tr=df.loc[sp.train]; te=df.loc[sp.test]
        for c in num:
            if c not in df: continue
            raw=continuous_shift(tr[c],te[c])
            q1,q3=pd.to_numeric(tr[c],errors='coerce').quantile([.25,.75])
            scale=float(q3-q1)
            rows.append({'split_name':sp.name,'protocol':sp.name.split(':')[0],'target_site':sp.metadata.get('target_site'),
                         'feature':c,'feature_type':'numeric','shift':raw,'normalized_shift':raw/scale if scale>0 else np.nan})
        for c in cat:
            if c not in df: continue
            js=js_divergence(tr[c],te[c])
            rows.append({'split_name':sp.name,'protocol':sp.name.split(':')[0],'target_site':sp.metadata.get('target_site'),
                         'feature':c,'feature_type':'categorical','shift':js,'normalized_shift':js})
    return pd.DataFrame(rows)


def conditional_coverage_report(pred: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    p=pred[pred.lower.notna() & pred.upper.notna()].copy()
    if p.empty: return pd.DataFrame()
    extra=sessions[['user_prior_count','connection_time']].copy(); extra['row_id']=extra.index
    p=p.merge(extra,on='row_id',how='left',suffixes=('','_session'))
    p['hour_bin']=pd.cut(p['connection_time_session'].dt.hour,[-1,5,11,17,23],labels=['00-05','06-11','12-17','18-23'])
    p['user_history_bin']=pd.cut(p['user_prior_count'].fillna(0),[-1,0,2,9,np.inf],labels=['0','1-2','3-9','10+'])
    p['covered']=((p.y_true>=p.lower)&(p.y_true<=p.upper)).astype(int)
    rows=[]
    for dimension in ['site','hour_bin','user_history_bin']:
        for keys,g in p.groupby(['target','protocol','model',dimension],dropna=False,observed=True):
            target,protocol,model,group=keys
            if len(g)<20: continue
            nominal=1-float(g.alpha.dropna().iloc[0])
            rows.append({'target':target,'protocol':protocol,'model':model,'dimension':dimension,'group':str(group),
                         'n':len(g),'coverage':g.covered.mean(),'nominal':nominal,'coverage_error':abs(g.covered.mean()-nominal),
                         'mean_width':(g.upper-g.lower).mean()})
    return pd.DataFrame(rows)


def paired_model_comparisons(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Average predictions across seeds before inference: seeds quantify algorithmic variation, not sample size.
    base=(pred.groupby(['target','protocol','split_name','target_site','row_id','date','model'],dropna=False,as_index=False)
              .agg(y_true=('y_true','first'),y_pred=('y_pred','mean')))
    rows=[]
    for keys,g in base.groupby(['target','protocol','split_name','target_site'],dropna=False):
        target,protocol,split_name,target_site=keys
        models=sorted(g.model.unique())
        family=[]
        for a,b in combinations(models,2):
            ga=g[g.model.isin([a,b])]
            # Only compare if the same evaluation rows exist for both models.
            counts=ga.groupby('row_id').model.nunique()
            common=set(counts[counts==2].index)
            z=ga[ga.row_id.isin(common)]
            if len(common)<30: continue
            r=paired_day_block_bootstrap(z,a,b,'mae',cfg['bootstrap']['n_boot'],cfg['project']['seed'],cfg['bootstrap']['confidence'])
            tail=r.get('p_two_sided', 2*min(r['p_a_better'],1-r['p_a_better']))
            family.append({**r,'target':target,'protocol':protocol,'split_name':split_name,'target_site':target_site,
                           'n_paired_rows':len(common),'bootstrap_tail_probability':tail})
        if family:
            adj=holm_adjust([x['bootstrap_tail_probability'] for x in family])
            for x,a in zip(family,adj): x['holm_adjusted_tail_probability']=float(a)
            rows.extend(family)
    return pd.DataFrame(rows)


def site_transfer_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    s=metrics[metrics.protocol=='site_disjoint'].copy()
    if s.empty: return pd.DataFrame()
    by_site=s.groupby(['target','model','target_site'],as_index=False).agg(mae=('mae','mean'),mae_seed_sd=('mae','std'))
    out=by_site.groupby(['target','model'],as_index=False).agg(macro_mae=('mae','mean'),worst_site_mae=('mae','max'),best_site_mae=('mae','min'))
    return out.merge(by_site.groupby(['target','model'])['target_site'].nunique().rename('n_sites'),on=['target','model'])
