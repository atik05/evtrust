from __future__ import annotations
import copy
import optuna
from sklearn.metrics import mean_absolute_error
from .models import fit_ridge, fit_catboost

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_ridge(train, ytr, val, yva, numeric, categorical, n_trials: int, seed: int) -> dict:
    def objective(trial):
        alpha=trial.suggest_float('alpha',1e-3,1e3,log=True)
        m=fit_ridge(train,ytr,numeric,categorical,alpha)
        return mean_absolute_error(yva,m.predict(val))
    st=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective,n_trials=n_trials)
    return {'alpha':float(st.best_params['alpha'])}


def tune_catboost(train,ytr,val,yva,numeric,categorical,base:dict,n_trials:int,seed:int)->dict:
    def objective(trial):
        p=copy.deepcopy(base)
        p.update({
            'iterations':trial.suggest_int('iterations',800,3000,step=200),
            'depth':trial.suggest_int('depth',6,10),
            'learning_rate':trial.suggest_float('learning_rate',.015,.10,log=True),
            'l2_leaf_reg':trial.suggest_float('l2_leaf_reg',1.0,20.0,log=True),
            'random_strength':trial.suggest_float('random_strength',.2,3.0,log=True),
            'verbose':False,
        })
        m=fit_catboost(train,ytr,val,yva,numeric,categorical,p,seed)
        return mean_absolute_error(yva,m.predict(val))
    st=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective,n_trials=n_trials)
    p=copy.deepcopy(base); p.update(st.best_params)
    return p


def tune_torch(train,ytr,val,yva,numeric,categorical,base:dict,n_trials:int,seed:int,tuning_epochs:int)->dict:
    from .torch_tabular import fit_torch_ft  # lazy import, see experiments.py

    def objective(trial):
        p=copy.deepcopy(base)
        p.update({
            'd_token':trial.suggest_categorical('d_token',[64,96,128,160]),
            'n_heads':trial.suggest_categorical('n_heads',[4,8]),
            'n_layers':trial.suggest_int('n_layers',2,5),
            'dropout':trial.suggest_float('dropout',.0,.25),
            'lr':trial.suggest_float('lr',2e-4,2e-3,log=True),
            'weight_decay':trial.suggest_float('weight_decay',1e-6,2e-3,log=True),
            'epochs':tuning_epochs,
            'patience':max(6,min(base.get('patience',15),10)),
        })
        # Ensure d_token is divisible by n_heads.
        if p['d_token'] % p['n_heads'] != 0:
            return 1e9
        m=fit_torch_ft(train,ytr,val,yva,numeric,categorical,p,seed)
        return mean_absolute_error(yva,m.predict(val))
    st=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective,n_trials=n_trials)
    p=copy.deepcopy(base); p.update(st.best_params)
    return p
