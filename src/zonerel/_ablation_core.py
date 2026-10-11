"""Retained Table S18 numerical functions; no import-time file access."""
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
SCHEMES = ['random', 'LOWO', 'LOZO']
from .domains import ordered_zones
def preprocessor(num, cat):
    trs=[]
    if num:
        trs.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num))
    if cat:
        trs.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat))
    return ColumnTransformer(trs, remainder='drop')

def model_obj(name):
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1)
    if name == 'RF':
        return RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1)
    raise ValueError(name)

def split_iter(df, scheme):
    if scheme == 'random':
        for i,(tr,te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(df), 1):
            yield f'random_{i}', np.asarray(tr), np.asarray(te)
    elif scheme == 'LOWO':
        for w in sorted(df['well'].dropna().unique()):
            te=np.where(df['well'].to_numpy()==w)[0]; tr=np.where(df['well'].to_numpy()!=w)[0]
            if len(te) and len(tr)>=10:
                yield str(w), tr, te
    elif scheme == 'LOZO':
        for z in ordered_zones(df, validate_lozo=True):
            te=np.where(df['zone'].to_numpy()==z)[0]; tr=np.where(df['zone'].to_numpy()!=z)[0]
            if len(te) and len(tr)>=10:
                yield str(z), tr, te

def spearman_safe(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    m=np.isfinite(y)&np.isfinite(p)
    if m.sum()<3 or len(np.unique(y[m]))<2 or len(np.unique(p[m]))<2:
        return np.nan
    r=spearmanr(y[m],p[m],nan_policy='omit').correlation
    return float(r) if np.isfinite(r) else np.nan

def metric_row(y,p):
    return {
        'MAE': float(mean_absolute_error(y,p)),
        'RMSE': float(math.sqrt(mean_squared_error(y,p))),
        'R2': float(r2_score(y,p)) if len(y)>1 else np.nan,
        'Spearman': spearman_safe(y,p)
    }

def run_eval(df, dataset, target, target_label, model_name, feature_set, num_cols, cat_cols, model=None):
    rows=[]; preds=[]
    x=df[num_cols + cat_cols].copy()
    y=df[target].astype(float).to_numpy()
    model=model_obj(model_name) if model is None else model
    for scheme in SCHEMES:
        for fold,tr,te in split_iter(df, scheme):
            pipe=Pipeline([('pre', preprocessor(num_cols, cat_cols)), ('model', clone(model))])
            pipe.fit(x.iloc[tr], y[tr])
            pr=np.asarray(pipe.predict(x.iloc[te]), float)
            row={'dataset':dataset,'target':target_label,'model':model_name,'feature_set':feature_set,'cv_scheme':scheme,'fold':fold,'n_train':len(tr),'n_test':len(te)}
            row.update(metric_row(y[te], pr)); rows.append(row)
            p=pd.DataFrame({'dataset':dataset,'target':target_label,'model':model_name,'feature_set':feature_set,'cv_scheme':scheme,'fold':fold,'y_true':y[te],'y_pred':pr})
            preds.append(p)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)

def summarize(folds, preds):
    out=[]
    for keys,g in folds.groupby(['dataset','target','model','feature_set','cv_scheme'], dropna=False):
        gp=preds
        for col,val in zip(['dataset','target','model','feature_set','cv_scheme'], keys):
            gp=gp[gp[col].eq(val)]
        w=g['n_test'].to_numpy(float)
        out.append({
            'dataset':keys[0], 'target':keys[1], 'model':keys[2], 'feature_set':keys[3], 'cv_scheme':keys[4],
            'folds':int(len(g)), 'n_test':int(g['n_test'].sum()),
            'MAE_w':float(np.average(g['MAE'], weights=w)),
            'RMSE_w':float(np.average(g['RMSE'], weights=w)),
            'fold_mean_R2_deprecated':float(g['R2'].mean()),
            'Spearman_mean':float(g['Spearman'].mean()),
            'pooled_R2':float(r2_score(gp['y_true'], gp['y_pred'])) if len(gp)>1 else np.nan,
            'pooled_n':int(len(gp)),
        })
    return pd.DataFrame(out)
