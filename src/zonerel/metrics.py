from __future__ import annotations
import math, random
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# Source: run_full_condition_evaluation.py:68
def fmt(x):
    """fmt: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    try:
        if x is None or not np.isfinite(float(x)):
            return ''
        return f'{float(x):.6g}'
    except Exception:
        return str(x)

# Source: run_full_condition_evaluation.py:75
def spearman_safe(y_true, y_pred):
    """spearman_safe: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3 or len(np.unique(y[m])) < 2 or len(np.unique(p[m])) < 2:
        return np.nan
    r = spearmanr(y[m], p[m], nan_policy='omit').correlation
    return float(r) if np.isfinite(r) else np.nan

# Source: run_full_condition_evaluation.py:81
def rmse(y, p):
    """rmse: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return math.sqrt(mean_squared_error(y, p))

# Source: run_full_condition_evaluation.py:82
def metric_row(y, p):
    """metric_row: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return {'MAE': float(mean_absolute_error(y, p)), 'RMSE': float(rmse(y, p)), 'R2': float(r2_score(y, p)) if len(y) > 1 else np.nan, 'Spearman': spearman_safe(y, p)}

# Source: run_full_condition_evaluation.py:191
def summarize(folds, preds):
    """summarize: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    rows = []
    for keys, g in folds.groupby(['dataset', 'target', 'model', 'cv_scheme'], dropna=False):
        w = g['n_test'].to_numpy(float)
        gp = preds[(preds['dataset'] == keys[0]) & (preds['target'] == keys[1]) & (preds['model'] == keys[2]) & (preds['cv_scheme'] == keys[3])]
        rows.append(dict(zip(['dataset', 'target', 'model', 'cv_scheme'], keys), folds=len(g), n_test=int(g['n_test'].sum()), MAE_w=float(np.average(g['MAE'], weights=w)), RMSE_w=float(np.average(g['RMSE'], weights=w)), fold_mean_R2_deprecated=float(g['R2'].mean()), Spearman_mean=float(g['Spearman'].mean()), pooled_R2=float(r2_score(gp['y_true'], gp['y_pred'])) if len(gp) > 1 else np.nan, pooled_n=len(gp)))
    return pd.DataFrame(rows)

# Source: run_full_condition_evaluation.py:198
def choose_models(summary, dataset, n=3):
    """choose_models: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    sub = summary[(summary.dataset == dataset) & summary.cv_scheme.isin(['LOWO', 'LOZO']) & ~summary.model.isin(['train_mean', 'zone_mean'])]
    rank = sub.groupby('model').agg(MAE=('MAE_w', 'mean'), S=('Spearman_mean', 'mean')).reset_index().sort_values(['MAE', 'S'], ascending=[True, False])
    return rank.head(n)['model'].tolist()
