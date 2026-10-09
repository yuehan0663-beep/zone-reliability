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

from .constants import NUM_COLS,CAT_COLS,MODEL_ORDER
from .models import regressor_pool,make_pipe
from .metrics import metric_row
from .splits import split_iter
# Source: run_full_condition_evaluation.py:168
def eval_baselines(datasets, model_names=None, progress=None):
    """eval_baselines: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    rows = []
    preds = []
    errors = []
    models = regressor_pool(model_names)
    for name, target, df in datasets:
        num = [c for c in NUM_COLS if c in df.columns]
        cat = [c for c in CAT_COLS if c in df.columns]
        x_model = df[num + cat].copy()
        x_all = df[num + cat + ['zone']].copy()
        y = df[target].astype(float).to_numpy()
        for model_name in (model_names or MODEL_ORDER):
            if progress is not None:
                progress(name, model_name)
            if model_name not in models:
                errors.append({'dataset': name, 'target': target, 'model': model_name, 'status': 'NOT RUN', 'reason': 'library unavailable'})
                continue
            for scheme in ['random', 'LOWO', 'LOZO']:
                for fold, tr, te in split_iter(df, scheme):
                    try:
                        pipe = make_pipe(model_name, models[model_name], num, cat)
                        Xtr = (x_all if model_name == 'zone_mean' else x_model).iloc[tr]
                        Xte = (x_all if model_name == 'zone_mean' else x_model).iloc[te]
                        pipe.fit(Xtr, y[tr])
                        pr = np.asarray(pipe.predict(Xte), float)
                        row = {'dataset': name, 'target': target, 'model': model_name, 'cv_scheme': scheme, 'fold': fold, 'n_train': len(tr), 'n_test': len(te), 'test_zones': ','.join(sorted(df.iloc[te]['zone'].astype(str).unique())), 'test_wells': ','.join(sorted(df.iloc[te]['well'].astype(str).unique()))}
                        row.update(metric_row(y[te], pr))
                        rows.append(row)
                        keep = [c for c in ['sample_id', 'well', 'zone', 'layer', 'depth_md', 'cleaning_confidence'] if c in df.columns]
                        p = df.iloc[te][keep].copy()
                        p['dataset'] = name
                        p['target'] = target
                        p['model'] = model_name
                        p['cv_scheme'] = scheme
                        p['fold'] = fold
                        p['y_true'] = y[te]
                        p['y_pred'] = pr
                        p['abs_error'] = np.abs(y[te] - pr)
                        preds.append(p)
                    except Exception as e:
                        errors.append({'dataset': name, 'target': target, 'model': model_name, 'cv_scheme': scheme, 'fold': fold, 'status': 'ERROR', 'reason': repr(e)})
    return (pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.DataFrame(errors))
