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

from .constants import MODEL_ORDER, FAST_MODELS
try:
    from xgboost import XGBRegressor
except ImportError:XGBRegressor=None
try:
    from lightgbm import LGBMRegressor
except ImportError:LGBMRegressor=None
try:
    from catboost import CatBoostRegressor
except ImportError:CatBoostRegressor=None
# Source: run_full_condition_evaluation.py:99
class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    """GroupMeanRegressor: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""

    def __init__(self, group_col='zone'):
        """__init__: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
        self.group_col = group_col

    def fit(self, X, y):
        """fit: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
        x = pd.DataFrame(X).copy()
        x['__y__'] = np.asarray(y, float)
        self.global_mean_ = float(np.nanmean(x['__y__']))
        self.group_means_ = x.groupby(self.group_col)['__y__'].mean().to_dict() if self.group_col in x.columns else {}
        return self

    def predict(self, X):
        """predict: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
        x = pd.DataFrame(X)
        if self.group_col not in x.columns:
            return np.full(len(x), self.global_mean_)
        return x[self.group_col].map(self.group_means_).fillna(self.global_mean_).to_numpy(float)

# Source: run_full_condition_evaluation.py:109
def preprocessor(num, cat):
    """preprocessor: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    trs = []
    if num:
        trs.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num))
    if cat:
        trs.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat))
    return ColumnTransformer(trs, remainder='drop')

# Source: run_full_condition_evaluation.py:115
def regressor_pool(model_names=None):
    """regressor_pool: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    models = {'train_mean': DummyRegressor(strategy='mean'), 'zone_mean': GroupMeanRegressor('zone'), 'LinearRegression': LinearRegression(), 'Ridge': Ridge(alpha=1.0), 'Lasso': Lasso(alpha=0.01, max_iter=20000, random_state=42), 'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=42), 'Huber': HuberRegressor(max_iter=2000), 'SVR_RBF': SVR(C=3.0, gamma='scale', epsilon=0.05), 'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'), 'RF': RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1), 'ExtraTrees': ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1), 'GBDT': GradientBoostingRegressor(random_state=42, n_estimators=180, learning_rate=0.04, max_depth=2), 'HistGBDT': HistGradientBoostingRegressor(random_state=42, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1)}
    if XGBRegressor is not None:
        models['XGBoost'] = XGBRegressor(n_estimators=180, max_depth=2, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, objective='reg:squarederror', random_state=42, n_jobs=2, verbosity=0)
    if LGBMRegressor is not None:
        models['LightGBM'] = LGBMRegressor(n_estimators=180, learning_rate=0.04, num_leaves=15, min_child_samples=5, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=2, verbose=-1)
    if CatBoostRegressor is not None:
        models['CatBoost'] = CatBoostRegressor(iterations=180, depth=3, learning_rate=0.04, loss_function='RMSE', random_seed=42, verbose=False, allow_writing_files=False)
    if model_names is not None:
        absent=set(model_names)-set(models)
        if absent:raise ImportError("Unavailable model libraries: "+", ".join(sorted(absent)))
        return {name:models[name] for name in model_names}
    return models

# Source: run_full_condition_evaluation.py:139
def make_pipe(name, model, num, cat):
    """make_pipe: retained research implementation; paper §3.3. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    if name == 'zone_mean':
        return clone(model)
    return Pipeline([('pre', preprocessor(num, cat)), ('model', clone(model))])
