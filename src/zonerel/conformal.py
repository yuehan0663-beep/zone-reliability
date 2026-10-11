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
from .constants import NUM_COLS, CAT_COLS, LOG_COLS
from .models import regressor_pool, make_pipe
from .metrics import metric_row, choose_models, spearman_safe
from .splits import split_iter
from .rejection import composite_risk, percentile_rank, rounded_component

def calibration_split(df, outer_train):
    """Partition outer training positions into proper fit/calibration (§3.4).

    Use 25% of sorted training wells without replacement (seed 42), or
    original ordered 75/25 row fallback when fewer than four wells exist.
    No held-out test row or test label can enter calibration.
    """
    wells = np.array(sorted(df.iloc[outer_train]['well'].dropna().unique()))
    rng = np.random.default_rng(42)
    if len(wells) >= 4:
        cal_w = set(rng.choice(wells, size=max(1, int(len(wells) * 0.25)), replace=False))
        cal = np.array([i for i in outer_train if df.iloc[i]['well'] in cal_w])
        train = np.array([i for i in outer_train if df.iloc[i]['well'] not in cal_w])
    else:
        cut = int(len(outer_train) * 0.75)
        train = outer_train[:cut]
        cal = outer_train[cut:]
    return (train, cal)

def quantile_90(residuals, finite_sample=False):
    """Default empirical linear 0.90 quantile; optional n+1 order statistic.

    This retains the original paper convention by default (§3.4, S24a).
    """
    a = np.asarray(residuals, float)
    if not len(a):
        raise ValueError('Empty calibration residuals')
    if not finite_sample:
        return float(np.quantile(a, 0.9))
    k = int(np.ceil((len(a) + 1) * 0.9))
    return float(np.sort(a)[k - 1]) if k <= len(a) else float('inf')

def transformed(pipe, x):
    """transformed: retained research implementation; paper §3.4–3.5. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return pipe.named_steps['pre'].transform(x) if isinstance(pipe, Pipeline) else x.select_dtypes(include=[np.number]).fillna(0).to_numpy()

def conformal_rejection(datasets, summary, finite_sample=False, extra_models=None):
    """Split-conformal and rejection; paper §3.4–3.5. risk_inflated is the distance-inflated alias. Five-neighbor distances are min-max normalized on current test-fold input features, never labels; calibration comes only from outer training. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    conf = []
    cpreds = []
    rej = []
    models = regressor_pool(extra_models=extra_models)
    fracs = [i / 100 for i in range(0, 41, 5)]
    for name, target, df in datasets:
        mnames = choose_models(summary, name)
        num = [c for c in NUM_COLS if c in df.columns]
        cat = [c for c in CAT_COLS if c in df.columns]
        x = df[num + cat].copy()
        y = df[target].astype(float).to_numpy()
        for mn in mnames:
            for scheme in ['LOWO', 'LOZO']:
                for fold, tr_all, te in split_iter(df, scheme):
                    tr, cal = calibration_split(df, tr_all)
                    if len(tr) < 10 or len(cal) < 5:
                        continue
                    pipe = make_pipe(mn, models[mn], num, cat)
                    pipe.fit(x.iloc[tr], y[tr])
                    cal_pred = np.asarray(pipe.predict(x.iloc[cal]), float)
                    pred = np.asarray(pipe.predict(x.iloc[te]), float)
                    resid = np.abs(y[cal] - cal_pred)
                    q90 = quantile_90(resid, finite_sample)
                    cal_zone = df.iloc[cal]['zone'].astype(str).to_numpy()
                    zq = {z: quantile_90(resid[cal_zone == z], finite_sample) for z in np.unique(cal_zone) if (cal_zone == z).sum() >= 3}
                    qzone = np.array([zq.get(str(z), q90) for z in df.iloc[te]['zone']])
                    Xtr = transformed(pipe, x.iloc[tr])
                    Xte = transformed(pipe, x.iloc[te])
                    dist = NearestNeighbors(n_neighbors=min(5, len(tr))).fit(Xtr).kneighbors(Xte, return_distance=True)[0].mean(axis=1)
                    dn = (dist - dist.min()) / (dist.max() - dist.min() + 1e-09)
                    miss = df.iloc[te][[c for c in LOG_COLS if c in df.columns]].isna().sum(axis=1).to_numpy(float) / max(1, len(LOG_COLS))
                    qual = np.zeros(len(te)) + 0.1
                    for method, q in [('global', np.full(len(te), q90)), ('zone_conditional', qzone), ('risk_inflated', q90 * (1 + 0.5 * dn))]:
                        lo = pred - q
                        hi = pred + q
                        covered = (y[te] >= lo) & (y[te] <= hi)
                        width = hi - lo
                        rr = {'dataset': name, 'target': target, 'model': mn, 'cv_scheme': scheme, 'fold': fold, 'method': method, 'n_train': len(tr), 'n_cal': len(cal), 'n_test': len(te), 'coverage': float(covered.mean()), 'mean_width': float(width.mean())}
                        rr.update(metric_row(y[te], pred))
                        rr['domain_abs_error_spearman'] = spearman_safe(dist, np.abs(y[te] - pred))
                        conf.append(rr)
                        base = pd.DataFrame({'dataset': name, 'target': target, 'model': mn, 'cv_scheme': scheme, 'fold': fold, 'method': method, 'sample_id': df.iloc[te]['sample_id'].to_numpy(), 'well': df.iloc[te]['well'].to_numpy(), 'zone': df.iloc[te]['zone'].to_numpy(), 'layer': df.iloc[te]['layer'].to_numpy(), 'y_true': y[te], 'y_pred': pred, 'lo': lo, 'hi': hi, 'covered': covered, 'interval_width': width, 'domain_knn': dist, 'missing_risk': miss, 'quality_risk': qual, 'abs_error': np.abs(y[te] - pred)})
                        base['composite_risk'] = composite_risk(base)
                        cpreds.append(base)
                        for risk in ['interval_width', 'domain_knn', 'composite_risk']:
                            score = base[risk] if risk == 'composite_risk' else rounded_component(base[risk])
                            sb = base.assign(_sorting_risk=score).sort_values('_sorting_risk', ascending=False).drop(columns='_sorting_risk').reset_index(drop=True)
                            for frac in fracs:
                                kept = sb.iloc[int(len(sb) * frac):]
                                if len(kept) < 5:
                                    continue
                                row = {'dataset': name, 'target': target, 'model': mn, 'cv_scheme': scheme, 'fold': fold, 'method': method, 'risk_score': risk, 'reject_frac': frac, 'kept_n': len(kept), 'coverage': float(kept.covered.mean()), 'mean_width': float(kept.interval_width.mean())}
                                row.update(metric_row(kept.y_true, kept.y_pred))
                                rej.append(row)
    return (pd.DataFrame(conf), pd.concat(cpreds, ignore_index=True), pd.DataFrame(rej))

def reliability(cpreds):
    """reliability: retained research implementation; paper §3.4–3.5. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    df = cpreds[cpreds.method == 'global'].copy()
    df['width_pct'] = df.groupby(['dataset', 'target']).interval_width.transform(percentile_rank)
    df['pred_pct'] = df.groupby(['dataset', 'target']).y_pred.rank(pct=True)
    df['reject_flag_20pct'] = df.groupby(['dataset', 'target']).composite_risk.transform(lambda s: s >= s.quantile(0.8))
    df['confidence_class'] = np.select([df.reject_flag_20pct, df.width_pct >= 0.67, df.width_pct <= 0.33], ['reject/out-of-support', 'uncertain', 'high-confidence'], default='medium-confidence')
    df['sweetspot_confidence_case'] = np.select([(df.pred_pct >= 0.67) & (df.confidence_class == 'high-confidence'), (df.pred_pct >= 0.67) & (df.confidence_class != 'high-confidence')], ['high_prediction_high_confidence', 'high_prediction_low_confidence'], default='other')
    zone = df.groupby(['dataset', 'target', 'zone']).agg(n=('sample_id', 'count'), mean_pred=('y_pred', 'mean'), mean_width=('interval_width', 'mean'), reject_rate=('reject_flag_20pct', 'mean'), coverage=('covered', 'mean')).reset_index()
    well = df.groupby(['dataset', 'target', 'well', 'zone']).agg(n=('sample_id', 'count'), mean_pred=('y_pred', 'mean'), mean_width=('interval_width', 'mean'), reject_rate=('reject_flag_20pct', 'mean'), high_pred_low_conf_n=('sweetspot_confidence_case', lambda s: int((s == 'high_prediction_low_confidence').sum()))).reset_index()
    return (df, zone, well)
