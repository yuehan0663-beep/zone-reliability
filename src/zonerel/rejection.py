from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
N_RANDOM=5000
SEED=20260629
REJECT_FRAC=.20
FOCUS_SCHEME="LOWO"
# Source: T3c2_random_baseline_diagnosis.py:23
def improvement(mae0: float, mae20: float) -> float:
    """improvement: retained research implementation; paper §3.5, S14. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return (mae0 - mae20) / mae0 * 100 if mae0 > 0 else np.nan

# Source: T3c2_random_baseline_diagnosis.py:47
def unit_table(preds: pd.DataFrame, dataset: str, target: str, method: str) -> pd.DataFrame:
    """unit_table: retained research implementation; paper §3.5, S14. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    rows = []
    sub = preds[preds['dataset'].eq(dataset) & preds['target'].eq(target) & preds['cv_scheme'].eq(FOCUS_SCHEME) & preds['method'].eq(method)]
    for (model, fold), g in sub.groupby(['model', 'fold'], dropna=False):
        y = g['y_true'].to_numpy(float)
        p = g['y_pred'].to_numpy(float)
        abs_err = np.abs(y - p)
        n = len(g)
        n_reject = int(n * REJECT_FRAC)
        eligible = n_reject > 0 and n - n_reject >= 5
        full_mae = float(abs_err.mean())
        if eligible:
            order = np.argsort(-g['composite_risk'].to_numpy(float), kind='mergesort')
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[order[:n_reject]] = False
            risk_mae20 = float(abs_err[keep_mask].mean())
        else:
            risk_mae20 = np.nan
        rows.append({'dataset': dataset, 'target': target, 'method': method, 'model': model, 'fold': fold, 'n': n, 'n_reject_20pct': n_reject, 'n_kept_20pct': n - n_reject, 'eligible_for_20pct': eligible, 'full_MAE': full_mae, 'risk_MAE_20pct_from_predictions': risk_mae20})
    return pd.DataFrame(rows)

# Source: T3c2_random_baseline_diagnosis.py:74
def random_true_uniform(abs_errors_list: list[np.ndarray], n_rejects: list[int], baseline_mae: float, rng: np.random.Generator) -> np.ndarray:
    """random_true_uniform: retained research implementation; paper §3.5, S14. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    imps = np.empty(N_RANDOM, dtype=float)
    for i in range(N_RANDOM):
        kept_maes = []
        for abs_err, n_reject in zip(abs_errors_list, n_rejects):
            n = len(abs_err)
            reject = rng.choice(n, size=n_reject, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[reject] = False
            kept_maes.append(float(abs_err[keep].mean()))
        imps[i] = improvement(baseline_mae, float(np.mean(kept_maes)))
    return imps

RANK_DECIMALS = 9

def rounded_component(values):
    """Author-approved fixed nine-decimal rounding, before risk ranking.

    Preserves a Series index and missing values. Precision is fixed before
    running either environment and must never be selected from results.
    """
    return values.round(RANK_DECIMALS) if isinstance(values, pd.Series) else np.round(np.asarray(values, float), RANK_DECIMALS)

def percentile_rank(values):
    """Nine-decimal components ranked with average ties, divided by n.

    For global intervals, within-fold width is constant and contributes no
    ordering information; the other components determine the risk order.
    pandas average percentile rank is equivalent to rankdata(average)/n for
    finite inputs. Missing values retain the historical pandas behavior.
    """
    values = values if isinstance(values, pd.Series) else pd.Series(values)
    return rounded_component(values).rank(pct=True, method='average')

def composite_risk(frame,weights=(.55,.25,.10,.10)):
    """Fold-local risk after fixed rounding and average ties (§3.5 erratum).

    Weights remain .55/.25/.10/.10, quality remains constant 0.1. A global
    interval's width is constant within its fold, so it supplies no ordering
    information; the remaining components determine the ranking. Interval
    bounds and width arithmetic retain the original implementation.
    """
    return sum(percentile_rank(frame[c])*w for c,w in zip(['interval_width','domain_knn','missing_risk','quality_risk'],weights))

def random_control(cpreds):
    """S14 true-uniform random comparison on the same eligible model-folds.

    Retains 5,000 iterations and seed 20260629. Excludes units that reject
    zero rows or retain fewer than five; both baseline and risk MAE use the
    same eligible units. Returns one summary per target and method.
    """
    rows=[];rng=np.random.default_rng(SEED)
    keys=cpreds[cpreds.cv_scheme=='LOWO'][['dataset','target','method']].drop_duplicates().sort_values(['dataset','method'])
    for dataset,target,method in keys.itertuples(index=False,name=None):
        ut=unit_table(cpreds,dataset,target,method);eligible=ut[ut.eligible_for_20pct]
        if eligible.empty:continue
        baseline=float(eligible.full_MAE.mean());retained=float(eligible.risk_MAE_20pct_from_predictions.mean())
        sub=cpreds[(cpreds.dataset==dataset)&(cpreds.target==target)&(cpreds.cv_scheme=='LOWO')&(cpreds.method==method)]
        errors=[];counts=[]
        for r in eligible.itertuples(index=False):
            g=sub[(sub.model==r.model)&(sub.fold==r.fold)]
            errors.append(np.abs(g.y_true.to_numpy(float)-g.y_pred.to_numpy(float)));counts.append(int(r.n_reject_20pct))
        samples=random_true_uniform(errors,counts,baseline,rng)
        effect=improvement(baseline,retained)
        rows.append(dict(dataset=dataset,target=target,cv_scheme='LOWO',method=method,reject_frac=.2,baseline_MAE_same_eligible_units=baseline,risk_MAE_20pct_same_eligible_units=retained,risk_improvement_pct_same_eligible_units=effect,random_improvement_mean_pct=float(samples.mean()),random_improvement_p95_pct=float(np.quantile(samples,.95)),empirical_p_random_ge_risk=float(np.mean(samples>=effect))))
    return pd.DataFrame(rows)
