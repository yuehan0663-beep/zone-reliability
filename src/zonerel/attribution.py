"""Fold-contained attribution diagnostics (Methods 3.4, Tables S18/S22).

TreeExplainer is evaluated on training rows. Signed one-hot contributions
are summed per original categorical feature before taking absolute values.
"""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from . import _shap_core as sc
from . import _ablation_core as ac
from .io import validate_dataset, target_datasets
from .models import (validate_regressor, ExtraTreesRegressor, RandomForestRegressor,
                     GradientBoostingRegressor, HistGradientBoostingRegressor,
                     XGBRegressor, LGBMRegressor, CatBoostRegressor)


def validate_backbones(backbones):
    """Require unfitted TreeExplainer-compatible tree ensembles; no fallback."""
    from collections.abc import Mapping
    if not isinstance(backbones, Mapping):
        raise TypeError('attribution_backbones must be a {target: estimator} mapping')
    unknown=set(backbones)-{'toc_pct','total_gas_m3_t'}
    if unknown:
        raise ValueError('Unknown attribution target columns: '+', '.join(sorted(unknown)))
    tree_types=tuple(t for t in [ExtraTreesRegressor,RandomForestRegressor,
        GradientBoostingRegressor,HistGradientBoostingRegressor,
        XGBRegressor,LGBMRegressor,CatBoostRegressor] if t is not None)
    for target,estimator in backbones.items():
        validate_regressor(target,estimator)
        if not isinstance(estimator,tree_types):
            raise TypeError(f'attribution_backbones[{target!r}] requires a TreeExplainer-compatible '
                            'tree ensemble; non-tree regressors are unsupported. No fallback explainer is used.')
    return backbones

def _ready(df, group_col, well_col):
    raw = validate_dataset(df, group_col, well_col)
    raw['zone'] = raw['zone'].replace({v:k for k,v in sc.ZONE_EN.items()})
    return target_datasets(raw)

def foldwise_shap_importance(df, feature_sets=('full', 'logs-only'),
                             schemes=('LOWO', 'LOZO'), group_col='zone', well_col='well',
                             attribution_backbones=None):
    """Return long-form mean absolute SHAP and minimum ranks for each fold.

    Use the fixed 180-tree, leaf-size-three, seed-42 target backbones.
    Labels require three available conventional logs. Fit preprocessing and
    the model exclusively on each training partition; never evaluate SHAP
    on held-out rows. Fold identities in this diagnostic are anonymized.
    """
    backbones=validate_backbones(attribution_backbones) if attribution_backbones is not None else {}
    rows = []
    for dataset, target_col, data in _ready(df, group_col, well_col):
        target = 'TOC' if target_col == 'toc_pct' else 'total_gas'
        model = backbones[target_col] if target_col in backbones else ac.model_obj('ExtraTrees' if target == 'TOC' else 'RF')
        well_map, zone_map = sc.anonymizers(data)
        y = data[target_col].to_numpy(float)
        for feature_set in feature_sets:
            num, cat = sc.feature_config(data, feature_set)
            x = data[num + cat]
            for scheme in schemes:
                for fold, train, _ in sc.split_iter(data, scheme):
                    try:
                        imp = sc.importance_for_fold(model, x.iloc[train], y[train], num, cat).reindex(x.columns)
                    except Exception as exc:
                        if target_col not in backbones:
                            raise
                        raise RuntimeError(f'TreeExplainer attribution failed for {target_col}/{type(model).__name__}; '
                                           'supply a supported tree ensemble. No fallback explainer is used.') from exc
                    rank = imp.rank(ascending=False, method='min').astype(int)
                    fold_id = well_map[fold] if scheme == 'LOWO' else zone_map[fold]
                    for feature in x.columns:
                        rows.append(dict(target=target, feature_set=feature_set, cv_scheme=scheme,
                                         fold_id=fold_id, feature=feature, importance=float(imp[feature]),
                                         rank=int(rank[feature])))
    if not rows:
        raise ValueError('No eligible attribution folds: training folds require at least ten rows')
    return pd.DataFrame(rows)

def ranking_stability(perfold, top_k=5, attribution_backbones=None):
    """Return pairwise Spearman mean/sample SD and top-k fold frequencies.

    Minimum ranks retain ties; frequency can include more than k features.
    Frequencies use the source six-decimal convention. Fewer than two folds
    produce NaN pairwise statistics. Cross-feature-set comparisons use mean ranks.
    """
    backbones=validate_backbones(attribution_backbones) if attribution_backbones is not None else {}
    rows, averages = [], {}
    for (target, feature_set, scheme), group in perfold.groupby(
            ['target', 'feature_set', 'cv_scheme'], sort=False):
        features = list(group.feature.drop_duplicates())
        matrix = group.pivot(index='fold_id', columns='feature', values='importance').reindex(columns=features)
        ranks = group.pivot(index='fold_id', columns='feature', values='rank').reindex(columns=features)
        pairwise = [sc.spearman_safe(matrix.iloc[i].to_numpy(), matrix.iloc[j].to_numpy())
                    for i,j in combinations(range(len(matrix)), 2)]
        pairwise = [v for v in pairwise if np.isfinite(v)]
        mean_rank = ranks.mean().sort_values()
        counts = (ranks <= top_k).sum().sort_values(ascending=False)
        frequencies = {f:round(float(n)/len(ranks), 6) for f,n in counts.items() if n > 0}
        averages[(target, feature_set, scheme)] = mean_rank
        rows.append(dict(target=target, feature_set=feature_set, cv_scheme=scheme,
            pairwise_spearman_mean=float(np.mean(pairwise)) if pairwise else np.nan,
            pairwise_spearman_sd=float(np.std(pairwise, ddof=1)) if len(pairwise)>1 else np.nan,
            n_folds=len(matrix), **{f'top{top_k}_membership_json':json.dumps(frequencies)},
            top8_features=', '.join(mean_rank.head(8).index), method='shap', seed=42,
            model='ExtraTreesRegressor' if target=='TOC' else 'RandomForestRegressor', n_features=len(features)))
        target_col='toc_pct' if target=='TOC' else 'total_gas_m3_t'
        if target_col in backbones:
            rows[-1]['model']=type(backbones[target_col]).__name__
            rows[-1]['seed']=backbones[target_col].get_params().get('random_state',backbones[target_col].get_params().get('random_seed'))
    result = pd.DataFrame(rows)
    def compare(row):
        full = averages.get((row.target, 'full', row.cv_scheme))
        logs = averages.get((row.target, 'logs-only', row.cv_scheme))
        if full is None or logs is None: return np.nan
        common = [f for f in logs.index if f in full.index]
        return sc.spearman_safe(full[common].to_numpy(), logs[common].to_numpy())
    result['full_vs_logs_only_common_feature_spearman'] = result.apply(compare, axis=1)
    return result

def logs_only_ablation(df, group_col='zone', well_col='well', attribution_backbones=None):
    """Refit the fixed backbones with logs-only and full features on identical folds.

    Return source-compatible pooled R2, sample-weighted MAE and fold-mean
    Spearman, plus pooled Spearman. Use random five-fold, LOWO and LOZO;
    preprocessing is fit anew on training rows for every feature configuration.
    """
    backbones=validate_backbones(attribution_backbones) if attribution_backbones is not None else {}
    folds, predictions = [], []
    for dataset, target_col, data in _ready(df, group_col, well_col):
        target = 'TOC' if target_col == 'toc_pct' else 'total_gas'
        name = 'ExtraTrees' if target == 'TOC' else 'RF'
        model=backbones.get(target_col)
        if model is not None:
            name=type(model).__name__
        for label, feature_set in [('logs_only','logs-only'),('full_logs_plus_descriptors','full')]:
            num, cat = sc.feature_config(data, feature_set)
            f,p = ac.run_eval(data, dataset, target_col, target, name, label, num, cat, model=model)
            folds.append(f); predictions.append(p)
    pred = pd.concat(predictions, ignore_index=True)
    result = ac.summarize(pd.concat(folds, ignore_index=True), pred)
    keys = ['dataset','target','model','feature_set','cv_scheme']
    pooled = pred.groupby(keys).apply(lambda g: ac.spearman_safe(g.y_true, g.y_pred), include_groups=False)
    result = result.merge(pooled.rename('pooled_Spearman').reset_index(), on=keys, validate='one_to_one')
    return result
