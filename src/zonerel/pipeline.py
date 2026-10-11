"""Parameter-only orchestration of the retained paper workflow."""
from .io import validate_dataset,target_datasets
from .constants import FAST_MODELS,MODEL_ORDER
from .evaluation import eval_baselines
from .metrics import summarize,choose_models
from .domains import ordered_zones
from .conformal import conformal_rejection,reliability
from .reference import zone_skill
from .recalibration import recalibration_ladder,load_gas_params,empty_recalibration
from .rejection import random_control
from .decision import decision_matrix

def run_workflow(df,config=None):
    """Return result tables for validation, skill, intervals, rejection and C0–C4.

    config: group_col/well_col, fast, model_names, extra_models,
    attribution_backbones, run_recalibration, finite_sample, gas_params,
    support_assessment, coverage_threshold, and random_control. All file paths
    remain caller-managed. support_assessment is explicit evidence per
    (target, zone); omitted assessments remain false (§5.5).
    """
    config=dict(config or {})
    raw=validate_dataset(df,config.get('group_col','zone'),config.get('well_col','well'))
    datasets=target_datasets(raw)
    for _,_,data in datasets:
        ordered_zones(data, validate_lozo=True)
    backbones=config.get('attribution_backbones')
    if backbones is not None:
        from .attribution import validate_backbones
        validate_backbones(backbones)
    names=config.get('model_names') or (FAST_MODELS if config.get('fast') else MODEL_ORDER)
    extra=config.get('extra_models')
    # Custom-only pools still need the paired Phase III zone-mean reference.
    if extra and 'zone_mean' not in names:
        names=[*names,'zone_mean']
    folds,preds,status=eval_baselines(datasets,names,config.get('progress'),extra)
    if len(status):raise RuntimeError('Model evaluation errors: '+status.to_json(orient='records'))
    summary=summarize(folds,preds)
    conf,cpreds,reject=conformal_rejection(datasets,summary,config.get('finite_sample',False),extra)
    _,zone,well=reliability(cpreds)
    coverage=cpreds[(cpreds.method=='global')&cpreds.cv_scheme.isin(config.get('decision_protocols',['LOWO','LOZO']))].groupby(['target','zone'],as_index=False).agg(coverage=('covered','mean'),mean_width=('interval_width','mean'),n=('sample_id','size'),n_wells=('well','nunique'))
    # Preserve optional author-supplied bootstrap aliases, never model folds.
    aliases=[c for c in ['sample_anon','well_anon'] if c in raw]
    if aliases:preds=preds.merge(raw[['sample_id',*aliases]],on='sample_id',validate='many_to_one')
    selected={target:choose_models(summary,dataset)[0] for dataset,target,_ in datasets} if extra else None
    skill=zone_skill(preds,selected)
    if config.get('run_recalibration',True):
        ladder=recalibration_ladder(raw,config.get('gas_params') or load_gas_params(config.get('gas_params_path')))
    else:
        ladder=empty_recalibration()
    decisions=decision_matrix(coverage,skill,ladder['benefits'],config.get('support_assessment',{}),config.get('coverage_threshold',.80))
    results={'validation_summary':summary,'zone_skill':skill,'conformal_coverage':cpreds.groupby(['target','cv_scheme','method','zone'],as_index=False).agg(coverage=('covered','mean'),mean_width=('interval_width','mean'),n=('sample_id','size')),'rejection_curve':reject,'recalibration_ladder':ladder['metrics'],'recalibration_benefits':ladder['benefits'],'drift_typing':ladder['drift'],'decision_matrix':decisions,'validation_folds':folds,'oof_predictions':preds,'conformal_folds':conf,'conformal_predictions':cpreds,'recalibration_predictions':ladder['predictions']}
    if config.get('random_control',True):results['random_rejection_control']=random_control(cpreds)
    if config.get('run_attribution',False):
        from .attribution import foldwise_shap_importance,ranking_stability,logs_only_ablation
        results['attribution_perfold']=foldwise_shap_importance(raw,attribution_backbones=backbones)
        results['attribution_stability']=ranking_stability(results['attribution_perfold'],attribution_backbones=backbones)
        results['logs_only_ablation']=logs_only_ablation(raw,attribution_backbones=backbones)
    return results
