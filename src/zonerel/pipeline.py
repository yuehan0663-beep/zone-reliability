"""Parameter-only orchestration of the retained paper workflow."""
from .io import validate_dataset,target_datasets
from .constants import FAST_MODELS,MODEL_ORDER
from .evaluation import eval_baselines
from .metrics import summarize
from .conformal import conformal_rejection,reliability
from .reference import zone_skill
from .recalibration import recalibration_ladder,load_gas_params
from .rejection import random_control
from .decision import decision_matrix

def run_workflow(df,config=None):
    """Return result tables for validation, skill, intervals, rejection and C0–C4.

    config: group_col/well_col, fast, model_names, finite_sample, gas_params,
    support_assessment, coverage_threshold, and random_control. All file paths
    remain caller-managed. support_assessment is explicit evidence per
    (target, zone); omitted assessments remain false (§5.5).
    """
    config=dict(config or {})
    raw=validate_dataset(df,config.get('group_col','zone'),config.get('well_col','well'))
    datasets=target_datasets(raw)
    names=config.get('model_names') or (FAST_MODELS if config.get('fast') else MODEL_ORDER)
    folds,preds,status=eval_baselines(datasets,names,config.get('progress'))
    if len(status):raise RuntimeError('Model evaluation errors: '+status.to_json(orient='records'))
    summary=summarize(folds,preds)
    conf,cpreds,reject=conformal_rejection(datasets,summary,config.get('finite_sample',False))
    _,zone,well=reliability(cpreds)
    coverage=cpreds[(cpreds.method=='global')&cpreds.cv_scheme.isin(config.get('decision_protocols',['LOWO','LOZO']))].groupby(['target','zone'],as_index=False).agg(coverage=('covered','mean'),mean_width=('interval_width','mean'),n=('sample_id','size'),n_wells=('well','nunique'))
    # Preserve optional author-supplied bootstrap aliases, never model folds.
    aliases=[c for c in ['sample_anon','well_anon'] if c in raw]
    if aliases:preds=preds.merge(raw[['sample_id',*aliases]],on='sample_id',validate='many_to_one')
    skill=zone_skill(preds)
    ladder=recalibration_ladder(raw,config.get('gas_params') or load_gas_params(config.get('gas_params_path')))
    decisions=decision_matrix(coverage,skill,ladder['benefits'],config.get('support_assessment',{}),config.get('coverage_threshold',.80))
    results={'validation_summary':summary,'zone_skill':skill,'conformal_coverage':cpreds.groupby(['target','cv_scheme','method','zone'],as_index=False).agg(coverage=('covered','mean'),mean_width=('interval_width','mean'),n=('sample_id','size')),'rejection_curve':reject,'recalibration_ladder':ladder['metrics'],'recalibration_benefits':ladder['benefits'],'drift_typing':ladder['drift'],'decision_matrix':decisions,'validation_folds':folds,'oof_predictions':preds,'conformal_folds':conf,'conformal_predictions':cpreds,'recalibration_predictions':ladder['predictions']}
    if config.get('random_control',True):results['random_rejection_control']=random_control(cpreds)
    if config.get('run_attribution',False):
        from .attribution import foldwise_shap_importance,ranking_stability,logs_only_ablation
        results['attribution_perfold']=foldwise_shap_importance(raw)
        results['attribution_stability']=ranking_stability(results['attribution_perfold'])
        results['logs_only_ablation']=logs_only_ablation(raw)
    return results
