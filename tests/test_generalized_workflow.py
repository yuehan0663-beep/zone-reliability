"""Generalized-input contracts, using simulated rows exclusively."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from zonerel.pipeline import run_workflow
from zonerel.models import regressor_pool, make_pipe
from zonerel.metrics import choose_models
from zonerel.domains import ordered_zones
from zonerel.splits import split_iter
from zonerel import _shap_core, _ablation_core
from zonerel.attribution import foldwise_shap_importance, ranking_stability, logs_only_ablation
from zonerel.synthetic import generate_synthetic


@pytest.fixture(scope='module')
def four_zones():
    # Eight independent wells, six rows each, two wells per fault block.
    data=generate_synthetic()
    wells=list(data[data.toc_pct.notna() & data.total_gas_m3_t.notna()].well.unique())[:8]
    data=data[data.well.isin(wells)].copy().reset_index(drop=True)
    data['zone']=data.well.map({w:f'F{i//2+1}' for i,w in enumerate(wells)})
    return data


@pytest.fixture(scope='module')
def custom_tables(four_zones):
    return run_workflow(four_zones, {'model_names':['zone_mean','Ridge'],
        'extra_models':{'my_knn':KNeighborsRegressor(n_neighbors=3)},
        'run_attribution':True,'random_control':False})


def test_custom_regressor_reaches_intervals_rejection_and_decisions(custom_tables):
    folds=custom_tables['validation_folds']
    assert set(folds.loc[folds.model=='my_knn','cv_scheme'])=={'random','LOWO','LOZO'}
    summary=custom_tables['validation_summary']
    for dataset in summary.dataset.unique():
        assert 'my_knn' in choose_models(summary,dataset)
    assert 'my_knn' in set(custom_tables['conformal_folds'].model)
    assert 'my_knn' in set(custom_tables['rejection_curve'].model)
    assert len(custom_tables['zone_skill'])==8
    assert set(custom_tables['zone_skill'].predictor_group)=={'A_user_selected'}
    assert len(custom_tables['decision_matrix'])==8


def test_four_arbitrary_zones_shap_and_ablation(custom_tables):
    lozo=custom_tables['validation_folds'].query("cv_scheme=='LOZO'")
    assert (lozo.groupby(['target','model']).fold.nunique()==4).all()
    attr=custom_tables['attribution_perfold'].query("cv_scheme=='LOZO'")
    assert (attr.groupby(['target','feature_set']).fold_id.nunique()==4).all()
    assert set(attr.fold_id)=={f'zone_en_F{i}' for i in range(1,5)}
    ablation=custom_tables['logs_only_ablation'].query("cv_scheme=='LOZO'")
    assert (ablation.folds==4).all()
    assert set(ablation.feature_set)=={'logs_only','full_logs_plus_descriptors'}


def test_custom_only_pool_adds_paired_reference_and_uses_custom_skill(four_zones):
    estimator=KNeighborsRegressor(n_neighbors=3)
    tables=run_workflow(four_zones,{'model_names':['my_knn'],
        'extra_models':{'my_knn':estimator},'run_recalibration':False,'random_control':False})
    assert set(tables['validation_folds'].model)=={'zone_mean','my_knn'}
    assert set(tables['zone_skill'].model)=={'my_knn'}
    assert set(tables['conformal_folds'].model)=={'my_knn'}
    assert len(tables['decision_matrix'])==8
    assert not hasattr(estimator,'_fit_X')


@pytest.mark.parametrize('splitter',[split_iter,_shap_core.split_iter,_ablation_core.split_iter])
def test_two_zone_warning_and_one_zone_error(four_zones,splitter):
    data=four_zones.copy()
    data['zone']=data.zone.replace({'F3':'F1','F4':'F2'})
    with pytest.warns(UserWarning,match='only 2 folds'):
        folds=list(splitter(data,'LOZO'))
    assert len(folds)==2
    data['zone']='only'
    with pytest.raises(ValueError,match='at least 2'):
        list(splitter(data,'LOZO'))


def test_one_zone_workflow_fails_before_fitting(four_zones):
    data=four_zones.copy();data['zone']='only'
    with pytest.raises(ValueError,match='at least 2'):
        run_workflow(data,{'fast':True})


@pytest.mark.parametrize('labels,expected',[
    (['South','North','Central'],['North','Central','South']),
    (['南','北','中'],['北','中','南']),
    (['F4','F1','F3','F2'],['F1','F2','F3','F4'])])
def test_canonical_and_arbitrary_order(labels,expected):
    assert ordered_zones(pd.DataFrame({'zone':labels}))==expected


def test_missing_pressure_and_porosity(four_zones):
    data=four_zones.drop(columns=['pressure_coeff','porosity_pct'])
    with pytest.warns(UserWarning,match='Gas physical proxy skipped') as caught:
        tables=run_workflow(data,{'model_names':['zone_mean','Ridge'],
            'extra_models':{'my_knn':KNeighborsRegressor(n_neighbors=3)},'random_control':False})
    assert len(caught)==1
    assert not tables['validation_folds'].empty
    assert not tables['decision_matrix'].empty
    assert not tables['decision_matrix'].local_correction_robust.any()
    assert 'recalibrate locally' not in set(tables['decision_matrix'].action)
    assert 'total_gas' not in set(tables['recalibration_ladder'].target)
    assert 'pressure_coeff' not in _shap_core.feature_config(data,'full')[0]


def test_skip_physical_recalibration_without_loading_params(four_zones,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Physical recalibration must not run')
    monkeypatch.setattr('zonerel.pipeline.recalibration_ladder',forbidden)
    monkeypatch.setattr('zonerel.pipeline.load_gas_params',forbidden)
    tables=run_workflow(four_zones,{'model_names':['zone_mean','Ridge'],
        'run_recalibration':False,'random_control':False})
    assert not tables['conformal_predictions'].empty
    assert not tables['rejection_curve'].empty
    for key in ['recalibration_ladder','recalibration_benefits','drift_typing','recalibration_predictions']:
        assert tables[key].empty and len(tables[key].columns)>0
    assert set(tables['decision_matrix'].action)<={'use with interval','zone-level only','acquire more data'}
    assert not tables['decision_matrix'].local_correction_robust.any()


@pytest.mark.parametrize('name',['RF','KNN','XGBoost','CatBoost'])
def test_reserved_name_rejected_even_if_library_unavailable(name,monkeypatch):
    monkeypatch.setattr('zonerel.models.XGBRegressor',None)
    monkeypatch.setattr('zonerel.models.CatBoostRegressor',None)
    with pytest.raises(ValueError,match='collides'):
        regressor_pool(['Ridge'],{name:KNeighborsRegressor()})


@pytest.mark.parametrize('bad',[object(),type('PredictOnly',(),{'predict':lambda self,x:x})()])
def test_missing_fit_predict_rejected(bad):
    with pytest.raises(TypeError,match='fit and predict'):
        regressor_pool(extra_models={'bad':bad})


def test_uncloneable_and_fitted_regressors_rejected():
    class Uncloneable:
        def fit(self,x,y): return self
        def predict(self,x): return x
    with pytest.raises(TypeError,match='cloneable'):
        regressor_pool(extra_models={'bad':Uncloneable()})
    fitted=KNeighborsRegressor().fit([[0],[1],[2]],[0,1,2])
    with pytest.raises(ValueError,match='unfitted'):
        regressor_pool(extra_models={'fitted':fitted})


def test_non_tree_attribution_rejected_early(four_zones):
    with pytest.raises(TypeError,match='TreeExplainer-compatible tree ensemble'):
        run_workflow(four_zones,{'run_attribution':True,
            'attribution_backbones':{'toc_pct':KNeighborsRegressor()}})


def test_custom_tree_backbone_is_used_and_not_fitted_in_place(four_zones):
    tree=ExtraTreesRegressor(n_estimators=7,min_samples_leaf=2,random_state=123)
    backbones={'toc_pct':tree}
    data=four_zones.drop(columns='total_gas_m3_t')
    attr=foldwise_shap_importance(data,attribution_backbones=backbones)
    stability=ranking_stability(attr,attribution_backbones=backbones)
    ablation=logs_only_ablation(data,attribution_backbones=backbones)
    assert set(stability.seed)=={123}
    assert set(stability.model)=={'ExtraTreesRegressor'}
    assert set(ablation.model)=={'ExtraTreesRegressor'}
    assert not hasattr(tree,'estimators_')


def test_cloning_and_preprocessing_use_training_rows_only():
    class Recorder(RegressorMixin,BaseEstimator):
        seen=[]
        def fit(self,x,y):
            type(self).seen.append(np.array(x,copy=True));self.mean_=float(np.mean(y));return self
        def predict(self,x): return np.full(len(x),self.mean_)
    estimator=Recorder()
    train=pd.DataFrame({'GR':[1.,np.nan,3.],'layer':['a','a','b']})
    test=pd.DataFrame({'GR':[1000.],'layer':['unseen']})
    pipe=make_pipe('custom',estimator,['GR'],['layer']).fit(train,[1.,2.,3.])
    pre=pipe.named_steps['pre']
    assert pre.named_transformers_['num'].named_steps['impute'].statistics_[0]==2.
    assert pre.named_transformers_['num'].named_steps['scale'].mean_[0]==2.
    assert 'unseen' not in pre.named_transformers_['cat'].named_steps['onehot'].categories_[0]
    assert pipe.predict(test).shape==(1,)
    assert not hasattr(estimator,'mean_')
    assert pipe.named_steps['model'] is not estimator
