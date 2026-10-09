"""Synthetic scientific-contract tests; no confidential anchors or input files."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from zonerel.synthetic import generate_synthetic
from zonerel.io import validate_dataset, target_datasets
from zonerel.splits import split_iter
from zonerel.rejection import percentile_rank, composite_risk, unit_table
from zonerel.attribution import ranking_stability
from zonerel._shap_core import aggregate_shap
from zonerel.recalibration import evaluate_zone, add_proxies, load_gas_params
from zonerel.models import regressor_pool, make_pipe
from zonerel.decision import decision_action

@pytest.fixture
def data():
    return validate_dataset(generate_synthetic())

def test_simulated_memberships_and_reproducibility(data):
    pd.testing.assert_frame_equal(data,validate_dataset(generate_synthetic()))
    for zone,nt,ng,no in zip(['North','Central','South'],[8,9,15],[5,11,10],[3,5,7]):
        g=data[data.zone==zone]
        toc=set(g[g.toc_pct.notna()].well);gas=set(g[g.total_gas_m3_t.notna()].well)
        assert (len(toc),len(gas),len(toc&gas))==(nt,ng,no)
        assert toc-gas and gas-toc
        assert g.well.str.startswith('SYN_').all()

@pytest.mark.parametrize('scheme,group',[('LOWO','well'),('LOZO','zone')])
def test_heldout_groups_never_in_training(data,scheme,group):
    for _,_,frame in target_datasets(data):
        seen=[]
        for fold,tr,te in split_iter(frame,scheme):
            assert set(frame.iloc[tr][group]).isdisjoint(frame.iloc[te][group])
            assert len(tr)>=10
            seen.extend(te)
        assert sorted(seen)==list(range(len(frame)))

def test_average_ties_and_subprecision_noise():
    rank=percentile_rank(pd.Series([1.,1.+1e-11,2.,2.]))
    np.testing.assert_array_equal(rank,[.375,.375,.875,.875])
    f=pd.DataFrame(dict(interval_width=[2]*4,domain_knn=[4,3,2,1],missing_risk=[0]*4,quality_risk=[.1]*4))
    r=composite_risk(f)
    assert r.is_monotonic_decreasing
    altered=f.copy();altered['interval_width']+=1e-11
    np.testing.assert_array_equal(composite_risk(altered),r)

def test_rejection_uses_same_eligible_units():
    f=pd.DataFrame(dict(dataset=['s']*11,target=['t']*11,cv_scheme=['LOWO']*11,method=['global']*11,
                        model=['m']*11,fold=['small']*5+['eligible']*6,y_true=np.arange(11),y_pred=np.zeros(11),composite_risk=np.arange(11)))
    units=unit_table(f,'s','t','global').set_index('fold')
    assert not units.loc['small','eligible_for_20pct']
    assert units.loc['eligible','eligible_for_20pct']
    assert units.loc['eligible','risk_MAE_20pct_from_predictions']==pytest.approx(np.mean(np.arange(5,10)))

def test_signed_onehot_shap_is_combined_before_absolute_value():
    x=pd.DataFrame({'facies':['a','b'],'GR':[1.,2.]})
    pre=ColumnTransformer([('num','passthrough',['GR']),('cat',
        __import__('sklearn.pipeline',fromlist=['Pipeline']).Pipeline([('onehot',OneHotEncoder(sparse_output=False))]),['facies'])]).fit(x)
    result=aggregate_shap(pre,np.array([[1,2,-2],[1,3,-1]],float),x,['GR'],['facies'])
    np.testing.assert_array_equal(result.facies,[0.,2.])
    assert result.facies.abs().mean()==1.

def test_stability_uses_ties_and_pairwise_rank_correlation():
    rows=[]
    for fold,values in [('one',[4,3,2,1]),('two',[1,2,3,4])]:
        rank=pd.Series(values).rank(ascending=False,method='min')
        for i,(v,r) in enumerate(zip(values,rank)):
            rows.append(dict(target='TOC',feature_set='full',cv_scheme='LOWO',fold_id=fold,feature=f'f{i}',importance=v,rank=r))
    s=ranking_stability(pd.DataFrame(rows),top_k=1).iloc[0]
    assert s.pairwise_spearman_mean==pytest.approx(-1)
    assert s.n_folds==2
    assert '0.5' in s.top1_membership_json

def test_gas_proxy_does_not_depend_on_target_labels(data):
    def records(frame):
        return [{k:(None if pd.isna(v) else v) for k,v in r.items()} for r in frame.to_dict('records')]
    original=records(data)
    other=data.copy();other['toc_pct']=10000.;other['total_gas_m3_t']=-10000.
    changed=records(other)
    params=load_gas_params();add_proxies(original,params);add_proxies(changed,params)
    np.testing.assert_array_equal([r['gas_total_proxy_quant'] for r in original],[r['gas_total_proxy_quant'] for r in changed])
    r=original[0];p=max(.001,r['pressure_coeff']*r['depth_md']*.0098)
    ads=.39*1.3975*p/(p+7.60)
    phi=r['porosity_pct'] if r['porosity_pct'] is not None else data.loc[data.zone==r['zone'],'porosity_pct'].median()
    free=phi/100*.5/(2.65*(.00035096*.9*(293.15+.03*r['depth_md'])/p))
    assert r['gas_total_proxy_quant']==pytest.approx(ads+free)

def test_recalibration_prohibits_cross_zone_labels():
    with pytest.raises(ValueError,match='Exactly one zone'):
        evaluate_zone([{'zone':'North'},{'zone':'South'}],'toc_pct','proxy','C1_offset')

@pytest.mark.parametrize('evidence,expected',[
 ((True,True,'skill_directional',False),'use with interval'),
 ((True,True,'no_skill',False),'zone-level only'),
 ((False,True,'skill_CI_robust',True),'recalibrate locally'),
 ((False,True,'skill_CI_robust',False),'acquire more data')])
def test_four_actions(evidence,expected):
    assert decision_action(*evidence)==expected

def test_xgboost_native_prediction_uses_tolerance(data):
    x=data[['GR','AC','DEN','CNL','RT']].iloc[:40]
    y=data.toc_pct.iloc[:40].fillna(1.)
    model=regressor_pool(['XGBoost'])['XGBoost']
    assert model.get_params()['n_jobs']==2
    pipe=make_pipe('XGBoost',model,list(x.columns),[])
    pipe.fit(x,y)
    a=pipe.predict(x);b=pipe.predict(x.copy())
    np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-5)
    assert np.isfinite(a).all()

def test_custom_input_columns_and_duplicate_ids(data):
    custom=data.rename(columns={'zone':'fault_zone','well':'well_id'})
    pd.testing.assert_frame_equal(validate_dataset(custom,'fault_zone','well_id'),data)
    bad=data.copy();bad.loc[1,'sample_id']=bad.loc[0,'sample_id']
    with pytest.raises(ValueError,match='sample_id'):
        validate_dataset(bad)
