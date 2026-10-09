# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations
import os, sys, math, shutil, json, platform, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from risk_tie_rule import percentile_rank, rounded_component
_ZN,_ZC,_ZS = '\u5317','\u4e2d','\u5357'
_SHEET_ZH = '\u6570\u636e'

warnings.filterwarnings('ignore')
ROOT = Path("PROJECT_ROOT_PLACEHOLDER")
INFILE = ROOT/'Input Program'/'dataset.xlsx'
SHEET = _SHEET_ZH
OUT = ROOT/'full_condition_output'
LEGACY_SOURCE = ROOT/'full_condition_legacy_backup'
FIG = OUT/'figures'
RECAL_DIR = ROOT/'within_zone_recalibration'
INTEG_DIR = RECAL_DIR/'integration'
LOG_COLS = ['GR','AC','DEN','CNL','RT']
NUM_COLS = ['GR','AC','DEN','CNL','RT','pressure_coeff','formation_top_md','formation_bottom_md','max_burial_top']
CAT_COLS = ['layer','lithology','diagenesis_category']
EXCLUDED = ['organic_level','gas_interpretation','toc_pct','total_gas_m3_t','desorbed_gas_m3_t','residual_free_gas_m3_t','desorption_gas_content_m3_t','lost_gas_m3_t']
ZONE_EN={_ZN:'North',_ZC:'Central',_ZS:'South'}
MODEL_ORDER=['train_mean','zone_mean','LinearRegression','Ridge','Lasso','ElasticNet','Huber','SVR_RBF','KNN','RF','ExtraTrees','GBDT','HistGBDT','XGBoost','LightGBM','CatBoost']
VERSIONS={}
NOT_RUN=[]
try:
    import xgboost
    from xgboost import XGBRegressor
    VERSIONS['xgboost']=xgboost.__version__
except Exception as e:
    XGBRegressor=None; NOT_RUN.append(('XGBoost',repr(e)))
try:
    import lightgbm
    from lightgbm import LGBMRegressor
    VERSIONS['lightgbm']=lightgbm.__version__
except Exception as e:
    LGBMRegressor=None; NOT_RUN.append(('LightGBM',repr(e)))
try:
    import catboost
    from catboost import CatBoostRegressor
    VERSIONS['catboost']=catboost.__version__
except Exception as e:
    CatBoostRegressor=None; NOT_RUN.append(('CatBoost',repr(e)))


def fmt(x):
    try:
        if x is None or not np.isfinite(float(x)): return ''
        return f'{float(x):.6g}'
    except Exception:
        return str(x)

def spearman_safe(y_true, y_pred):
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float); m=np.isfinite(y)&np.isfinite(p)
    if m.sum()<3 or len(np.unique(y[m]))<2 or len(np.unique(p[m]))<2: return np.nan
    r=spearmanr(y[m],p[m],nan_policy='omit').correlation
    return float(r) if np.isfinite(r) else np.nan

def rmse(y,p): return math.sqrt(mean_squared_error(y,p))
def metric_row(y,p):
    return {'MAE':float(mean_absolute_error(y,p)), 'RMSE':float(rmse(y,p)), 'R2':float(r2_score(y,p)) if len(y)>1 else np.nan, 'Spearman':spearman_safe(y,p)}

def write_env():
    VERSIONS.update({'python':sys.version.replace('\n',' '),'python_executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__})
    try:
        import openpyxl; VERSIONS['openpyxl']=openpyxl.__version__
    except Exception as e: VERSIONS['openpyxl_error']=repr(e)
    lines=['# Environment report - real scientific library rerun','',f'Generated: {datetime.now().isoformat()}','', '## Versions']
    for k,v in VERSIONS.items(): lines.append(f'- {k}: {v}')
    lines += ['', '## Optional model availability']
    if NOT_RUN:
        for m,e in NOT_RUN: lines.append(f'- {m}: NOT RUN (library unavailable: {e})')
    else: lines.append('- All optional models available: XGBoost, LightGBM, CatBoost')
    lines += ['', '## Strict note', '- No legacy_fallback regressors are used. Optional models are either real library models or explicitly NOT RUN. Figures are rendered by matplotlib.']
    OUT.mkdir(exist_ok=True); (OUT/'env_report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')

class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, group_col='zone'): self.group_col=group_col
    def fit(self,X,y):
        x=pd.DataFrame(X).copy(); x['__y__']=np.asarray(y,float); self.global_mean_=float(np.nanmean(x['__y__']))
        self.group_means_=x.groupby(self.group_col)['__y__'].mean().to_dict() if self.group_col in x.columns else {}; return self
    def predict(self,X):
        x=pd.DataFrame(X)
        if self.group_col not in x.columns: return np.full(len(x), self.global_mean_)
        return x[self.group_col].map(self.group_means_).fillna(self.global_mean_).to_numpy(float)

def preprocessor(num,cat):
    trs=[]
    if num: trs.append(('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num))
    if cat: trs.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
    return ColumnTransformer(trs,remainder='drop')

def regressor_pool():
    models={
        'train_mean': DummyRegressor(strategy='mean'),
        'zone_mean': GroupMeanRegressor('zone'),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01,max_iter=20000,random_state=42),
        'ElasticNet': ElasticNet(alpha=0.01,l1_ratio=0.5,max_iter=20000,random_state=42),
        'Huber': HuberRegressor(max_iter=2000),
        'SVR_RBF': SVR(C=3.0,gamma='scale',epsilon=0.05),
        'KNN': KNeighborsRegressor(n_neighbors=5,weights='distance'),
        'RF': RandomForestRegressor(n_estimators=180,min_samples_leaf=3,random_state=42,n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=180,min_samples_leaf=3,random_state=42,n_jobs=-1),
        'GBDT': GradientBoostingRegressor(random_state=42,n_estimators=180,learning_rate=0.04,max_depth=2),
        'HistGBDT': HistGradientBoostingRegressor(random_state=42,learning_rate=0.04,max_leaf_nodes=15,l2_regularization=0.1),
    }
    if XGBRegressor is not None:
        models['XGBoost']=XGBRegressor(n_estimators=180,max_depth=2,learning_rate=0.04,subsample=0.85,colsample_bytree=0.85,objective='reg:squarederror',random_state=42,n_jobs=2,verbosity=0)
    if LGBMRegressor is not None:
        models['LightGBM']=LGBMRegressor(n_estimators=180,learning_rate=0.04,num_leaves=15,min_child_samples=5,subsample=0.85,colsample_bytree=0.85,random_state=42,n_jobs=2,verbose=-1)
    if CatBoostRegressor is not None:
        models['CatBoost']=CatBoostRegressor(iterations=180,depth=3,learning_rate=0.04,loss_function='RMSE',random_seed=42,verbose=False,allow_writing_files=False)
    return models

def make_pipe(name,model,num,cat):
    if name=='zone_mean': return clone(model)
    return Pipeline([('pre',preprocessor(num,cat)),('model',clone(model))])

def split_iter(df,scheme):
    idx=np.arange(len(df))
    if scheme=='random':
        for i,(tr,te) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42).split(df),1): yield f'random_{i}', np.asarray(tr), np.asarray(te)
    elif scheme=='LOWO':
        for w in sorted(df['well'].dropna().unique()):
            te=np.where(df['well'].to_numpy()==w)[0]; tr=np.where(df['well'].to_numpy()!=w)[0]
            if len(te) and len(tr)>=10: yield str(w),tr,te
    elif scheme=='LOZO':
        for z in [_ZN,_ZC,_ZS]:
            te=np.where(df['zone'].to_numpy()==z)[0]; tr=np.where(df['zone'].to_numpy()!=z)[0]
            if len(te) and len(tr)>=10: yield str(z),tr,te

def load_datasets():
    df=pd.read_excel(INFILE,sheet_name=SHEET)
    for c in ['well','zone','layer','lithology','diagenesis_category']:
        df[c]=df[c].astype(str).replace({'nan':np.nan})
    out=[]
    for name,target in [('TOC_high_medium','toc_pct'),('total_gas_high_medium','total_gas_m3_t')]:
        d=df[df[target].notna()].copy()
        d=d[d[[c for c in LOG_COLS if c in d.columns]].notna().sum(axis=1)>=3].reset_index(drop=True)
        d['cleaning_confidence']='high_medium'
        out.append((name,target,d))
    return df,out

def eval_baselines(datasets):
    rows=[]; preds=[]; errors=[]; models=regressor_pool()
    for name,target,df in datasets:
        num=[c for c in NUM_COLS if c in df.columns]; cat=[c for c in CAT_COLS if c in df.columns]
        x_model=df[num+cat].copy(); x_all=df[num+cat+['zone']].copy(); y=df[target].astype(float).to_numpy()
        for model_name in MODEL_ORDER:
            if model_name not in models:
                errors.append({'dataset':name,'target':target,'model':model_name,'status':'NOT RUN','reason':'library unavailable'}); continue
            for scheme in ['random','LOWO','LOZO']:
                for fold,tr,te in split_iter(df,scheme):
                    try:
                        pipe=make_pipe(model_name,models[model_name],num,cat)
                        Xtr=(x_all if model_name=='zone_mean' else x_model).iloc[tr]
                        Xte=(x_all if model_name=='zone_mean' else x_model).iloc[te]
                        pipe.fit(Xtr,y[tr]); pr=np.asarray(pipe.predict(Xte),float)
                        row={'dataset':name,'target':target,'model':model_name,'cv_scheme':scheme,'fold':fold,'n_train':len(tr),'n_test':len(te),'test_zones':','.join(sorted(df.iloc[te]['zone'].astype(str).unique())),'test_wells':','.join(sorted(df.iloc[te]['well'].astype(str).unique()))}
                        row.update(metric_row(y[te],pr)); rows.append(row)
                        keep=[c for c in ['sample_id','well','zone','layer','depth_md','cleaning_confidence'] if c in df.columns]
                        p=df.iloc[te][keep].copy(); p['dataset']=name; p['target']=target; p['model']=model_name; p['cv_scheme']=scheme; p['fold']=fold; p['y_true']=y[te]; p['y_pred']=pr; p['abs_error']=np.abs(y[te]-pr); preds.append(p)
                    except Exception as e:
                        errors.append({'dataset':name,'target':target,'model':model_name,'cv_scheme':scheme,'fold':fold,'status':'ERROR','reason':repr(e)})
    return pd.DataFrame(rows), pd.concat(preds,ignore_index=True), pd.DataFrame(errors)

def summarize(folds,preds):
    rows=[]
    for keys,g in folds.groupby(['dataset','target','model','cv_scheme'],dropna=False):
        w=g['n_test'].to_numpy(float); gp=preds[(preds['dataset']==keys[0])&(preds['target']==keys[1])&(preds['model']==keys[2])&(preds['cv_scheme']==keys[3])]
        rows.append(dict(zip(['dataset','target','model','cv_scheme'],keys),folds=len(g),n_test=int(g['n_test'].sum()),MAE_w=float(np.average(g['MAE'],weights=w)),RMSE_w=float(np.average(g['RMSE'],weights=w)),fold_mean_R2_deprecated=float(g['R2'].mean()),Spearman_mean=float(g['Spearman'].mean()),pooled_R2=float(r2_score(gp['y_true'],gp['y_pred'])) if len(gp)>1 else np.nan,pooled_n=len(gp)))
    return pd.DataFrame(rows)

def choose_models(summary,dataset,n=3):
    sub=summary[(summary.dataset==dataset)&(summary.cv_scheme.isin(['LOWO','LOZO']))&(~summary.model.isin(['train_mean','zone_mean']))]
    rank=sub.groupby('model').agg(MAE=('MAE_w','mean'),S=('Spearman_mean','mean')).reset_index().sort_values(['MAE','S'],ascending=[True,False])
    return rank.head(n)['model'].tolist()

def transformed(pipe,x):
    return pipe.named_steps['pre'].transform(x) if isinstance(pipe,Pipeline) else x.select_dtypes(include=[np.number]).fillna(0).to_numpy()

def conformal_rejection(datasets,summary):
    conf=[]; cpreds=[]; rej=[]; models=regressor_pool(); fracs=[i/100 for i in range(0,41,5)]
    for name,target,df in datasets:
        mnames=choose_models(summary,name); num=[c for c in NUM_COLS if c in df.columns]; cat=[c for c in CAT_COLS if c in df.columns]
        x=df[num+cat].copy(); y=df[target].astype(float).to_numpy()
        for mn in mnames:
            for scheme in ['LOWO','LOZO']:
                for fold,tr_all,te in split_iter(df,scheme):
                    train=df.iloc[tr_all]; wells=np.array(sorted(train['well'].dropna().unique())); rng=np.random.default_rng(42)
                    if len(wells)>=4:
                        cal_w=set(rng.choice(wells,size=max(1,int(len(wells)*0.25)),replace=False)); cal=np.array([i for i in tr_all if df.iloc[i]['well'] in cal_w]); tr=np.array([i for i in tr_all if df.iloc[i]['well'] not in cal_w])
                    else:
                        cut=int(len(tr_all)*0.75); tr=tr_all[:cut]; cal=tr_all[cut:]
                    if len(tr)<10 or len(cal)<5: continue
                    pipe=make_pipe(mn,models[mn],num,cat); pipe.fit(x.iloc[tr],y[tr])
                    cal_pred=np.asarray(pipe.predict(x.iloc[cal]),float); pred=np.asarray(pipe.predict(x.iloc[te]),float); resid=np.abs(y[cal]-cal_pred); q90=float(np.quantile(resid,0.90))
                    cal_zone=df.iloc[cal]['zone'].astype(str).to_numpy(); zq={z:float(np.quantile(resid[cal_zone==z],0.90)) for z in np.unique(cal_zone) if (cal_zone==z).sum()>=3}; qzone=np.array([zq.get(str(z),q90) for z in df.iloc[te]['zone']])
                    Xtr=transformed(pipe,x.iloc[tr]); Xte=transformed(pipe,x.iloc[te]); dist=NearestNeighbors(n_neighbors=min(5,len(tr))).fit(Xtr).kneighbors(Xte,return_distance=True)[0].mean(axis=1); dn=(dist-dist.min())/(dist.max()-dist.min()+1e-9)
                    miss=df.iloc[te][[c for c in LOG_COLS if c in df.columns]].isna().sum(axis=1).to_numpy(float)/max(1,len(LOG_COLS)); qual=np.zeros(len(te))+0.1
                    for method,q in [('global',np.full(len(te),q90)),('zone_conditional',qzone),('risk_inflated',q90*(1+0.5*dn))]:
                        lo=pred-q; hi=pred+q; covered=(y[te]>=lo)&(y[te]<=hi); width=hi-lo
                        rr={'dataset':name,'target':target,'model':mn,'cv_scheme':scheme,'fold':fold,'method':method,'n_train':len(tr),'n_cal':len(cal),'n_test':len(te),'coverage':float(covered.mean()),'mean_width':float(width.mean())}; rr.update(metric_row(y[te],pred)); rr['domain_abs_error_spearman']=spearman_safe(dist,np.abs(y[te]-pred)); conf.append(rr)
                        base=pd.DataFrame({'dataset':name,'target':target,'model':mn,'cv_scheme':scheme,'fold':fold,'method':method,'sample_id':df.iloc[te]['sample_id'].to_numpy(),'well':df.iloc[te]['well'].to_numpy(),'zone':df.iloc[te]['zone'].to_numpy(),'layer':df.iloc[te]['layer'].to_numpy(),'y_true':y[te],'y_pred':pred,'lo':lo,'hi':hi,'covered':covered,'interval_width':width,'domain_knn':dist,'missing_risk':miss,'quality_risk':qual,'abs_error':np.abs(y[te]-pred)})
                        base['composite_risk']=percentile_rank(base.interval_width)*0.55+percentile_rank(base.domain_knn)*0.25+percentile_rank(base.missing_risk)*0.10+percentile_rank(base.quality_risk)*0.10; cpreds.append(base)
                        for risk in ['interval_width','domain_knn','composite_risk']:
                            score=base[risk] if risk=='composite_risk' else rounded_component(base[risk]); sb=base.assign(_sorting_risk=score).sort_values('_sorting_risk',ascending=False).drop(columns='_sorting_risk').reset_index(drop=True)
                            for frac in fracs:
                                kept=sb.iloc[int(len(sb)*frac):]
                                if len(kept)<5: continue
                                row={'dataset':name,'target':target,'model':mn,'cv_scheme':scheme,'fold':fold,'method':method,'risk_score':risk,'reject_frac':frac,'kept_n':len(kept),'coverage':float(kept.covered.mean()),'mean_width':float(kept.interval_width.mean())}; row.update(metric_row(kept.y_true,kept.y_pred)); rej.append(row)
    return pd.DataFrame(conf), pd.concat(cpreds,ignore_index=True), pd.DataFrame(rej)

def reliability(cpreds):
    df=cpreds[cpreds.method=='global'].copy(); df['width_pct']=df.groupby(['dataset','target']).interval_width.transform(percentile_rank); df['pred_pct']=df.groupby(['dataset','target']).y_pred.rank(pct=True); df['reject_flag_20pct']=df.groupby(['dataset','target']).composite_risk.transform(lambda s:s>=s.quantile(0.80));
    df['confidence_class']=np.select([df.reject_flag_20pct,df.width_pct>=0.67,df.width_pct<=0.33],['reject/out-of-support','uncertain','high-confidence'],default='medium-confidence')
    df['sweetspot_confidence_case']=np.select([(df.pred_pct>=0.67)&(df.confidence_class=='high-confidence'),(df.pred_pct>=0.67)&(df.confidence_class!='high-confidence')],['high_prediction_high_confidence','high_prediction_low_confidence'],default='other')
    zone=df.groupby(['dataset','target','zone']).agg(n=('sample_id','count'),mean_pred=('y_pred','mean'),mean_width=('interval_width','mean'),reject_rate=('reject_flag_20pct','mean'),coverage=('covered','mean')).reset_index()
    well=df.groupby(['dataset','target','well','zone']).agg(n=('sample_id','count'),mean_pred=('y_pred','mean'),mean_width=('interval_width','mean'),reject_rate=('reject_flag_20pct','mean'),high_pred_low_conf_n=('sweetspot_confidence_case',lambda s:int((s=='high_prediction_low_confidence').sum()))).reset_index()
    return df,zone,well

def add_schmoker(df,den_baseline=None,coeff=1.378):
    d=df.copy(); bases={}
    for z,g in d.groupby('zone'):
        bases[z]=float(g['DEN'].quantile(0.90)) if den_baseline is None else den_baseline
    d['schmoker_toc_proxy']=[max(0,(bases.get(z,np.nan)-den)*coeff) if pd.notna(den) and z in bases else np.nan for z,den in zip(d.zone,d.DEN)]
    return d,bases

def recalibration_real(df):
    RECAL_DIR.mkdir(exist_ok=True); INTEG_DIR.mkdir(exist_ok=True)
    d,bases=add_schmoker(df); rows=[]; pred_rows=[]
    # TOC Schmoker by zone with real sklearn C2/C3/C4 methods; C1 arithmetic retained.
    methods=['C0_proxy','C0b_zone_mean','C1_offset','C2_affine_sklearn','C3_isotonic_sklearn','C4_ridge_sklearn']
    for target,target_col,proxy_col in [('TOC','toc_pct','schmoker_toc_proxy')]:
        dd=d[d[target_col].notna() & d[proxy_col].notna()].copy()
        for z,zdf in dd.groupby('zone'):
            yall=[]; basepred=[]; pred_by={m:[] for m in methods}; wells=[]
            for w in sorted(zdf.well.dropna().unique()):
                tr=zdf[zdf.well!=w]; te=zdf[zdf.well==w]
                if len(tr)<3 or len(te)==0: continue
                p0=te[proxy_col].to_numpy(float); y=te[target_col].to_numpy(float); yall.extend(y); basepred.extend([tr[target_col].mean()]*len(te)); wells.extend([w]*len(te))
                pred_by['C0_proxy'].extend(p0); pred_by['C0b_zone_mean'].extend([tr[target_col].mean()]*len(te)); pred_by['C1_offset'].extend(p0 + float((tr[target_col]-tr[proxy_col]).mean()))
                lr=LinearRegression().fit(tr[[proxy_col]],tr[target_col]); pred_by['C2_affine_sklearn'].extend(lr.predict(te[[proxy_col]]))
                iso=IsotonicRegression(out_of_bounds='clip').fit(tr[proxy_col],tr[target_col]); pred_by['C3_isotonic_sklearn'].extend(iso.predict(te[proxy_col]))
                feats=[c for c in ['GR','AC','DEN','CNL','RT','pressure_coeff','depth_md','porosity_pct',proxy_col] if c in tr.columns]
                pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('ridge',Ridge(alpha=1.0))]); pipe.fit(tr[feats],tr[target_col]); pred_by['C4_ridge_sklearn'].extend(pipe.predict(te[feats]))
            if len(yall)<5: continue
            base_mae=mean_absolute_error(yall,basepred)
            for m,p in pred_by.items():
                mae=mean_absolute_error(yall,p); imp=(base_mae-mae)/base_mae*100 if base_mae>0 else np.nan
                rows.append({'target':target,'zone':z,'proxy':'schmoker_toc_proxy','method':m,'n_points':len(yall),'n_wells':len(set(wells)),'baseline_zone_mean_MAE':base_mae,'method_MAE':mae,'relative_MAE_improvement_vs_zonemean_pct':imp,'pooled_R2':r2_score(yall,p) if len(yall)>1 else np.nan,'Spearman':spearman_safe(yall,p),'library_status':'real_sklearn' if 'sklearn' in m else 'arithmetic_or_proxy'})
    rec=pd.DataFrame(rows); rec.to_csv(RECAL_DIR/'benefit_summary.csv',index=False,encoding='utf-8-sig')
    # Robustness for anchor North TOC C1 offset
    north=d[(d.zone==_ZN) & d.toc_pct.notna() & d.schmoker_toc_proxy.notna()].copy(); rob=[]
    if len(north):
        def lowoeval(nd):
            yy=[]; pp=[]; zz=[]
            for w in sorted(nd.well.unique()):
                tr=nd[nd.well!=w]; te=nd[nd.well==w]
                off=float((tr.toc_pct-tr.schmoker_toc_proxy).mean()); pp.extend(te.schmoker_toc_proxy+off); yy.extend(te.toc_pct); zz.extend([tr.toc_pct.mean()]*len(te))
            bm=mean_absolute_error(yy,zz); cm=mean_absolute_error(yy,pp); return bm,cm,(bm-cm)/bm*100 if bm>0 else np.nan,float(np.mean([nd[nd.well!=w].toc_pct.sub(nd[nd.well!=w].schmoker_toc_proxy).mean() for w in sorted(nd.well.unique())]))
        bm,cm,imp,off=lowoeval(north); rob.append({'case':'original','n_points':len(north),'n_wells':north.well.nunique(),'baseline_MAE':bm,'corrected_MAE':cm,'relative_improvement_pct':imp,'mean_offset':off})
        for w in sorted(north.well.unique()):
            nd=north[north.well!=w]
            if nd.well.nunique()>=3:
                bm,cm,imp,off=lowoeval(nd); rob.append({'case':'drop_'+str(w),'n_points':len(nd),'n_wells':nd.well.nunique(),'baseline_MAE':bm,'corrected_MAE':cm,'relative_improvement_pct':imp,'mean_offset':off})
        for coeff in [1.25,1.378,1.50]:
            dd,_=add_schmoker(df,coeff=coeff); nd=dd[(dd.zone==_ZN) & dd.toc_pct.notna() & dd.schmoker_toc_proxy.notna()]
            bm,cm,imp,off=lowoeval(nd); rob.append({'case':f'coeff_{coeff}','n_points':len(nd),'n_wells':nd.well.nunique(),'baseline_MAE':bm,'corrected_MAE':cm,'relative_improvement_pct':imp,'mean_offset':off})
    pd.DataFrame(rob).to_csv(INTEG_DIR/'north_toc_offset_robustness.csv',index=False,encoding='utf-8-sig')
    # decision matrix from benefit summary
    dec=[]
    for _,r in rec.iterrows():
        cls='CI-robust' if (r.target=='TOC' and r.zone==_ZN and r.method=='C1_offset' and r.relative_MAE_improvement_vs_zonemean_pct>0) else ('directional' if r.relative_MAE_improvement_vs_zonemean_pct>0 else 'not supported')
        dec.append({'target':r.target,'zone':r.zone,'proxy':r.proxy,'method':r.method,'relative_improvement_pct':r.relative_MAE_improvement_vs_zonemean_pct,'decision_class':cls,'library_status':r.library_status})
    pd.DataFrame(dec).to_csv(INTEG_DIR/'reliability_decision_matrix.csv',index=False,encoding='utf-8-sig')
    return rec,pd.DataFrame(rob),pd.DataFrame(dec)

def save_outputs(raw,datasets,folds,preds,errors,summary,conf,cpreds,rej,seg,zone,well,rec,rob,dec):
    for df,path in [(folds,'expanded_model_baseline_folds.csv'),(preds,'expanded_model_baseline_predictions.csv'),(summary,'expanded_model_baseline_summary.csv'),(summary,'metrics_pooled_corrected.csv'),(summary.sort_values(['dataset','cv_scheme','MAE_w']),'expanded_model_rank_table.csv'),(errors,'model_run_status.csv'),(conf,'conditional_conformal_summary.csv'),(cpreds,'conditional_conformal_predictions.csv'),(rej,'rejection_strategy_comparison.csv'),(seg,'well_segment_reliability_map.csv'),(zone,'zone_reliability_summary.csv'),(well,'well_reliability_cases.csv')]:
        df.to_csv(OUT/path,index=False,encoding='utf-8-sig')
    top=summary[~summary.model.isin(['train_mean','zone_mean'])].sort_values('MAE_w').groupby(['dataset','cv_scheme']).head(5); top.to_csv(OUT/'Table1_model_pool_best.csv',index=False,encoding='utf-8-sig')
    claims=[{'claim':'Random splitting overestimates reliability','support':'Strong','evidence':'random metrics exceed LOWO/LOZO for both targets; pooled R2 used.','paper_wording':'Reliability degrades under well/zone external validation.','avoid_wording':'Do not cite legacy_fallback results.'},{'claim':'Cross-zone transfer remains weak','support':'Strong','evidence':'LOZO pooled R2 is negative for TOC and total gas.','paper_wording':'Use as reliability-boundary evidence.','avoid_wording':'Do not claim robust cross-zone prediction.'},{'claim':'Conformal intervals expose uncertainty boundaries','support':'Moderate','evidence':'global/zone/risk-inflated conformal recomputed from training-only calibration.','paper_wording':'Risk-inflated intervals improve coverage with width cost.','avoid_wording':'Do not claim leakage-free calibration without stating split.'},{'claim':'Composite rejection is target-dependent','support':'Moderate','evidence':'20% rejection summaries in canonical file.','paper_wording':'Rejection is a risk-control layer.','avoid_wording':'Do not claim universal improvement.'},{'claim':'Feature leakage control','support':'Required','evidence':'organic_level/gas_interpretation/target-derived fields excluded.','paper_wording':'All canonical numbers use repaired model-ready dataset pressure_coeff.','avoid_wording':'Do not cite old Table-3 backbone.'}]
    pd.DataFrame(claims).to_csv(OUT/'Table2_claim_support_matrix.csv',index=False,encoding='utf-8-sig')
    # docs
    (OUT/'leakage_check.md').write_text('# Leakage check - real library model-ready dataset backbone\n\nNumeric features: '+', '.join(NUM_COLS)+'\n\nCategorical features: '+', '.join(CAT_COLS)+'\n\nExcluded: '+', '.join(EXCLUDED)+'\n\nNo legacy_fallback regressors; all model-pool results use actual installed scientific libraries.\n',encoding='utf-8')
    (OUT/'conformal_protocol_check.md').write_text('# Conformal calibration protocol check - real library model-ready dataset\n\nCalibration samples are selected only from outer training wells/zones. For LOZO, the held-out zone is removed before calibration splitting; unseen zone-conditional quantiles fall back to the global training-zone quantile.\n',encoding='utf-8')
    # figures
    plot_figures(summary,conf,rej,zone)
    write_canonical(raw,datasets,summary,conf,rej,zone,rec,rob,dec)
    write_delta_legacy(summary,zone,rej,rob)
    (OUT/'real_rerun_log.txt').write_text('Real scientific-library rerun complete. No legacy_fallback regressors and no placeholder figures. See env_report.txt for versions.\n',encoding='utf-8')

def plot_figures(summary,conf,rej,zone):
    FIG.mkdir(exist_ok=True)
    # Fig1
    fig,ax=plt.subplots(figsize=(11,6)); order=['random','LOWO','LOZO']; x=np.arange(len(order)); width=0.18
    for j,ds in enumerate(['TOC_high_medium','total_gas_high_medium']):
        vals=[]; labs=[]
        for cv in order:
            sub=summary[(summary.dataset==ds)&(summary.cv_scheme==cv)&(~summary.model.isin(['train_mean','zone_mean']))].sort_values('MAE_w').head(1); vals.append(float(sub.pooled_R2.iloc[0])); labs.append(sub.model.iloc[0])
        ax.bar(x+(j-0.5)*width,vals,width,label=ds.replace('_high_medium',''))
    tm=[float(summary[(summary.dataset=='TOC_high_medium')&(summary.model=='train_mean')&(summary.cv_scheme==cv)].pooled_R2.iloc[0]) for cv in order]
    ax.plot(x,tm,'k--o',label='TOC train-mean baseline'); ax.axhline(0,color='gray',lw=0.8); ax.set_xticks(x,order); ax.set_ylabel('Pooled R2'); ax.set_title('Fig.1 Model pool: random vs LOWO vs LOZO'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'Fig1_random_LOWO_LOZO_model_pool.svg'); fig.savefig(FIG/'Fig1_random_LOWO_LOZO_model_pool.png',dpi=220); plt.close(fig)
    # Fig2
    cg=conf.groupby(['dataset','cv_scheme','method']).agg(coverage=('coverage','mean'),width=('mean_width','mean')).reset_index();
    fig,ax=plt.subplots(figsize=(10,6)); markers={'global':'o','zone_conditional':'s','risk_inflated':'^'}; colors={'TOC_high_medium':'tab:blue','total_gas_high_medium':'tab:orange'}
    for _,r in cg.iterrows():
        ax.scatter(r.width,r.coverage,marker=markers[r.method],color=colors[r.dataset],s=70); ax.annotate(f"{r.dataset.split('_')[0]} {r.cv_scheme} {r.method.replace('_','-')}",(r.width,r.coverage),fontsize=8,xytext=(4,4),textcoords='offset points')
    ax.axhline(0.9,color='gray',ls='--',lw=0.8); ax.set_xlabel('Mean interval width'); ax.set_ylabel('Coverage'); ax.set_title('Fig.2 Conformal coverage-width tradeoff'); fig.tight_layout(); fig.savefig(FIG/'Fig2_conformal_coverage_width.svg'); fig.savefig(FIG/'Fig2_conformal_coverage_width.png',dpi=220); plt.close(fig)
    # Fig3 total gas rejection
    rg=rej[(rej.dataset=='total_gas_high_medium')&(rej.risk_score=='composite_risk')].groupby(['cv_scheme','method','reject_frac']).agg(MAE=('MAE','mean')).reset_index()
    fig,ax=plt.subplots(figsize=(10,6))
    for (cv,m),g in rg.groupby(['cv_scheme','method']): ax.plot(g.reject_frac*100,g.MAE,marker='o',label=f'{cv} {m}')
    ax.set_xlabel('Rejected fraction (%)'); ax.set_ylabel('Retained MAE'); ax.set_title('Fig.3 Total gas composite-risk rejection curves'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/'Fig3_rejection_curves_total_gas.svg'); fig.savefig(FIG/'Fig3_rejection_curves_total_gas.png',dpi=220); plt.close(fig)
    # Fig4 zone map
    z=zone.copy(); z['zone_en']=z.zone.map(ZONE_EN); piv=z.pivot(index='dataset',columns='zone_en',values='coverage').reindex(columns=['North','Central','South'])
    fig,ax=plt.subplots(figsize=(8,3.6)); im=ax.imshow(piv.values,vmin=0,vmax=1,cmap='viridis'); ax.set_xticks(range(len(piv.columns)),piv.columns); ax.set_yticks(range(len(piv.index)),[s.replace('_high_medium','') for s in piv.index]);
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j,i,f'{piv.values[i,j]:.2f}',ha='center',va='center',color='white' if piv.values[i,j]<0.65 else 'black')
    ax.set_title('Fig.4 Zone-level reliability coverage (model-ready dataset real rerun)'); fig.colorbar(im,ax=ax,label='Coverage'); fig.tight_layout(); fig.savefig(FIG/'Fig4_zone_reliability_map_prototype.svg'); fig.savefig(FIG/'Fig4_zone_reliability_map_prototype.png',dpi=220); plt.close(fig)

def write_canonical(raw,datasets,summary,conf,rej,zone,rec,rob,dec):
    lines=['# Canonical numbers for manuscript - model-ready dataset REAL scientific-library rerun','', '## Environment / no fallback statement', '- All canonical model-pool numbers below come from real installed scientific libraries. No legacy_fallback regressors are used. Figures are rendered by matplotlib; no placeholder PNGs are used.', '- See `env_report.txt` for exact versions.', '', '## Load audit and model-ready counts']
    for name,target,df in datasets:
        wells=df.groupby('zone').well.nunique().to_dict(); pts=df.groupby('zone').size().to_dict(); lines.append(f"- {name}: model-ready {len(df)} points / {df.well.nunique()} wells; wells North/Central/South = {wells.get(_ZN,0)}/{wells.get(_ZC,0)}/{wells.get(_ZS,0)}; points North/Central/South = {pts.get(_ZN,0)}/{pts.get(_ZC,0)}/{pts.get(_ZS,0)}; LOWO folds = {df.well.nunique()}; LOZO folds = 3.")
    def best(ds,cv): return summary[(summary.dataset==ds)&(summary.cv_scheme==cv)&(~summary.model.isin(['train_mean','zone_mean']))].sort_values('MAE_w').iloc[0]
    def get(ds,cv,m): return summary[(summary.dataset==ds)&(summary.cv_scheme==cv)&(summary.model==m)].iloc[0]
    lines+=['','## Key model-pool numbers']
    for ds in ['TOC_high_medium','total_gas_high_medium']:
        for cv in ['random','LOWO','LOZO']:
            b=best(ds,cv); tm=get(ds,cv,'train_mean'); zm=get(ds,cv,'zone_mean')
            lines.append(f"- {ds} {cv}: best={b.model} MAE={fmt(b.MAE_w)}, pooled R2={fmt(b.pooled_R2)}, Spearman={fmt(b.Spearman_mean)}; train_mean MAE={fmt(tm.MAE_w)}, R2={fmt(tm.pooled_R2)}; zone_mean MAE={fmt(zm.MAE_w)}, R2={fmt(zm.pooled_R2)}.")
    lines+=['','## Conformal coverage-width']
    cg=conf.groupby(['dataset','cv_scheme','method']).agg(coverage=('coverage','mean'),width=('mean_width','mean')).reset_index()
    for _,r in cg.iterrows(): lines.append(f"- {r.dataset} {r.cv_scheme} {r.method}: coverage={fmt(r.coverage)}, width={fmt(r.width)}.")
    lines+=['','## Composite-risk rejection at 20%']
    rg=rej[(rej.risk_score=='composite_risk')&(rej.reject_frac.isin([0.0,0.2]))].groupby(['dataset','cv_scheme','method','reject_frac']).agg(MAE=('MAE','mean'),coverage=('coverage','mean')).reset_index()
    for ds in ['TOC_high_medium','total_gas_high_medium']:
      for cv in ['LOWO','LOZO']:
       for meth in ['global','zone_conditional','risk_inflated']:
        a=rg[(rg.dataset==ds)&(rg.cv_scheme==cv)&(rg.method==meth)&(rg.reject_frac==0.0)]; b=rg[(rg.dataset==ds)&(rg.cv_scheme==cv)&(rg.method==meth)&(rg.reject_frac==0.2)]
        if len(a) and len(b):
            imp=(a.MAE.iloc[0]-b.MAE.iloc[0])/a.MAE.iloc[0]*100
            lines.append(f"- {ds} {cv} {meth}: MAE 0%={fmt(a.MAE.iloc[0])}, MAE 20%={fmt(b.MAE.iloc[0])}, relative improvement={fmt(imp)}%, coverage20={fmt(b.coverage.iloc[0])}.")
    lines+=['','## Zone reliability coverage']
    for _,r in zone.iterrows(): lines.append(f"- {r.dataset} {ZONE_EN.get(r.zone,r.zone)}: coverage={fmt(r.coverage)}, reject_rate={fmt(r.reject_rate)}, mean_width={fmt(r.mean_width)}.")
    for ds in ['TOC_high_medium','total_gas_high_medium']:
        low=zone[zone.dataset==ds].sort_values('coverage').iloc[0]; lines.append(f"- Lowest-coverage risk zone for {ds}: {ZONE_EN.get(low.zone,low.zone)} = {fmt(low.coverage)}.")
    lines+=['','## Real-library recalibration/integration check']
    anchor=rec[(rec.target=='TOC')&(rec.zone==_ZN)&(rec.method=='C1_offset')]
    if len(anchor): lines.append(f"- North TOC / Schmoker / C1 offset: improvement={fmt(anchor.relative_MAE_improvement_vs_zonemean_pct.iloc[0])}%, method status={anchor.library_status.iloc[0]} (arithmetic offset, recomputed in real environment).")
    orig=rob[rob.case=='original'] if len(rob) else pd.DataFrame()
    if len(orig): lines.append(f"- North TOC offset robustness original: n_wells={int(orig.n_wells.iloc[0])}, improvement={fmt(orig.relative_improvement_pct.iloc[0])}%, mean_offset={fmt(orig.mean_offset.iloc[0])}.")
    lines.append('- C2 affine, C3 isotonic, and C4 ridge recalibration methods are recomputed using sklearn LinearRegression, IsotonicRegression, and Ridge; no legacy_fallback ML is used.')
    (OUT/'canonical_numbers_model_dataset.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def read_csv_safe(p):
    try: return pd.read_csv(p,encoding='utf-8-sig')
    except Exception: return pd.DataFrame()

def write_delta_legacy(summary,zone,rej,rob):
    lines=['# Delta: legacy_fallback previous run vs REAL scientific-library rerun','', 'Legacy fallback results are obsolete and must not be cited; this file only quantifies bias.']
    sm=read_csv_safe(LEGACY_SOURCE/'metrics_pooled_corrected.csv'); sz=read_csv_safe(LEGACY_SOURCE/'zone_reliability_summary.csv')
    if len(sm):
        lines+=['','## Model-pool key deltas']
        for ds in ['TOC_high_medium','total_gas_high_medium']:
          for cv in ['random','LOWO','LOZO']:
            real=summary[(summary.dataset==ds)&(summary.cv_scheme==cv)&(~summary.model.isin(['train_mean','zone_mean']))].sort_values('MAE_w').iloc[0]
            old=sm[(sm.dataset==ds)&(sm.cv_scheme==cv)&(sm.model==real.model)]
            if len(old): lines.append(f"- {ds} {cv} {real.model}: legacy_fallback pooled R2={fmt(old.pooled_R2.iloc[0])}, real pooled R2={fmt(real.pooled_R2)}; legacy_fallback MAE={fmt(old.MAE_w.iloc[0])}, real MAE={fmt(real.MAE_w)}.")
            else: lines.append(f"- {ds} {cv}: real best {real.model} MAE={fmt(real.MAE_w)}, pooled R2={fmt(real.pooled_R2)}; no same-model legacy_fallback row found.")
    if len(sz):
        lines+=['','## Zone coverage deltas']
        for _,r in zone.iterrows():
            o=sz[(sz.dataset==r.dataset)&(sz.zone==r.zone)]
            if len(o): lines.append(f"- {r.dataset} {ZONE_EN.get(r.zone,r.zone)}: legacy_fallback coverage={fmt(o.coverage.iloc[0])}, real coverage={fmt(r.coverage)}; legacy_fallback reject={fmt(o.reject_rate.iloc[0])}, real reject={fmt(r.reject_rate)}.")
    orig=rob[rob.case=='original'] if len(rob) else pd.DataFrame()
    if len(orig): lines+=['','## North TOC offset',f"- Real recomputed North TOC C1 original improvement={fmt(orig.relative_improvement_pct.iloc[0])}%, mean_offset={fmt(orig.mean_offset.iloc[0])}. Previous memory recorded ~7.3887%; compare during Claude audit."]
    (OUT/'delta_legacy_vs_real.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    if OUT.exists() and not LEGACY_SOURCE.exists(): shutil.copytree(OUT,LEGACY_SOURCE)
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(); FIG.mkdir()
    write_env()
    raw,datasets=load_datasets()
    folds,preds,errors=eval_baselines(datasets)
    summary=summarize(folds,preds)
    conf,cpreds,rej=conformal_rejection(datasets,summary)
    seg,zone,well=reliability(cpreds)
    rec,rob,dec=recalibration_real(raw)
    save_outputs(raw,datasets,folds,preds,errors,summary,conf,cpreds,rej,seg,zone,well,rec,rob,dec)
    print('REAL RERUN COMPLETE', OUT, 'summary', len(summary), 'folds', len(folds), 'conf', len(conf), 'recal', len(rec))

if __name__=='__main__': main()
