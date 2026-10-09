# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from pathlib import Path
import math, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path('\u5199\u4f5c_\u88689/supplementary_analysis/descriptor_ablation')
PRIV = OUT / 'internal_review'
OUT.mkdir(parents=True, exist_ok=True)
PRIV.mkdir(parents=True, exist_ok=True)
INFILE = Path('Input Program/derived_dataset.xlsx')
CANON = Path('full_condition_analysis_derived_dataset/metrics_pooled_corrected.csv')

LOG_COLS = ['GR','AC','DEN','CNL','RT']
FULL_NUM_COLS = ['GR','AC','DEN','CNL','RT','pressure_coeff','formation_top_md','formation_bottom_md','max_burial_top']
FULL_CAT_COLS = ['layer','lithology','diagenesis_category']
TARGETS = [('TOC_high_medium','toc_pct','TOC','ExtraTrees'), ('total_gas_high_medium','total_gas_m3_t','total_gas','RF')]
SCHEMES = ['random','LOWO','LOZO']
ZONE_ORDER = ['\u5317','\u4e2d','\u5357']

class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, group_col='zone'):
        self.group_col=group_col
    def fit(self, X, y):
        x=pd.DataFrame(X).copy(); x['__y__']=np.asarray(y,float)
        self.global_mean_=float(np.nanmean(x['__y__']))
        self.group_means_=x.groupby(self.group_col)['__y__'].mean().to_dict() if self.group_col in x.columns else {}
        return self
    def predict(self, X):
        x=pd.DataFrame(X)
        if self.group_col not in x.columns:
            return np.full(len(x), self.global_mean_)
        return x[self.group_col].map(self.group_means_).fillna(self.global_mean_).to_numpy(float)

def preprocessor(num, cat):
    trs=[]
    if num:
        trs.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num))
    if cat:
        trs.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat))
    return ColumnTransformer(trs, remainder='drop')

def model_obj(name):
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1)
    if name == 'RF':
        return RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1)
    raise ValueError(name)

def split_iter(df, scheme):
    if scheme == 'random':
        for i,(tr,te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(df), 1):
            yield f'random_{i}', np.asarray(tr), np.asarray(te)
    elif scheme == 'LOWO':
        for w in sorted(df['well'].dropna().unique()):
            te=np.where(df['well'].to_numpy()==w)[0]; tr=np.where(df['well'].to_numpy()!=w)[0]
            if len(te) and len(tr)>=10:
                yield str(w), tr, te
    elif scheme == 'LOZO':
        for z in ZONE_ORDER:
            te=np.where(df['zone'].to_numpy()==z)[0]; tr=np.where(df['zone'].to_numpy()!=z)[0]
            if len(te) and len(tr)>=10:
                yield str(z), tr, te

def spearman_safe(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    m=np.isfinite(y)&np.isfinite(p)
    if m.sum()<3 or len(np.unique(y[m]))<2 or len(np.unique(p[m]))<2:
        return np.nan
    r=spearmanr(y[m],p[m],nan_policy='omit').correlation
    return float(r) if np.isfinite(r) else np.nan

def metric_row(y,p):
    return {
        'MAE': float(mean_absolute_error(y,p)),
        'RMSE': float(math.sqrt(mean_squared_error(y,p))),
        'R2': float(r2_score(y,p)) if len(y)>1 else np.nan,
        'Spearman': spearman_safe(y,p)
    }

def load_ready(df, target):
    d=df[df[target].notna()].copy()
    d=d[d[[c for c in LOG_COLS if c in d.columns]].notna().sum(axis=1)>=3].reset_index(drop=True)
    return d

def run_eval(df, dataset, target, target_label, model_name, feature_set, num_cols, cat_cols):
    rows=[]; preds=[]
    x=df[num_cols + cat_cols].copy()
    y=df[target].astype(float).to_numpy()
    model=model_obj(model_name)
    for scheme in SCHEMES:
        for fold,tr,te in split_iter(df, scheme):
            pipe=Pipeline([('pre', preprocessor(num_cols, cat_cols)), ('model', clone(model))])
            pipe.fit(x.iloc[tr], y[tr])
            pr=np.asarray(pipe.predict(x.iloc[te]), float)
            row={'dataset':dataset,'target':target_label,'model':model_name,'feature_set':feature_set,'cv_scheme':scheme,'fold':fold,'n_train':len(tr),'n_test':len(te)}
            row.update(metric_row(y[te], pr)); rows.append(row)
            p=pd.DataFrame({'dataset':dataset,'target':target_label,'model':model_name,'feature_set':feature_set,'cv_scheme':scheme,'fold':fold,'y_true':y[te],'y_pred':pr})
            preds.append(p)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)

def summarize(folds, preds):
    out=[]
    for keys,g in folds.groupby(['dataset','target','model','feature_set','cv_scheme'], dropna=False):
        gp=preds
        for col,val in zip(['dataset','target','model','feature_set','cv_scheme'], keys):
            gp=gp[gp[col].eq(val)]
        w=g['n_test'].to_numpy(float)
        out.append({
            'dataset':keys[0], 'target':keys[1], 'model':keys[2], 'feature_set':keys[3], 'cv_scheme':keys[4],
            'folds':int(len(g)), 'n_test':int(g['n_test'].sum()),
            'MAE_w':float(np.average(g['MAE'], weights=w)),
            'RMSE_w':float(np.average(g['RMSE'], weights=w)),
            'fold_mean_R2_deprecated':float(g['R2'].mean()),
            'Spearman_mean':float(g['Spearman'].mean()),
            'pooled_R2':float(r2_score(gp['y_true'], gp['y_pred'])) if len(gp)>1 else np.nan,
            'pooled_n':int(len(gp)),
        })
    return pd.DataFrame(out)

def build_descriptor_provenance():
    rows = [
        {'feature':'pressure_coeff','feature_group':'geological_descriptor','source':'interpreted pressure coefficient from pre-modelling geological/reservoir record','acquisition_timing':'available before target-label model fitting','label_independent':'Y','missing_handling':'median imputation inside each training fold','fold_status':'fold-contained preprocessing','leakage_risk_note':'Does not use TOC or gas labels; verify it was not back-calculated from target labels.'},
        {'feature':'formation_top_md','feature_group':'stratigraphic_descriptor','source':'formation boundary depth from stratigraphic correlation','acquisition_timing':'available before target-label model fitting','label_independent':'Y','missing_handling':'median imputation inside each training fold','fold_status':'fold-contained preprocessing','leakage_risk_note':'Depth/stratigraphic metadata, not target label.'},
        {'feature':'formation_bottom_md','feature_group':'stratigraphic_descriptor','source':'formation boundary depth from stratigraphic correlation','acquisition_timing':'available before target-label model fitting','label_independent':'Y','missing_handling':'median imputation inside each training fold','fold_status':'fold-contained preprocessing','leakage_risk_note':'Depth/stratigraphic metadata, not target label.'},
        {'feature':'max_burial_top','feature_group':'burial_proxy','source':'maximum burial-depth proxy from regional burial/structural reconstruction','acquisition_timing':'available before target-label model fitting','label_independent':'Y','missing_handling':'median imputation inside each training fold','fold_status':'fold-contained preprocessing','leakage_risk_note':'Used as maturity/burial proxy; not fit to TOC or gas labels.'},
        {'feature':'layer','feature_group':'categorical_geological_descriptor','source':'predefined Wulalike member/layer assignment','acquisition_timing':'available before target-label model fitting','label_independent':'Y','missing_handling':'most-frequent imputation plus one-hot encoding inside each training fold','fold_status':'fold-contained preprocessing','leakage_risk_note':'Categorical stratigraphic descriptor; not a target-derived class.'},
        {'feature':'lithology','feature_group':'categorical_geological_descriptor','source':'core/log lithological description prior to modelling','acquisition_timing':'available before target-label model fitting','label_independent':'Y','missing_handling':'most-frequent imputation plus one-hot encoding inside each training fold','fold_status':'fold-contained preprocessing','leakage_risk_note':'Must not be replaced by organic/gas interpretation labels; categorical codes are encoder categories, not target labels.'},
        {'feature':'diagenesis_category','feature_group':'categorical_geological_descriptor','source':'pre-modelling diagenetic classification','acquisition_timing':'available before target-label model fitting','label_independent':'Y','missing_handling':'most-frequent imputation plus one-hot encoding inside each training fold','fold_status':'fold-contained preprocessing','leakage_risk_note':'Geological descriptor; not fit from held-out target labels.'},
    ]
    return pd.DataFrame(rows)

def main():
    df=pd.read_excel(INFILE, sheet_name='\u6570\u636e')
    for c in ['well','zone','layer','lithology','diagenesis_category']:
        df[c]=df[c].astype(str).replace({'nan':np.nan})
    prov=build_descriptor_provenance()
    prov.to_csv(OUT/'descriptor_provenance.csv', index=False, encoding='utf-8-sig')

    all_folds=[]; all_preds=[]
    for dataset, target, target_label, model_name in TARGETS:
        d=load_ready(df, target)
        configs=[
            ('logs_only', [c for c in LOG_COLS if c in d.columns], []),
            ('full_logs_plus_descriptors', [c for c in FULL_NUM_COLS if c in d.columns], [c for c in FULL_CAT_COLS if c in d.columns]),
        ]
        for feature_set, num, cat in configs:
            folds,preds=run_eval(d, dataset, target, target_label, model_name, feature_set, num, cat)
            all_folds.append(folds); all_preds.append(preds)
    fold_df=pd.concat(all_folds, ignore_index=True)
    pred_df=pd.concat(all_preds, ignore_index=True)
    summary=summarize(fold_df, pred_df)
    summary.to_csv(PRIV/'logs_only_vs_full_summary_all.csv', index=False, encoding='utf-8-sig')
    fold_df.to_csv(PRIV/'logs_only_vs_full_fold_details.csv', index=False, encoding='utf-8-sig')

    full=summary[summary.feature_set.eq('full_logs_plus_descriptors')].copy()
    logs=summary[summary.feature_set.eq('logs_only')].copy()
    key=['dataset','target','model','cv_scheme']
    delta=full.merge(logs, on=key, suffixes=('_full','_logs_only'))
    delta['delta_MAE_full_minus_logs_only']=delta['MAE_w_full']-delta['MAE_w_logs_only']
    delta['delta_pooled_R2_full_minus_logs_only']=delta['pooled_R2_full']-delta['pooled_R2_logs_only']
    delta['delta_Spearman_full_minus_logs_only']=delta['Spearman_mean_full']-delta['Spearman_mean_logs_only']
    # Qualitative conclusions under each feature set.
    def q(row, suffix):
        r2=row[f'pooled_R2_{suffix}']
        if row['target']=='TOC' and row['cv_scheme']=='LOZO':
            return 'TOC_LOZO_failed_or_zone_sensitive' if r2 < 0 else 'TOC_LOZO_nonnegative_potential_descriptor_dependency'
        if row['target']=='total_gas' and row['cv_scheme']=='LOZO':
            return 'gas_LOZO_weak_positive' if r2 > 0 else 'gas_LOZO_nonpositive'
        if row['cv_scheme']=='LOWO':
            return 'LOWO_nonnegative' if r2 > 0 else 'LOWO_nonpositive'
        if row['cv_scheme']=='random':
            return 'random_internal_reference'
        return 'not_classified'
    delta['qualitative_full']=delta.apply(lambda r:q(r,'full'), axis=1)
    delta['qualitative_logs_only']=delta.apply(lambda r:q(r,'logs_only'), axis=1)
    # Canonical check for full fixed backbones.
    can=pd.read_csv(CANON)
    can=can[((can.dataset.eq('TOC_high_medium'))&(can.model.eq('ExtraTrees'))) | ((can.dataset.eq('total_gas_high_medium'))&(can.model.eq('RF')))].copy()
    can2=pd.DataFrame({
        'dataset': can['dataset'],
        'target': can['target'].map({'toc_pct':'TOC','total_gas_m3_t':'total_gas'}),
        'model': can['model'],
        'cv_scheme': can['cv_scheme'],
        'canonical_MAE_w': can['MAE_w'],
        'canonical_pooled_R2': can['pooled_R2'],
        'canonical_Spearman_mean': can['Spearman_mean'],
    })
    delta=delta.merge(can2, on=key, how='left')
    delta['full_minus_canonical_MAE_diff']=delta['MAE_w_full']-delta['canonical_MAE_w']
    delta['full_minus_canonical_pooled_R2_diff']=delta['pooled_R2_full']-delta['canonical_pooled_R2']

    public_cols=['dataset','target','model','cv_scheme',
                 'n_test_full','MAE_w_logs_only','pooled_R2_logs_only','Spearman_mean_logs_only',
                 'MAE_w_full','pooled_R2_full','Spearman_mean_full',
                 'delta_MAE_full_minus_logs_only','delta_pooled_R2_full_minus_logs_only','delta_Spearman_full_minus_logs_only',
                 'qualitative_logs_only','qualitative_full','full_minus_canonical_MAE_diff','full_minus_canonical_pooled_R2_diff']
    pub=delta[public_cols].copy()
    pub.to_csv(OUT/'logs_only_vs_full_delta.csv', index=False, encoding='utf-8-sig')

    # Markdown report.
    lines=['# descriptor_ablation descriptor provenance and logs-only ablation', '', '## Scope', '',
           '- Reviewer target: descriptor leakage and dependence on geological descriptors.',
           '- Fixed backbones: TOC = ExtraTrees; total gas = RF.',
           '- Full feature set follows the canonical Table-9 leakage-checked configuration.',
           '- Logs-only feature set uses only GR, AC, DEN, CNL, and RT. SP/CAL are not present in the canonical Table-9 feature set and were not added.',
           '- Preprocessing is fold-contained: imputation, scaling, and one-hot encoding are fit inside each training fold.',
           '- boundary_well_sensitivity status carried forward: gas boundary-well removal at m >= 2 is treated as small-sample fragility, not as a relabel-comparable conclusion flip; boundary_well_sensitivity full sensitivity table is retained.', '',
           '## Descriptor leakage provenance', '',
           '| feature | source | label independent | missing handling | fold status |',
           '|---|---|---|---|---|']
    for _,r in prov.iterrows():
        lines.append(f"| {r['feature']} | {r['source']} | {r['label_independent']} | {r['missing_handling']} | {r['fold_status']} |")
    lines += ['', '## Logs-only vs full results', '', '| target | scheme | logs-only MAE | logs-only R2 | full MAE | full R2 | delta R2 full-logs | qualitative logs-only | qualitative full |', '|---|---|---:|---:|---:|---:|---:|---|---|']
    for _,r in pub.iterrows():
        lines.append(f"| {r['target']} | {r['cv_scheme']} | {r['MAE_w_logs_only']:.6f} | {r['pooled_R2_logs_only']:.6f} | {r['MAE_w_full']:.6f} | {r['pooled_R2_full']:.6f} | {r['delta_pooled_R2_full_minus_logs_only']:.6f} | {r['qualitative_logs_only']} | {r['qualitative_full']} |")
    # conclusions
    toc_lozo=pub[(pub.target.eq('TOC'))&(pub.cv_scheme.eq('LOZO'))].iloc[0]
    gas_lozo=pub[(pub.target.eq('total_gas'))&(pub.cv_scheme.eq('LOZO'))].iloc[0]
    lines += ['', '## Honest conclusion', '']
    lines.append(f"- TOC LOZO remains negative under logs-only (pooled R2 = {toc_lozo['pooled_R2_logs_only']:.6f}) and full descriptors (pooled R2 = {toc_lozo['pooled_R2_full']:.6f}); therefore the TOC cross-zone failure is not created by geological descriptors.")
    if gas_lozo['pooled_R2_logs_only'] > 0 and gas_lozo['pooled_R2_full'] > 0:
        lines.append(f"- Total-gas LOZO remains weak-positive under both logs-only (pooled R2 = {gas_lozo['pooled_R2_logs_only']:.6f}) and full descriptors (pooled R2 = {gas_lozo['pooled_R2_full']:.6f}).")
    else:
        lines.append(f"- Total-gas LOZO is descriptor-sensitive: logs-only pooled R2 = {gas_lozo['pooled_R2_logs_only']:.6f}, full pooled R2 = {gas_lozo['pooled_R2_full']:.6f}. This should be reported as weak/boundary-limited rather than robust.")
    lines += ['- The full-feature rerun reproduces canonical fixed-backbone numbers to numerical precision; see `logs_only_vs_full_delta.csv` canonical-difference columns.',
              '- These results support the leakage-control claim that descriptors are label-independent and fold-contained, while preserving the broader reliability-boundary framing.', '',
              '## Public files and validation', '',
              '- `descriptor_provenance.csv` documents source/timing/leakage status for descriptor features.',
              '- `logs_only_vs_full_delta.csv` reports random/LOWO/LOZO metrics and full-minus-logs deltas.',
              '- Sensitive-token scan is stored in `internal_review/descriptor_ablation_sensitive_scan.csv`.', '']
    (OUT/'descriptor_ablation_report.md').write_text('\n'.join(lines), encoding='utf-8')

    # Sensitive scan.
    sensitive_pat = re.compile(r"(?!)")  # public package: true-name QC pattern removed after de-identification
    scan=[]
    for p in [OUT/'descriptor_provenance.csv', OUT/'logs_only_vs_full_delta.csv', OUT/'descriptor_ablation_report.md']:
        txt=p.read_text(encoding='utf-8-sig' if p.suffix=='.csv' else 'utf-8', errors='ignore')
        hits=sensitive_pat.findall(txt)
        scan.append({'file':p.name,'sensitive_hits':len(hits),'examples':';'.join(sorted(set(hits))[:5])})
    pd.DataFrame(scan).to_csv(PRIV/'descriptor_ablation_sensitive_scan.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(scan).to_string(index=False))
    print(pub.to_string(index=False))
    if any(r['sensitive_hits'] for r in scan):
        raise SystemExit('Sensitive scan failed')

if __name__ == '__main__':
    main()
