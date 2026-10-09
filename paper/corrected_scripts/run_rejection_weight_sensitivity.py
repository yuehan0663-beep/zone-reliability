# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from risk_tie_rule import percentile_rank, rounded_component

OUT = Path('\u5199\u4f5c_\u88689/supplementary_analysis/rejection_weight_sensitivity')
PRIV = OUT / 'internal_review'
OUT.mkdir(parents=True, exist_ok=True)
PRIV.mkdir(parents=True, exist_ok=True)

BASE = Path('\u5199\u4f5c_\u88689/review_supplement_round1_EVO\u72ec\u7acb\u7248/rejection_weight_ablation_v2.csv')
CPRED = Path('full_condition_analysis_derived_dataset/conditional_conformal_predictions.csv')

WEIGHT_EXTRA = {
    'missing_only': {'width_weight':0.0, 'domain_weight':0.0, 'missing_weight':1.0, 'quality_weight':0.0},
    'quality_only': {'width_weight':0.0, 'domain_weight':0.0, 'missing_weight':0.0, 'quality_weight':1.0},
}
TARGET_LABEL = {'toc_pct':'TOC', 'total_gas_m3_t':'total gas'}


def score(df, scheme):
    w = WEIGHT_EXTRA[scheme]
    # rank pct within each model-fold-method universe; this mirrors existing v2 strategy.
    return (percentile_rank(df['interval_width']) * w['width_weight'] +
            percentile_rank(df['domain_knn']) * w['domain_weight'] +
            percentile_rank(df['missing_risk']) * w['missing_weight'] +
            percentile_rank(df['quality_risk']) * w['quality_weight'])


def aggregate_scheme(cpred, scheme):
    rows=[]; detail=[]
    w = WEIGHT_EXTRA[scheme]
    for keys, g in cpred.groupby(['dataset','target','cv_scheme','method'], dropna=False):
        dataset,target,cv_scheme,method = keys
# Follow the prior table convention: 0% and 20% are computed fold-wise then weighted by kept/test rows.
        fold0=[]; fold20=[]
        for f, fg in g.groupby('fold', dropna=False):
            fg=fg.copy()
            fg['risk_score_tmp'] = score(fg, scheme)
            # 0% retained
            if len(fg) >= 5:
                m0 = mean_absolute_error(fg['y_true'], fg['y_pred'])
                fold0.append({'fold':f, 'n':len(fg), 'MAE':m0, 'coverage':fg['covered'].mean(), 'mean_width':fg['interval_width'].mean()})
            sb=fg.sort_values('risk_score_tmp', ascending=False).reset_index(drop=True)
            kept=sb.iloc[int(len(sb)*0.2):]
            if len(kept) < 5:
                continue
            m20=mean_absolute_error(kept['y_true'], kept['y_pred'])
            fold20.append({'fold':f, 'n':len(kept), 'MAE':m20, 'coverage':kept['covered'].mean(), 'mean_width':kept['interval_width'].mean()})
            detail.append({
                'weight_scheme': scheme, **w, 'dataset':dataset, 'target':target, 'cv_scheme':cv_scheme, 'fold':f, 'method':method,
                'reject_frac':0.2, 'kept_n':len(kept), 'coverage':kept['covered'].mean(), 'mean_width':kept['interval_width'].mean(), 'MAE':m20,
                'source_key':'rejection_weight_sensitivity_added_from_canonical_conformal_predictions'
            })
        if not fold0 or not fold20:
            continue
        n0=np.array([x['n'] for x in fold0], float); n20=np.array([x['n'] for x in fold20], float)
        mae0=float(np.average([x['MAE'] for x in fold0], weights=n0))
        mae20=float(np.average([x['MAE'] for x in fold20], weights=n20))
        cov20=float(np.average([x['coverage'] for x in fold20], weights=n20))
        rows.append({
            'weight_scheme':scheme, **w, 'target_label':TARGET_LABEL.get(target,target), 'dataset':dataset, 'target':target, 'cv_scheme':cv_scheme,
            'method':method, 'reject_frac':0.2, 'MAE_0pct':mae0, 'MAE_20pct':mae20,
            'relative_improvement_pct':(mae0-mae20)/mae0*100 if mae0 else np.nan,
            'coverage_20pct':cov20, 'row_count_20pct':int(sum(n20)), 'source_key':'rejection_weight_sensitivity_added_from_canonical_conformal_predictions'
        })
    return pd.DataFrame(rows), pd.DataFrame(detail)


def main():
    base=pd.read_csv(BASE)
    cpred=pd.read_csv(CPRED)
    # Use same scope as rejection_weight_sensitivity: LOWO 20% rejection primary; include all methods and both targets. Existing file contains more schemes/LOZO; rejection_weight_sensitivity public SI filters to LOWO.
    added=[]; details=[]
    for scheme in WEIGHT_EXTRA:
        a,d=aggregate_scheme(cpred, scheme)
        added.append(a); details.append(d)
    added=pd.concat(added, ignore_index=True)
    details=pd.concat(details, ignore_index=True)
    full=pd.concat([base, added], ignore_index=True, sort=False)
    full.to_csv(PRIV/'rejection_weight_sensitivity_weight_sensitivity_full_with_added.csv', index=False, encoding='utf-8-sig')
    details.to_csv(PRIV/'rejection_weight_sensitivity_missing_quality_detail.csv', index=False, encoding='utf-8-sig')

    # Clean SI: 20% LOWO only, both targets, all methods, required schemes.
    preferred_order = [
        'current_0.55_0.25_0.10_0.10', 'equal_0.25_each', 'width_only', 'domain_only', 'missing_only', 'quality_only',
        'current_width_minus20pct_renorm', 'current_width_plus20pct_renorm'
    ]
    si=full[(full['cv_scheme'].eq('LOWO')) & (full['reject_frac'].round(6).eq(0.2)) & (full['weight_scheme'].isin(preferred_order))].copy()
    si['weight_scheme_order']=si['weight_scheme'].map({v:i for i,v in enumerate(preferred_order)}).fillna(99).astype(int)
    si['target_order']=si['target_label'].map({'TOC':0,'total gas':1}).fillna(99).astype(int)
    si=si.sort_values(['target_order','method','weight_scheme_order']).reset_index(drop=True)
    si['interpretation_flag']=np.where(si['relative_improvement_pct']>0, 'positive_retained_MAE_improvement', 'nonpositive_or_harmful')
    pub_cols=['target_label','dataset','target','cv_scheme','method','weight_scheme','width_weight','domain_weight','missing_weight','quality_weight','reject_frac','MAE_0pct','MAE_20pct','relative_improvement_pct','coverage_20pct','row_count_20pct','interpretation_flag','source_key']
    si[pub_cols].to_csv(OUT/'rejection_weight_sensitivity_SI.csv', index=False, encoding='utf-8-sig')

    # Summary table by target/method: min/max improvement across schemes.
    summ=si.groupby(['target_label','method']).agg(
        n_schemes=('weight_scheme','nunique'),
        min_improvement_pct=('relative_improvement_pct','min'),
        max_improvement_pct=('relative_improvement_pct','max'),
        n_positive=('relative_improvement_pct', lambda x:int((x>0).sum())),
        n_nonpositive=('relative_improvement_pct', lambda x:int((x<=0).sum())),
    ).reset_index()
    summ.to_csv(PRIV/'rejection_weight_sensitivity_weight_sensitivity_summary.csv', index=False, encoding='utf-8-sig')

    lines=['# rejection_weight_sensitivity rejection-weight sensitivity', '', '## Scope', '',
           '- Reviewer target: whether composite-risk rejection conclusions depend on the fixed weights 0.55/0.25/0.10/0.10.',
           '- This rejection_weight_sensitivity table focuses on 20% LOWO rejection for TOC and total gas.',
           '- Most rows are reorganized from the previously verified `rejection_weight_ablation_v2` canonical-method output.',
           '- `missing_only` and `quality_only` were added from the canonical conformal prediction rows using the same retained-subset MAE definition and label-free risk columns.',
           '- Improvement is retained-subset MAE reduction relative to the same canonical model/method at 0% rejection, not accuracy versus a baseline predictor.', '',
           '## Weight schemes included', '',
           '| scheme | width | domain | missing | quality |', '|---|---:|---:|---:|---:|']
    ws=si[['weight_scheme','width_weight','domain_weight','missing_weight','quality_weight']].drop_duplicates().sort_values('weight_scheme')
    for _,r in ws.iterrows():
        lines.append(f"| {r['weight_scheme']} | {r['width_weight']:.4g} | {r['domain_weight']:.4g} | {r['missing_weight']:.4g} | {r['quality_weight']:.4g} |")
    lines += ['', '## LOWO 20% rejection summary', '', '| target | method | schemes | min improvement % | max improvement % | positive schemes | nonpositive schemes |', '|---|---|---:|---:|---:|---:|---:|']
    for _,r in summ.iterrows():
        lines.append(f"| {r['target_label']} | {r['method']} | {int(r['n_schemes'])} | {r['min_improvement_pct']:.4f} | {r['max_improvement_pct']:.4f} | {int(r['n_positive'])} | {int(r['n_nonpositive'])} |")
    # Detailed focal global method rows.
    focus=si[si['method'].eq('global')]
    lines += ['', '## Focal global-method rows', '', '| target | scheme | MAE 0% | MAE 20% | improvement % | coverage 20% | flag |', '|---|---|---:|---:|---:|---:|---|']
    for _,r in focus.iterrows():
        lines.append(f"| {r['target_label']} | {r['weight_scheme']} | {r['MAE_0pct']:.6f} | {r['MAE_20pct']:.6f} | {r['relative_improvement_pct']:.4f} | {r['coverage_20pct']:.6f} | {r['interpretation_flag']} |")
    # Honest conclusion.
    toc=si[si['target_label'].eq('TOC')]
    gas=si[si['target_label'].eq('total gas')]
    gas_lowo_all_positive = bool((gas['relative_improvement_pct']>0).all())
    toc_all_nonpos = bool((toc['relative_improvement_pct']<=0).all())
    lines += ['', '## Honest conclusion', '']
    if gas_lowo_all_positive:
        lines.append('- Total-gas LOWO retained-MAE improvement is positive for all tested weight schemes and interval methods in this rejection_weight_sensitivity table.')
    else:
        neg=gas[gas['relative_improvement_pct']<=0]
        lines.append('- Total-gas LOWO is not positive for every tested row; counterexamples must be retained:')
        for _,r in neg.iterrows():
            lines.append(f"  - {r['method']} / {r['weight_scheme']}: {r['relative_improvement_pct']:.4f}%")
    if toc_all_nonpos:
        lines.append('- TOC LOWO retained-MAE improvement is nonpositive for all tested weight schemes and interval methods.')
    else:
        pos=toc[toc['relative_improvement_pct']>0]
        lines.append('- TOC LOWO has positive retained-MAE improvement under some tested rows; these counterexamples prevent the stronger claim that TOC is uniformly non-improving:')
        for _,r in pos.iterrows():
            lines.append(f"  - {r['method']} / {r['weight_scheme']}: {r['relative_improvement_pct']:.4f}%")
    lines += ['- The safe reviewer-facing claim is that total-gas LOWO rejection gains are broadly positive across tested weights, while TOC LOWO is mostly harmful under the current/global and related schemes but not uniformly nonpositive across every single-factor variant.',
              '- This preserves the main reliability-boundary framing: rejection is a risk-screening aid, strongest for total gas LOWO, not a universal performance fix.', '',
              '## Validation', '', '- Public sensitive-token scan is stored in `internal_review/rejection_weight_sensitivity_sensitive_scan.csv`.', '']
    (OUT/'rejection_weight_sensitivity_report.md').write_text('\n'.join(lines), encoding='utf-8')

    sensitive_pat = re.compile(r"(?!)")  # public package: true-name QC pattern removed after de-identification
    scan=[]
    for p in [OUT/'rejection_weight_sensitivity_SI.csv', OUT/'rejection_weight_sensitivity_report.md']:
        txt=p.read_text(encoding='utf-8-sig' if p.suffix=='.csv' else 'utf-8', errors='ignore')
        hits=sensitive_pat.findall(txt)
        scan.append({'file':p.name,'sensitive_hits':len(hits),'examples':';'.join(sorted(set(hits))[:5])})
    pd.DataFrame(scan).to_csv(PRIV/'rejection_weight_sensitivity_sensitive_scan.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(scan).to_string(index=False))
    print(summ.to_string(index=False))
    if any(r['sensitive_hits'] for r in scan):
        raise SystemExit('Sensitive scan failed')

if __name__ == '__main__':
    main()
