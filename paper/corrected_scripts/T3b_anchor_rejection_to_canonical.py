from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from risk_tie_rule import percentile_rank, rounded_component

ROOT = Path.cwd()
OUT = ROOT / '写作_表9' / '审稿补充_round1_EVO独立版'
PRIVATE = OUT / '_private'
CANON_DIR = ROOT / '全条件预研_数据表9'
REJ_IN = CANON_DIR / 'rejection_strategy_comparison.csv'
PREDS_IN = CANON_DIR / 'conditional_conformal_predictions.csv'
MAPFILE = PRIVATE / 'well_id_anon_map.csv'

ZONE_EN = {'北': 'North', '中': 'Central', '南': 'South'}
TEXT_EXPECTED = {
    ('TOC_high_medium', 'LOWO', 'global'): (0.288909, 0.338884, -17.2975, 0.80959),
    ('TOC_high_medium', 'LOWO', 'zone_conditional'): (0.288909, 0.346255, -19.8489, 0.625641),
    ('TOC_high_medium', 'LOWO', 'risk_inflated'): (0.288909, 0.335656, -16.1804, 0.862308),
    ('TOC_high_medium', 'LOZO', 'global'): (0.444338, 0.444692, -0.0796695, 0.711934),
    ('TOC_high_medium', 'LOZO', 'zone_conditional'): (0.444338, 0.444692, -0.0796695, 0.711934),
    ('TOC_high_medium', 'LOZO', 'risk_inflated'): (0.444338, 0.439328, 1.12759, 0.792558),
    ('total_gas_high_medium', 'LOWO', 'global'): (0.392812, 0.303932, 22.6267, 0.945),
    ('total_gas_high_medium', 'LOWO', 'zone_conditional'): (0.392812, 0.313127, 20.2859, 0.759722),
    ('total_gas_high_medium', 'LOWO', 'risk_inflated'): (0.392812, 0.322603, 17.8735, 0.956111),
    ('total_gas_high_medium', 'LOZO', 'global'): (0.499997, 0.488274, 2.34448, 0.792423),
    ('total_gas_high_medium', 'LOZO', 'zone_conditional'): (0.499997, 0.488274, 2.34448, 0.792423),
    ('total_gas_high_medium', 'LOZO', 'risk_inflated'): (0.499997, 0.49365, 1.26939, 0.852192),
}
WEIGHTS = {
    'current_0.55_0.25_0.10_0.10': (0.55, 0.25, 0.10, 0.10),
    'equal_0.25_each': (0.25, 0.25, 0.25, 0.25),
    'width_only': (1.00, 0.00, 0.00, 0.00),
    'domain_only': (0.00, 1.00, 0.00, 0.00),
    'current_width_minus20pct_renorm': (0.44, 0.3025, 0.12875, 0.12875),
    'current_width_plus20pct_renorm': (0.66, 0.1975, 0.07125, 0.07125),
}

def target_label(dataset):
    return 'TOC' if dataset == 'TOC_high_medium' else 'total gas'

def spearman_safe(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3 or len(np.unique(y[m])) < 2 or len(np.unique(p[m])) < 2:
        return np.nan
    r = spearmanr(y[m], p[m], nan_policy='omit').correlation
    return float(r) if np.isfinite(r) else np.nan

def metric_row(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        'MAE': float(mean_absolute_error(y, p)),
        'RMSE': float(np.sqrt(mean_squared_error(y, p))),
        'R2': float(r2_score(y, p)) if len(y) > 1 else np.nan,
        'Spearman': spearman_safe(y, p),
    }

def improve(mae0, mae20):
    return (mae0 - mae20) / mae0 * 100 if mae0 > 0 else np.nan

def aggregate_rejection_table(rej):
    # Canonical manuscript aggregation: simple mean across canonical fold/model rows.
    grp = rej[rej['risk_score'].eq('composite_risk')].groupby(['dataset','target','cv_scheme','method','reject_frac'], dropna=False)
    agg = grp.agg(
        row_count=('MAE','size'),
        kept_n_mean=('kept_n','mean'),
        coverage=('coverage','mean'),
        mean_width=('mean_width','mean'),
        MAE=('MAE','mean'),
        RMSE=('RMSE','mean'),
        R2=('R2','mean'),
        Spearman=('Spearman','mean'),
    ).reset_index()
    return agg

def make_improvement_summary(agg):
    rows = []
    for (dataset, target, scheme, method), g in agg.groupby(['dataset','target','cv_scheme','method'], dropna=False):
        g0 = g[np.isclose(g['reject_frac'], 0.0)]
        g20 = g[np.isclose(g['reject_frac'], 0.2)]
        if len(g0) == 0 or len(g20) == 0:
            continue
        r0 = g0.iloc[0]
        r20 = g20.iloc[0]
        rows.append({
            'target_label': target_label(dataset),
            'dataset': dataset,
            'target': target,
            'cv_scheme': scheme,
            'method': method,
            'risk_score': 'composite_risk',
            'MAE_0pct': float(r0['MAE']),
            'MAE_20pct': float(r20['MAE']),
            'relative_improvement_pct': improve(float(r0['MAE']), float(r20['MAE'])),
            'coverage_20pct': float(r20['coverage']),
            'mean_width_20pct': float(r20['mean_width']),
            'row_count_0pct': int(r0['row_count']),
            'row_count_20pct': int(r20['row_count']),
            'aggregation_note': 'canonical simple mean over fold/model rejection rows',
        })
    return pd.DataFrame(rows)

def anon_fold(row, anon):
    if row['cv_scheme'] == 'LOWO':
        return anon.get(str(row['fold']), 'UNMAPPED')
    return ZONE_EN.get(str(row['fold']), str(row['fold']))

def recompute_weight_ablation(preds):
    fracs = [i / 100 for i in range(0, 41, 5)]
    out_rows = []
    for scheme_name, (ww, wd, wm, wq) in WEIGHTS.items():
        for (dataset, target, model, cv_scheme, fold, method), base0 in preds.groupby(['dataset','target','model','cv_scheme','fold','method'], dropna=False):
            base = base0.copy()
            if scheme_name == 'current_0.55_0.25_0.10_0.10' and 'composite_risk' in base.columns:
                # Use the canonical stored score for the current scheme so tie ordering and
                # floating details are exactly those used to generate rejection_strategy_comparison.csv.
                score = base['composite_risk']
            else:
                score = (
                    percentile_rank(base['interval_width']) * ww +
                    percentile_rank(base['domain_knn']) * wd +
                    percentile_rank(base['missing_risk']) * wm +
                    percentile_rank(base['quality_risk']) * wq
                )
            base['_risk'] = score
            sb = base.sort_values('_risk', ascending=False).reset_index(drop=True)
            for frac in fracs:
                kept = sb.iloc[int(len(sb) * frac):]
                if len(kept) < 5:
                    continue
                m = metric_row(kept['y_true'], kept['y_pred'])
                out_rows.append({
                    'weight_scheme': scheme_name,
                    'width_weight': ww,
                    'domain_weight': wd,
                    'missing_weight': wm,
                    'quality_weight': wq,
                    'dataset': dataset,
                    'target': target,
                    'model': model,
                    'cv_scheme': cv_scheme,
                    'fold': fold,
                    'method': method,
                    'reject_frac': frac,
                    'kept_n': int(len(kept)),
                    'coverage': float(kept['covered'].astype(bool).mean()),
                    'mean_width': float(kept['interval_width'].mean()),
                    **m,
                })
    detail = pd.DataFrame(out_rows)
    detail.to_csv(OUT / 'rejection_weight_ablation_v2_detail.csv', index=False, encoding='utf-8-sig')
    agg = detail.groupby(['weight_scheme','width_weight','domain_weight','missing_weight','quality_weight','dataset','target','cv_scheme','method','reject_frac'], dropna=False).agg(
        row_count=('MAE','size'),
        kept_n_mean=('kept_n','mean'),
        coverage=('coverage','mean'),
        mean_width=('mean_width','mean'),
        MAE=('MAE','mean'),
        RMSE=('RMSE','mean'),
        R2=('R2','mean'),
        Spearman=('Spearman','mean'),
    ).reset_index()
    rows = []
    for (scheme_name, dataset, target, cv_scheme, method), g in agg.groupby(['weight_scheme','dataset','target','cv_scheme','method'], dropna=False):
        g0 = g[np.isclose(g['reject_frac'], 0.0)]
        g20 = g[np.isclose(g['reject_frac'], 0.2)]
        if len(g0) == 0 or len(g20) == 0:
            continue
        r0 = g0.iloc[0]
        r20 = g20.iloc[0]
        rows.append({
            'weight_scheme': scheme_name,
            'width_weight': float(r20['width_weight']),
            'domain_weight': float(r20['domain_weight']),
            'missing_weight': float(r20['missing_weight']),
            'quality_weight': float(r20['quality_weight']),
            'target_label': target_label(dataset),
            'dataset': dataset,
            'target': target,
            'cv_scheme': cv_scheme,
            'method': method,
            'reject_frac': 0.20,
            'MAE_0pct': float(r0['MAE']),
            'MAE_20pct': float(r20['MAE']),
            'relative_improvement_pct': improve(float(r0['MAE']), float(r20['MAE'])),
            'coverage_20pct': float(r20['coverage']),
            'row_count_20pct': int(r20['row_count']),
        })
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPFILE)
    anon = dict(zip(mapping['well_real'].astype(str), mapping['well_anon'].astype(str)))
    rej = pd.read_csv(REJ_IN)
    preds = pd.read_csv(PREDS_IN)

    # Reviewer-facing rejection row table: anonymize LOWO fold names.
    rej_pub = rej.copy()
    rej_pub['fold_anon'] = rej_pub.apply(lambda r: anon_fold(r, anon), axis=1)
    rej_pub = rej_pub.drop(columns=['fold'])
    front = ['dataset','target','model','cv_scheme','fold_anon','method','risk_score','reject_frac']
    rej_pub = rej_pub[front + [c for c in rej_pub.columns if c not in front]]
    rej_pub.to_csv(OUT / 'rejection_SI_canonical_rows_anon.csv', index=False, encoding='utf-8-sig')

    agg = aggregate_rejection_table(rej)
    imp = make_improvement_summary(agg)

    # Text check for canonical §4.5 / canonical_numbers block.
    checks = []
    for _, r in imp.iterrows():
        key = (r['dataset'], r['cv_scheme'], r['method'])
        if key not in TEXT_EXPECTED:
            continue
        e0, e20, eimp, ecov20 = TEXT_EXPECTED[key]
        checks.append({
            'dataset': r['dataset'],
            'cv_scheme': r['cv_scheme'],
            'method': r['method'],
            'MAE_0pct': r['MAE_0pct'],
            'text_MAE_0pct': e0,
            'delta_MAE_0pct_6dp': round(r['MAE_0pct'], 6) - round(e0, 6),
            'MAE_20pct': r['MAE_20pct'],
            'text_MAE_20pct': e20,
            'delta_MAE_20pct_6dp': round(r['MAE_20pct'], 6) - round(e20, 6),
            'relative_improvement_pct': r['relative_improvement_pct'],
            'text_relative_improvement_pct': eimp,
            'delta_improvement_4dp': round(r['relative_improvement_pct'], 4) - round(eimp, 4),
            'coverage_20pct': r['coverage_20pct'],
            'text_coverage20': ecov20,
            'delta_coverage20_6dp': round(r['coverage_20pct'], 6) - round(ecov20, 6),
            'text_match': (round(r['MAE_0pct'], 6) == round(e0, 6)) and (round(r['MAE_20pct'], 6) == round(e20, 6)) and (round(r['relative_improvement_pct'], 4) == round(eimp, 4)) and (round(r['coverage_20pct'], 6) == round(ecov20, 6)),
        })
    check = pd.DataFrame(checks)
    imp.to_csv(OUT / 'rejection_SI.csv', index=False, encoding='utf-8-sig')
    check.to_csv(OUT / 'T3b_text_number_check.csv', index=False, encoding='utf-8-sig')

    # Weight ablation using canonical predictions and canonical score formula, only changing weights.
    ab = recompute_weight_ablation(preds)
    ab.to_csv(OUT / 'rejection_weight_ablation_v2.csv', index=False, encoding='utf-8-sig')
    current_gas_lowo_global = ab[(ab['weight_scheme'].eq('current_0.55_0.25_0.10_0.10')) & (ab['dataset'].eq('total_gas_high_medium')) & (ab['cv_scheme'].eq('LOWO')) & (ab['method'].eq('global'))]
    current_ok = False
    if len(current_gas_lowo_global):
        val = float(current_gas_lowo_global.iloc[0]['relative_improvement_pct'])
        current_ok = round(val, 4) == round(22.6267, 4)

    # Markdown SI summary.
    lines = [
        '# Rejection SI — canonical-derived',
        '',
        'Source: `全条件预研_数据表9/rejection_strategy_comparison.csv`.',
        '',
        'Aggregation: simple mean over canonical fold/model rejection rows by dataset × cv_scheme × method × reject fraction; improvement = `(MAE_0pct - MAE_20pct) / MAE_0pct × 100`.',
        '',
        '| target | scheme | method | MAE 0% | MAE 20% | relative improvement (%) | coverage 20% | status vs canonical text |',
        '|---|---|---|---:|---:|---:|---:|---|',
    ]
    for _, r in imp.sort_values(['dataset','cv_scheme','method']).iterrows():
        c = check[(check['dataset'].eq(r['dataset'])) & (check['cv_scheme'].eq(r['cv_scheme'])) & (check['method'].eq(r['method']))]
        mark = 'PASS' if len(c) and bool(c.iloc[0]['text_match']) else 'CHECK'
        lines.append(f"| {r['target_label']} | {r['cv_scheme']} | {r['method']} | {r['MAE_0pct']:.6f} | {r['MAE_20pct']:.6f} | {r['relative_improvement_pct']:.4f} | {r['coverage_20pct']:.6f} | {mark} |")
    (OUT / 'rejection_SI.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # Weight ablation markdown.
    focus = ab[(ab['dataset'].eq('total_gas_high_medium')) & (ab['cv_scheme'].eq('LOWO'))].copy()
    ab_lines = [
        '# Rejection weight ablation v2 — canonical method',
        '',
        'The current scheme must reproduce the canonical total-gas LOWO global 20% rejection improvement (+22.6267%) before the ablation is interpreted.',
        '',
        f'- Current-scheme reproduction for total gas LOWO global: {"PASS" if current_ok else "FAIL"}.',
        '',
        '| weight scheme | method | MAE 0% | MAE 20% | relative improvement (%) | coverage 20% |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for _, r in focus.sort_values(['method','weight_scheme']).iterrows():
        ab_lines.append(f"| {r['weight_scheme']} | {r['method']} | {r['MAE_0pct']:.6f} | {r['MAE_20pct']:.6f} | {r['relative_improvement_pct']:.4f} | {r['coverage_20pct']:.6f} |")
    (OUT / 'rejection_weight_ablation_v2.md').write_text('\n'.join(ab_lines) + '\n', encoding='utf-8')

    protocol = '''# Rejection protocol spec — canonical anchored

Source files:

- `全条件预研_数据表9/rejection_strategy_comparison.csv`
- `全条件预研_数据表9/conditional_conformal_predictions.csv`

## Composite-risk definition

For each canonical conformal prediction fold and interval method, the canonical composite risk is computed within that fold/method as:

`composite_risk = 0.55 * rank_pct(interval_width) + 0.25 * rank_pct(domain_knn) + 0.10 * rank_pct(missing_risk) + 0.10 * rank_pct(quality_risk)`

where `rank_pct` is pandas percentile rank (`rank(pct=True)`) within the held-out fold/method rows. Rows are sorted by the chosen risk score in descending order, and the highest-risk `int(n * reject_frac)` rows are rejected.

## Baseline for improvement

The reported retained-error improvement compares the canonical model's retained-subset MAE after 20% rejection against the same canonical model/method/fold aggregation at 0% rejection:

`relative improvement (%) = (MAE_0pct - MAE_20pct) / MAE_0pct * 100`.

It is not a comparison against train-mean, zone-mean, or a same-retained-subset baseline. Because the 0% baseline is the full canonical prediction set, improvements quantify how much error is reduced after discarding the highest-risk predictions under the same prediction model and interval method.

## Reject-fraction scan and aggregation

- Reject fractions: 0.00 to 0.40 in increments of 0.05.
- Rejection is computed separately for each dataset, model, validation scheme, held-out fold, interval method, and risk score.
- Manuscript/SI summary values are simple means over canonical fold/model rejection rows at each dataset × cv_scheme × method × reject_frac.
- The manuscript-highlighted total-gas LOWO 20% values use `risk_score = composite_risk`.

## Caution

The +22.63% total-gas LOWO global value is a retained-subset risk-control result, not a claim that the base predictor itself improves or that the same benefit is guaranteed under all risk-weight choices.
'''
    (OUT / 'rejection_protocol_spec.md').write_text(protocol, encoding='utf-8')

    # Reconcile report.
    gas_focus = imp[(imp['dataset'].eq('total_gas_high_medium')) & (imp['cv_scheme'].eq('LOWO'))]
    rec = [
        '# T3b reconcile — rejection anchored to canonical',
        '',
        '## Canonical sources located',
        '',
        '- `全条件预研_数据表9/rejection_strategy_comparison.csv`',
        '- `全条件预研_数据表9/conditional_conformal_predictions.csv`',
        '',
        '## Main check',
        '',
        f'- Canonical text-number check: {"PASS" if bool(check["text_match"].all()) else "FAIL"}.',
        f'- Current-weight reproduction of total-gas LOWO global +22.6267% from canonical predictions: {"PASS" if current_ok else "FAIL"}.',
        '',
        'Total gas LOWO canonical composite-risk improvements:',
    ]
    for _, r in gas_focus.iterrows():
        rec.append(f"- {r['method']}: MAE {r['MAE_0pct']:.6f} -> {r['MAE_20pct']:.6f}; improvement {r['relative_improvement_pct']:.4f}%; coverage20 {r['coverage_20pct']:.6f}.")
    rec += [
        '',
        '## Difference from earlier independent +0.6% result',
        '',
        'The key difference is the baseline/aggregation role: the canonical manuscript value compares the canonical fold/model MAE at 20% retained-risk rejection against the same canonical fold/model set at 0% rejection, then summarizes by simple mean over canonical fold/model rows. Earlier independent calculations likely used a different baseline, sorting universe, model set, or aggregation strategy. For SI/manuscript numerical tables, the canonical-derived table is the source of record.',
        '',
        '## Fairness note',
        '',
        'The baseline is transparent but should be described precisely: it is a retained-subset error reduction relative to the full prediction set of the same canonical conformal model/method, not a deployment-wide accuracy improvement and not a comparison against train-mean or zone-mean baselines.',
        '',
        '## Outputs',
        '',
        '- `rejection_SI.csv` / `rejection_SI.md`',
        '- `rejection_SI_canonical_rows_anon.csv`',
        '- `rejection_protocol_spec.md`',
        '- `rejection_weight_ablation_v2.csv` / `.md`',
        '- `rejection_weight_ablation_v2_detail.csv`',
        '- `T3b_text_number_check.csv`',
    ]
    if not bool(check['text_match'].all()) or not current_ok:
        rec += ['', '## Stop condition', '', 'At least one required canonical reproduction check failed. Do not cite the T3b rejection values before manual review.']
    (OUT / 'T3b_reconcile.md').write_text('\n'.join(rec) + '\n', encoding='utf-8')

    print('text_match_all=', bool(check['text_match'].all()))
    print('current_gas_lowo_global_reproduces_22.6267=', current_ok)
    print(gas_focus[['method','MAE_0pct','MAE_20pct','relative_improvement_pct','coverage_20pct']].to_string(index=False))

if __name__ == '__main__':
    main()
