from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path.cwd()
OUT = ROOT / '写作_表9' / '审稿补充_round1_EVO独立版'
PREDS_IN = ROOT / '全条件预研_数据表9' / 'conditional_conformal_predictions.csv'
REJ_IN = ROOT / '全条件预研_数据表9' / 'rejection_strategy_comparison.csv'

N_RANDOM = 5000
SEED = 20260629
REJECT_FRAC = 0.20
FOCUS_SCHEME = 'LOWO'
FOCUS_DATASETS = ['TOC_high_medium', 'total_gas_high_medium']
FOCUS_METHODS = ['global', 'risk_inflated', 'zone_conditional']

def target_label(dataset: str) -> str:
    return 'TOC' if dataset == 'TOC_high_medium' else 'total gas'

def improvement(mae0: float, mae20: float) -> float:
    return (mae0 - mae20) / mae0 * 100 if mae0 > 0 else np.nan

def canonical_text_like_summary(rej: pd.DataFrame) -> pd.DataFrame:
    sub = rej[(rej['risk_score'].eq('composite_risk')) & (rej['cv_scheme'].eq(FOCUS_SCHEME)) & (rej['dataset'].isin(FOCUS_DATASETS)) & (rej['method'].isin(FOCUS_METHODS))]
    agg = sub.groupby(['dataset','target','cv_scheme','method','reject_frac'], dropna=False).agg(
        MAE=('MAE','mean'), coverage=('coverage','mean'), rows=('MAE','size')
    ).reset_index()
    rows = []
    for (dataset, target, scheme, method), g in agg.groupby(['dataset','target','cv_scheme','method'], dropna=False):
        if not np.isclose(g['reject_frac'], 0.0).any() or not np.isclose(g['reject_frac'], REJECT_FRAC).any():
            continue
        r0 = g[np.isclose(g['reject_frac'], 0.0)].iloc[0]
        r20 = g[np.isclose(g['reject_frac'], REJECT_FRAC)].iloc[0]
        rows.append({
            'dataset': dataset, 'target': target, 'cv_scheme': scheme, 'method': method,
            'canonical_MAE_0pct': float(r0['MAE']),
            'canonical_MAE_20pct': float(r20['MAE']),
            'canonical_improvement_pct': improvement(float(r0['MAE']), float(r20['MAE'])),
            'canonical_rows_0pct': int(r0['rows']),
            'canonical_rows_20pct': int(r20['rows']),
        })
    return pd.DataFrame(rows)

def unit_table(preds: pd.DataFrame, dataset: str, target: str, method: str) -> pd.DataFrame:
    rows = []
    sub = preds[(preds['dataset'].eq(dataset)) & (preds['target'].eq(target)) & (preds['cv_scheme'].eq(FOCUS_SCHEME)) & (preds['method'].eq(method))]
    for (model, fold), g in sub.groupby(['model','fold'], dropna=False):
        y = g['y_true'].to_numpy(float)
        p = g['y_pred'].to_numpy(float)
        abs_err = np.abs(y - p)
        n = len(g)
        n_reject = int(n * REJECT_FRAC)
        eligible = (n_reject > 0) and (n - n_reject >= 5)
        full_mae = float(abs_err.mean())
        if eligible:
            order = np.argsort(-g['composite_risk'].to_numpy(float), kind='mergesort')
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[order[:n_reject]] = False
            risk_mae20 = float(abs_err[keep_mask].mean())
        else:
            risk_mae20 = np.nan
        rows.append({
            'dataset': dataset, 'target': target, 'method': method, 'model': model, 'fold': fold,
            'n': n, 'n_reject_20pct': n_reject, 'n_kept_20pct': n - n_reject,
            'eligible_for_20pct': eligible,
            'full_MAE': full_mae,
            'risk_MAE_20pct_from_predictions': risk_mae20,
        })
    return pd.DataFrame(rows)

def random_true_uniform(abs_errors_list: list[np.ndarray], n_rejects: list[int], baseline_mae: float, rng: np.random.Generator) -> np.ndarray:
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

def pure_simulation_check(preds: pd.DataFrame, dataset: str, target: str, method: str, rng: np.random.Generator) -> dict:
    # Pool all abs residuals and uniformly drop 20%; expectation should be approximately 0.
    sub = preds[(preds['dataset'].eq(dataset)) & (preds['target'].eq(target)) & (preds['cv_scheme'].eq(FOCUS_SCHEME)) & (preds['method'].eq(method))]
    abs_err = np.abs(sub['y_true'].to_numpy(float) - sub['y_pred'].to_numpy(float))
    baseline = float(abs_err.mean())
    n = len(abs_err)
    n_reject = int(n * REJECT_FRAC)
    imps = np.empty(N_RANDOM, dtype=float)
    for i in range(N_RANDOM):
        reject = rng.choice(n, size=n_reject, replace=False)
        keep = np.ones(n, dtype=bool)
        keep[reject] = False
        imps[i] = improvement(baseline, float(abs_err[keep].mean()))
    return {
        'pooled_random_mean_pct': float(np.mean(imps)),
        'pooled_random_sd_pct': float(np.std(imps, ddof=1)),
        'pooled_random_p05_pct': float(np.quantile(imps, 0.05)),
        'pooled_random_p50_pct': float(np.quantile(imps, 0.50)),
        'pooled_random_p95_pct': float(np.quantile(imps, 0.95)),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(PREDS_IN)
    rej = pd.read_csv(REJ_IN)
    canon = canonical_text_like_summary(rej)
    rng = np.random.default_rng(SEED)
    sim_rng = np.random.default_rng(SEED + 1)
    results = []
    unit_rows = []
    for _, c in canon.sort_values(['dataset','method']).iterrows():
        dataset, target, method = c['dataset'], c['target'], c['method']
        ut = unit_table(preds, dataset, target, method)
        unit_rows.append(ut)
        eligible = ut[ut['eligible_for_20pct']].copy()
        # Fair baseline: same model-fold units that are eligible for 20% rejection.
        baseline_eligible = float(eligible['full_MAE'].mean())
        risk_mae20_eligible = float(eligible['risk_MAE_20pct_from_predictions'].mean())
        risk_imp_eligible = improvement(baseline_eligible, risk_mae20_eligible)

        # True random on same eligible model-fold units; this should center around 0.
        abs_lists = []
        n_rejects = []
        sub = preds[(preds['dataset'].eq(dataset)) & (preds['target'].eq(target)) & (preds['cv_scheme'].eq(FOCUS_SCHEME)) & (preds['method'].eq(method))]
        for _, row in eligible.iterrows():
            g = sub[(sub['model'].eq(row['model'])) & (sub['fold'].eq(row['fold']))]
            abs_lists.append(np.abs(g['y_true'].to_numpy(float) - g['y_pred'].to_numpy(float)))
            n_rejects.append(int(row['n_reject_20pct']))
        rand = random_true_uniform(abs_lists, n_rejects, baseline_eligible, rng)
        p_ge = float(np.mean(rand >= risk_imp_eligible))
        percentile = float(np.mean(rand <= risk_imp_eligible) * 100)
        sim = pure_simulation_check(preds, dataset, target, method, sim_rng)
        results.append({
            'target_label': target_label(dataset),
            'dataset': dataset,
            'target': target,
            'cv_scheme': FOCUS_SCHEME,
            'method': method,
            'reject_frac': REJECT_FRAC,
            'canonical_MAE_0pct_text_like': c['canonical_MAE_0pct'],
            'canonical_MAE_20pct_text_like': c['canonical_MAE_20pct'],
            'canonical_improvement_pct_text_like': c['canonical_improvement_pct'],
            'canonical_rows_0pct': int(c['canonical_rows_0pct']),
            'canonical_rows_20pct': int(c['canonical_rows_20pct']),
            'eligible_units_20pct': int(len(eligible)),
            'all_units_0pct': int(len(ut)),
            'baseline_MAE_same_eligible_units': baseline_eligible,
            'risk_MAE_20pct_same_eligible_units': risk_mae20_eligible,
            'risk_improvement_pct_same_eligible_units': risk_imp_eligible,
            'random_iterations': N_RANDOM,
            'random_seed': SEED,
            'random_improvement_mean_pct': float(np.mean(rand)),
            'random_improvement_sd_pct': float(np.std(rand, ddof=1)),
            'random_improvement_p05_pct': float(np.quantile(rand, 0.05)),
            'random_improvement_p50_pct': float(np.quantile(rand, 0.50)),
            'random_improvement_p95_pct': float(np.quantile(rand, 0.95)),
            'empirical_p_random_ge_risk': p_ge,
            'risk_percentile_vs_random': percentile,
            **sim,
            'interpretation': 'risk_above_true_random_p<0.05' if p_ge < 0.05 else 'risk_not_above_true_random',
        })
    out = pd.DataFrame(results)
    pd.concat(unit_rows, ignore_index=True).to_csv(OUT / 'T3c2_unit_eligibility_diagnosis.csv', index=False, encoding='utf-8-sig')
    out.to_csv(OUT / 'rejection_vs_random_baseline_v2.csv', index=False, encoding='utf-8-sig')

    anchor = out[(out['dataset'].eq('total_gas_high_medium')) & (out['method'].eq('global'))]
    anchor_ok = len(anchor) == 1 and round(float(anchor.iloc[0]['canonical_improvement_pct_text_like']), 4) == round(22.6267, 4)
    true_random_ok = bool((out['random_improvement_mean_pct'].abs() < 0.5).all())
    pooled_sim_ok = bool((out['pooled_random_mean_pct'].abs() < 0.5).all())

    lines = [
        '# T3c-2 random baseline correctness diagnosis',
        '',
        '## Summary',
        '',
        f'- Canonical anchor total gas LOWO global +22.6267%: {"PASS" if anchor_ok else "FAIL"}.',
        f'- True uniform random baseline centered near 0% for model-fold eligible units: {"PASS" if true_random_ok else "FAIL"}.',
        f'- Pure pooled residual simulation centered near 0%: {"PASS" if pooled_sim_ok else "FAIL"}.',
        '',
        '## Bug diagnosed in T3c',
        '',
        'The mathematical review identified an invalid T3c v1 null distribution: its denominator used all canonical 0% MAE units, whereas random 20% retained MAE used only eligible model-fold units retaining at least five rows. Mismatched unit universes caused a systematic nonzero random benefit; this was baseline mismatch, not evidence of true random-rejection gain.',
        '',
        'T3c-2 correction: risk and random rejection use the same eligible model-fold units, whose 0% MAE also supplies the common baseline.',
        '',
        '## Corrected random-baseline results',
        '',
        '| target | method | canonical text-like improvement (%) | eligible-unit risk improvement (%) | true random mean ± sd (%) | random p05 / p50 / p95 (%) | p(random ≥ risk) | interpretation |',
        '|---|---|---:|---:|---:|---:|---:|---|',
    ]
    for _, r in out.sort_values(['dataset','method']).iterrows():
        lines.append(
            f"| {r['target_label']} | {r['method']} | {r['canonical_improvement_pct_text_like']:.4f} | {r['risk_improvement_pct_same_eligible_units']:.4f} | {r['random_improvement_mean_pct']:.4f} ± {r['random_improvement_sd_pct']:.4f} | {r['random_improvement_p05_pct']:.4f} / {r['random_improvement_p50_pct']:.4f} / {r['random_improvement_p95_pct']:.4f} | {r['empirical_p_random_ge_risk']:.4f} | {r['interpretation']} |"
        )
    lines += [
        '',
        '## Pure simulation check',
        '',
        '| target | method | pooled random mean ± sd (%) | pooled p05 / p50 / p95 (%) |',
        '|---|---|---:|---:|',
    ]
    for _, r in out.sort_values(['dataset','method']).iterrows():
        lines.append(f"| {r['target_label']} | {r['method']} | {r['pooled_random_mean_pct']:.4f} ± {r['pooled_random_sd_pct']:.4f} | {r['pooled_random_p05_pct']:.4f} / {r['pooled_random_p50_pct']:.4f} / {r['pooled_random_p95_pct']:.4f} |")
    lines += [
        '',
        '## Honest interpretation after correction',
        '',
    ]
    gas = out[out['dataset'].eq('total_gas_high_medium')]
    toc = out[out['dataset'].eq('TOC_high_medium')]
    if len(gas) and (gas['empirical_p_random_ge_risk'] < 0.05).all():
        lines.append('- Total gas LOWO: all three canonical risk-rejection methods are far above the corrected true-random baseline. The rejection benefit is not obtainable by uniformly discarding any 20% of predictions.')
    elif len(gas):
        lines.append('- Total gas LOWO: at least one method is not above corrected random; report method-specific limitations.')
    if len(toc) and not (toc['empirical_p_random_ge_risk'] < 0.05).any():
        lines.append('- TOC LOWO: risk rejection is not above corrected random, consistent with no useful rejection benefit for TOC.')
    elif len(toc):
        lines.append('- TOC LOWO: at least one method is above corrected random; inspect before broad claims.')
    lines += [
        '',
        '## Outputs',
        '',
        '- `rejection_vs_random_baseline_v2.csv`',
        '- `T3c2_unit_eligibility_diagnosis.csv`',
        '- `T3c2_random_baseline_diagnosis.md`',
        '- `T3c2_random_baseline_diagnosis.py`',
    ]
    (OUT / 'T3c2_random_baseline_diagnosis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out.to_string(index=False))
    print('anchor_ok=', anchor_ok, 'true_random_ok=', true_random_ok, 'pooled_sim_ok=', pooled_sim_ok)

if __name__ == '__main__':
    main()
