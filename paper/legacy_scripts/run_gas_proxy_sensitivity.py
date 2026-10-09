# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path('\u5199\u4f5c_\u88689/supplementary_analysis/gas_proxy_sensitivity')
PRIV = OUT / 'internal_review'
OUT.mkdir(parents=True, exist_ok=True)
PRIV.mkdir(parents=True, exist_ok=True)
INFILE = Path('Input Program/derived_dataset.xlsx')
ZONE_EN = {'\u5317':'North','\u4e2d':'Central','\u5357':'South'}

BASE = {
    'V_L_m3_per_tocwt': 2.5,
    'P_L_MPa': 6.0,
    'S_w_fraction': 0.50,
    'hydrostatic_gradient_MPa_per_m': 0.00981,
    'temperature_gradient_C_per_km': 25.0,
    'surface_temperature_C': 15.0,
    'z_factor': 1.0,
    'porosity_imputation': 'training-independent median of available porosity_pct for proxy-only calculation',
}

VL_GRID = [1.5, 2.5, 3.5]
PL_GRID = [3.0, 6.0, 9.0]
SW_GRID = [0.35, 0.50, 0.65]


def rank_corr(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3 or len(np.unique(a[m]))<2 or len(np.unique(b[m]))<2:
        return np.nan
    return float(spearmanr(a[m], b[m]).correlation)


def compute_proxy(df, V_L, P_L, S_w):
    d=df.copy()
    # Pressure coefficient converts hydrostatic pressure to estimated reservoir pressure.
    d['pressure_MPa'] = d['pressure_coeff'].astype(float) * BASE['hydrostatic_gradient_MPa_per_m'] * d['depth_md'].astype(float)
    # Temperature and Z are specified for reproducibility but not used to tune against labels.
    d['temperature_C'] = BASE['surface_temperature_C'] + BASE['temperature_gradient_C_per_km'] * d['depth_md'].astype(float) / 1000.0
    por_med = float(d['porosity_pct'].median(skipna=True))
    toc_med = float(d['toc_pct'].median(skipna=True))
    d['porosity_fraction'] = d['porosity_pct'].fillna(por_med).clip(lower=0, upper=40).astype(float) / 100.0
    d['toc_pct_proxy'] = d['toc_pct'].fillna(toc_med).clip(lower=0).astype(float)
    # Adsorbed-gas enrichment component: Langmuir occupancy scaled by available organic matter.
    occupancy = d['pressure_MPa'] / (P_L + d['pressure_MPa'])
    d['adsorbed_proxy'] = V_L * occupancy * d['toc_pct_proxy']
    # Free-gas enrichment component: proportional to effective gas pore volume and pressure.
    # It is a dimensionless proxy scaled to comparable magnitude, not an absolute gas reserve calculation.
    d['free_proxy'] = d['porosity_fraction'] * (1.0 - S_w) * d['pressure_MPa'] / BASE['z_factor']
    d['gas_proxy_index'] = d['adsorbed_proxy'] + d['free_proxy']
    return d


def main():
    df=pd.read_excel(INFILE, sheet_name='\u6570\u636e')
    for c in ['well','zone']:
        df[c]=df[c].astype(str)
    base_df=compute_proxy(df, BASE['V_L_m3_per_tocwt'], BASE['P_L_MPa'], BASE['S_w_fraction'])
    base_index=base_df['gas_proxy_index'].to_numpy(float)
    # Save an anonymized base proxy row table as private only; public sensitivity is aggregate.
    wells=sorted(base_df['well'].unique())
    wmap={w:f'W{i+1:03d}' for i,w in enumerate(wells)}
    private_rows=base_df[['well','zone','depth_md','pressure_MPa','temperature_C','toc_pct_proxy','porosity_fraction','adsorbed_proxy','free_proxy','gas_proxy_index']].copy()
    private_rows['well_anon']=private_rows['well'].map(wmap)
    private_rows['zone_en']=private_rows['zone'].map(ZONE_EN)
    private_rows.drop(columns=['well','zone']).to_csv(PRIV/'gas_proxy_base_rows_anoninternal_review.csv', index=False, encoding='utf-8-sig')

    param_rows=[
        {'parameter':'V_L','value':BASE['V_L_m3_per_tocwt'],'unit':'m3 per t rock per wt% TOC proxy scale','role':'Langmuir maximum adsorption scaling','source_or_assumption':'fixed baseline sensitivity value; not fitted to gas labels','sensitivity_values':'1.5; 2.5; 3.5'},
        {'parameter':'P_L','value':BASE['P_L_MPa'],'unit':'MPa','role':'Langmuir pressure half-saturation constant','source_or_assumption':'fixed baseline sensitivity value; not fitted to gas labels','sensitivity_values':'3; 6; 9'},
        {'parameter':'pressure conversion','value':BASE['hydrostatic_gradient_MPa_per_m'],'unit':'MPa/m times pressure coefficient and depth','role':'reservoir-pressure proxy','source_or_assumption':'hydrostatic-gradient convention multiplied by pressure coefficient','sensitivity_values':'held fixed'},
        {'parameter':'S_w','value':BASE['S_w_fraction'],'unit':'fraction','role':'water saturation in free-gas pore-volume component','source_or_assumption':'fixed baseline sensitivity value; not fitted to gas labels','sensitivity_values':'0.35; 0.50; 0.65'},
        {'parameter':'porosity','value':'porosity_pct if available; otherwise median-imputed','unit':'fraction after percent-to-fraction conversion','role':'free-gas pore-volume component','source_or_assumption':'non-gas-label field in source table; missing values median-imputed without target fitting','sensitivity_values':'held fixed'},
        {'parameter':'TOC input for adsorption component','value':'toc_pct if available; otherwise median-imputed','unit':'wt%','role':'organic-matter scaling for adsorption proxy','source_or_assumption':'measured TOC field; gas labels are not used. Report as traditional proxy input, not gas-supervised fitting.','sensitivity_values':'held fixed'},
        {'parameter':'temperature gradient','value':BASE['temperature_gradient_C_per_km'],'unit':'C/km','role':'reported for reproducibility; not used to tune proxy parameters','source_or_assumption':'fixed baseline assumption','sensitivity_values':'held fixed'},
        {'parameter':'Z-factor','value':BASE['z_factor'],'unit':'dimensionless','role':'free-gas scaling factor','source_or_assumption':'fixed idealized value for proxy index; not fitted to gas labels','sensitivity_values':'held fixed'},
    ]
    pd.DataFrame(param_rows).to_csv(OUT/'gas_proxy_params.csv', index=False, encoding='utf-8-sig')

    rows=[]
    for VL in VL_GRID:
        for PL in PL_GRID:
            for SW in SW_GRID:
                v=compute_proxy(df, VL, PL, SW)
                idx=v['gas_proxy_index'].to_numpy(float)
                top_base=set(np.where(base_index >= np.nanquantile(base_index, 0.9))[0])
                top_var=set(np.where(idx >= np.nanquantile(idx, 0.9))[0])
                overlap=len(top_base & top_var)/max(1,len(top_base | top_var))
                row={
                    'V_L':VL,'P_L':PL,'S_w':SW,
                    'mean_proxy':float(np.nanmean(idx)),
                    'median_proxy':float(np.nanmedian(idx)),
                    'p10_proxy':float(np.nanquantile(idx,0.1)),
                    'p90_proxy':float(np.nanquantile(idx,0.9)),
                    'spearman_vs_base_proxy':rank_corr(idx, base_index),
                    'top_decile_jaccard_vs_base':float(overlap),
                }
                for z_cn,z_en in ZONE_EN.items():
                    mask=v['zone'].eq(z_cn)
                    row[f'{z_en}_mean_proxy']=float(np.nanmean(idx[mask.to_numpy()])) if mask.any() else np.nan
                rows.append(row)
    sens=pd.DataFrame(rows)
    # Flag whether ranking is stable; no gas labels used.
    sens['rank_stability_flag']=np.where(sens['spearman_vs_base_proxy']>=0.95, 'very_high_rank_stability', np.where(sens['spearman_vs_base_proxy']>=0.90, 'high_rank_stability', 'moderate_or_lower_rank_stability'))
    sens.to_csv(OUT/'gas_proxy_sensitivity.csv', index=False, encoding='utf-8-sig')

    # Compact summary for report.
    summary={
        'n_rows':len(df),
        'n_wells':df['well'].nunique(),
        'base_mean_proxy':float(base_df['gas_proxy_index'].mean()),
        'base_median_proxy':float(base_df['gas_proxy_index'].median()),
        'min_spearman_vs_base':float(sens['spearman_vs_base_proxy'].min()),
        'median_spearman_vs_base':float(sens['spearman_vs_base_proxy'].median()),
        'min_top_decile_jaccard':float(sens['top_decile_jaccard_vs_base'].min()),
        'median_top_decile_jaccard':float(sens['top_decile_jaccard_vs_base'].median()),
    }
    pd.DataFrame([summary]).to_csv(PRIV/'gas_proxy_sensitivity_gas_proxy_sensitivity_summary.csv', index=False, encoding='utf-8-sig')

    lines=['# gas_proxy_sensitivity gas-proxy specification and parameter sensitivity', '', '## Scope and red line', '',
           '- This gas_proxy_sensitivity artifact documents a traditional gas-enrichment proxy and its non-supervised parameter sensitivity.',
           '- No measured total-gas labels are used to fit, choose, or tune proxy parameters.',
           '- The proxy is reported as a reproducibility and enrichment-map baseline, not as the locked ML model and not as evidence of robust cross-zone gas prediction.',
           '- Because no locked earlier gas-proxy implementation was found in the canonical scripts, this report states the formula and assumptions explicitly rather than implying hidden calibration.', '',
           '## Proxy formula', '',
           'For each sample row:', '',
           '`P = pressure_coeff * 0.00981 * depth_md`', '',
           '`adsorbed_proxy = V_L * P / (P_L + P) * TOC_proxy`', '',
           '`free_proxy = porosity_fraction * (1 - S_w) * P / Z`', '',
           '`gas_proxy_index = adsorbed_proxy + free_proxy`', '',
           'where `TOC_proxy` and porosity use available source-table values with median imputation only when missing. This creates a dimensionless enrichment index rather than an absolute reserve estimate.', '',
           '## Parameters', '',
           '| parameter | baseline | unit | sensitivity values | note |', '|---|---:|---|---|---|']
    for r in param_rows:
        lines.append(f"| {r['parameter']} | {r['value']} | {r['unit']} | {r['sensitivity_values']} | {r['source_or_assumption']} |")
    lines += ['', '## Sensitivity summary', '',
              f"- Rows evaluated: {summary['n_rows']} sample rows from {summary['n_wells']} wells.",
              f"- Baseline mean proxy index: {summary['base_mean_proxy']:.6f}; median: {summary['base_median_proxy']:.6f}.",
              f"- Across the V_L/P_L/S_w grid, Spearman rank correlation vs the baseline proxy ranges from {summary['min_spearman_vs_base']:.4f} to 1.0000, with median {summary['median_spearman_vs_base']:.4f}.",
              f"- Top-decile overlap vs baseline ranges from {summary['min_top_decile_jaccard']:.4f} to 1.0000, with median {summary['median_top_decile_jaccard']:.4f}.", '',
              '## Honest interpretation', '',
              '- The parameter scan tests proxy-index stability only. It does not validate the proxy against measured gas labels.',
              '- High rank stability would mean the enrichment ranking is not very sensitive to the scanned non-supervised parameters; lower top-decile overlap identifies cases where enrichment-map hot spots change.',
              '- The proxy should therefore be described as a transparent traditional baseline/sensitivity check, not as a supervised gas-model result.',
              '- This gas_proxy_sensitivity does not change the main current interpretation: total-gas prediction remains weak, descriptor-assisted, and boundary-limited.', '',
              '## Public files', '', '- `gas_proxy_params.csv`', '- `gas_proxy_sensitivity.csv`', '- `gas_proxy_sensitivity_report.md`', '', '## Validation', '', '- Public sensitive-token scan is stored in `internal_review/gas_proxy_sensitivity_sensitive_scan.csv`.', '']
    (OUT/'gas_proxy_sensitivity_report.md').write_text('\n'.join(lines), encoding='utf-8')

    sensitive_pat = re.compile(r"(?!)")  # public package: true-name QC pattern removed after de-identification
    scan=[]
    for p in [OUT/'gas_proxy_params.csv', OUT/'gas_proxy_sensitivity.csv', OUT/'gas_proxy_sensitivity_report.md']:
        txt=p.read_text(encoding='utf-8-sig' if p.suffix=='.csv' else 'utf-8', errors='ignore')
        hits=sensitive_pat.findall(txt)
        scan.append({'file':p.name,'sensitive_hits':len(hits),'examples':';'.join(sorted(set(hits))[:5])})
    pd.DataFrame(scan).to_csv(PRIV/'gas_proxy_sensitivity_sensitive_scan.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(scan).to_string(index=False))
    print(pd.DataFrame([summary]).to_string(index=False))
    if any(r['sensitive_hits'] for r in scan):
        raise SystemExit('Sensitive scan failed')

if __name__ == '__main__':
    main()
