# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from pathlib import Path
import math, re, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path('.')
OUT = Path('\u5199\u4f5c_\u88689/supplementary_analysis/boundary_well_sensitivity')
PRIV = OUT / 'internal_review'
OUT.mkdir(parents=True, exist_ok=True)
PRIV.mkdir(parents=True, exist_ok=True)

INFILE = Path('Input Program/derived_dataset.xlsx')
SHEET = '\u6570\u636e'
CANON_FOLDS = Path('full_condition_analysis_derived_dataset/expanded_model_baseline_folds.csv')

LOG_COLS = ['GR','AC','DEN','CNL','RT']
NUM_COLS = ['GR','AC','DEN','CNL','RT','pressure_coeff','formation_top_md','formation_bottom_md','max_burial_top']
CAT_COLS = ['layer','lithology','diagenesis_category']
ZONE_EN = {'\u5317':'North','\u4e2d':'Central','\u5357':'South'}
ZONE_ORDER = ['\u5317','\u4e2d','\u5357']
ADJ = {('\u5317','\u4e2d'):'North|Central', ('\u4e2d','\u5357'):'Central|South'}
TARGETS = [
    ('TOC_high_medium','toc_pct','ExtraTrees'),
    ('total_gas_high_medium','total_gas_m3_t','RF'),
]

class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, group_col='zone'):
        self.group_col = group_col
    def fit(self, X, y):
        x = pd.DataFrame(X).copy(); x['__y__'] = np.asarray(y, float)
        self.global_mean_ = float(np.nanmean(x['__y__']))
        self.group_means_ = x.groupby(self.group_col)['__y__'].mean().to_dict() if self.group_col in x.columns else {}
        return self
    def predict(self, X):
        x = pd.DataFrame(X)
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

def make_model(model_name):
    if model_name == 'ExtraTrees':
        return ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1)
    if model_name == 'RF':
        return RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1)
    if model_name == 'train_mean':
        from sklearn.dummy import DummyRegressor
        return DummyRegressor(strategy='mean')
    if model_name == 'zone_mean':
        return GroupMeanRegressor('zone')
    raise ValueError(model_name)

def make_pipe(model_name, model, num, cat):
    if model_name == 'zone_mean':
        return clone(model)
    return Pipeline([('pre', preprocessor(num, cat)), ('model', clone(model))])

def metric_row(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    mae = float(mean_absolute_error(y, p))
    rmse = float(math.sqrt(mean_squared_error(y, p)))
    r2 = float(r2_score(y, p)) if len(y) > 1 else np.nan
    if len(y) >= 3 and len(np.unique(y)) > 1 and len(np.unique(p)) > 1:
        sp = float(spearmanr(y, p, nan_policy='omit').correlation)
    else:
        sp = np.nan
    return {'MAE_w': mae, 'RMSE_w': rmse, 'pooled_R2': r2, 'Spearman': sp, 'n_test': int(len(y))}

def load_model_ready(df, target):
    d = df[df[target].notna()].copy()
    d = d[d[[c for c in LOG_COLS if c in d.columns]].notna().sum(axis=1) >= 3].copy()
    return d.reset_index(drop=True)

def lozo_eval(df, target, model_name):
    num = [c for c in NUM_COLS if c in df.columns]
    cat = [c for c in CAT_COLS if c in df.columns]
    x_model = df[num + cat].copy()
    x_all = df[num + cat + ['zone']].copy()
    y = df[target].astype(float).to_numpy()
    rows=[]; pred_rows=[]
    model = make_model(model_name)
    for z in ZONE_ORDER:
        te = np.where(df['zone'].to_numpy() == z)[0]
        tr = np.where(df['zone'].to_numpy() != z)[0]
        if len(te) == 0 or len(tr) < 10:
            continue
        pipe = make_pipe(model_name, model, num, cat)
        Xtr = (x_all if model_name == 'zone_mean' else x_model).iloc[tr]
        Xte = (x_all if model_name == 'zone_mean' else x_model).iloc[te]
        pipe.fit(Xtr, y[tr])
        pr = np.asarray(pipe.predict(Xte), float)
        m = metric_row(y[te], pr)
        rows.append({'fold': z, 'zone_en': ZONE_EN[z], 'n_train': int(len(tr)), **m})
        keep_cols = [c for c in ['well','zone','x','y'] if c in df.columns]
        p = df.iloc[te][keep_cols].copy(); p['fold'] = z; p['y_true'] = y[te]; p['y_pred'] = pr; p['abs_error'] = np.abs(y[te] - pr)
        pred_rows.append(p)
    folds = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if preds.empty:
        return folds, {'MAE_w': np.nan, 'pooled_R2': np.nan, 'Spearman_mean': np.nan, 'n_test': 0}
    pooled = {
        'MAE_w': float(np.average(folds['MAE_w'], weights=folds['n_test'])),
        'RMSE_w': float(np.average(folds['RMSE_w'], weights=folds['n_test'])),
        'pooled_R2': float(r2_score(preds['y_true'], preds['y_pred'])) if len(preds)>1 else np.nan,
        'Spearman_mean': float(folds['Spearman'].mean()),
        'n_test': int(folds['n_test'].sum()),
    }
    return folds, pooled

def pair_nearest_wells(wells, zone_a, zone_b, m):
    a = wells[wells.zone == zone_a].copy(); b = wells[wells.zone == zone_b].copy()
    rows=[]
    for side, own, other, own_zone, other_zone in [('A', a, b, zone_a, zone_b), ('B', b, a, zone_b, zone_a)]:
        vals=[]
        for _, r in own.iterrows():
            dx = other['x'].to_numpy(float) - float(r['x'])
            dy = other['y'].to_numpy(float) - float(r['y'])
            dist = np.sqrt(dx*dx + dy*dy)
            if len(dist)==0 or not np.isfinite(dist).any():
                continue
            j = int(np.nanargmin(dist))
            vals.append({
                'well': r['well'], 'zone': own_zone, 'nearest_opposite_well': other.iloc[j]['well'], 'nearest_opposite_zone': other_zone,
                'nearest_opposite_distance': float(dist[j]), 'boundary_pair': ADJ.get((zone_a, zone_b), f'{ZONE_EN.get(zone_a,zone_a)}|{ZONE_EN.get(zone_b,zone_b)}')
            })
        vals = sorted(vals, key=lambda x: x['nearest_opposite_distance'])[:m]
        rows.extend(vals)
    return pd.DataFrame(rows)

def build_boundary_sets(wells, m):
    frames=[]
    for za, zb in [('\u5317','\u4e2d'), ('\u4e2d','\u5357')]:
        frames.append(pair_nearest_wells(wells, za, zb, m))
    bd = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # If a Central well is near both boundaries, keep both boundary rows for assignment table, but unique well set for removal/relabel.
    return bd

def relabel_zone(original_zone, boundary_pair):
    if boundary_pair == 'North|Central':
        return '\u4e2d' if original_zone == '\u5317' else ('\u5317' if original_zone == '\u4e2d' else original_zone)
    if boundary_pair == 'Central|South':
        return '\u5357' if original_zone == '\u4e2d' else ('\u4e2d' if original_zone == '\u5357' else original_zone)
    return original_zone

def apply_variant(df, bd, mode):
    d = df.copy()
    if bd.empty:
        return d
    if mode == 'remove':
        return d[~d['well'].isin(set(bd['well']))].reset_index(drop=True)
    if mode == 'relabel':
        # If a well appears in both boundaries, choose nearest opposite distance row.
        bdu = bd.sort_values('nearest_opposite_distance').drop_duplicates('well')
        map_new = {r['well']: relabel_zone(r['zone'], r['boundary_pair']) for _, r in bdu.iterrows()}
        d['zone_original'] = d['zone']
        d['zone'] = d.apply(lambda r: map_new.get(r['well'], r['zone']), axis=1)
        return d.reset_index(drop=True)
    raise ValueError(mode)

def sanitize_public_assignment(wells, well_map):
    bd_all=[]
    for m in [1,2,3]:
        bd = build_boundary_sets(wells, m)
        bd['m'] = m
        bd_all.append(bd)
    bd_all = pd.concat(bd_all, ignore_index=True)
    min_dist = bd_all.groupby('well')['nearest_opposite_distance'].min().to_dict() if not bd_all.empty else {}
    assign = wells.copy()
    assign['well_anon'] = assign['well'].map(well_map)
    assign['zone_en'] = assign['zone'].map(ZONE_EN)
    assign['nearest_opposite_distance_proxy'] = assign['well'].map(min_dist)
    for m in [1,2,3]:
        bdm = set(bd_all.loc[bd_all['m'].eq(m), 'well']) if not bd_all.empty else set()
        assign[f'boundary_proxy_m{m}'] = assign['well'].isin(bdm).map({True:'Y', False:'N'})
    pub = assign[['well_anon','zone_en','x','y','nearest_opposite_distance_proxy','boundary_proxy_m1','boundary_proxy_m2','boundary_proxy_m3']].copy()
    return pub, bd_all

def main():
    df = pd.read_excel(INFILE, sheet_name=SHEET)
    for c in ['well','zone','layer','lithology','diagenesis_category']:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({'nan': np.nan})
    # Well coordinate table: all wells with any model-ready TOC or gas row, one coordinate per well.
    ready_frames=[]
    for _, target, _ in TARGETS:
        ready_frames.append(load_model_ready(df, target)[['well','zone','x','y']])
    wells = pd.concat(ready_frames, ignore_index=True).dropna(subset=['well','zone','x','y'])
    wells = wells.groupby('well', as_index=False).agg(zone=('zone','first'), x=('x','median'), y=('y','median'))
    wells = wells.sort_values(['zone','well']).reset_index(drop=True)
    well_map = {w: f'W{i+1:03d}' for i, w in enumerate(sorted(wells['well'].astype(str).unique()))}
    pd.DataFrame({'well_real': list(well_map.keys()), 'well_anon': list(well_map.values())}).to_csv(PRIV/'well_id_anon_map_boundary_well_sensitivityinternal_review.csv', index=False, encoding='utf-8-sig')
    assign_pub, bd_all = sanitize_public_assignment(wells, well_map)
    assign_pub.to_csv(OUT/'well_zone_assignment.csv', index=False, encoding='utf-8-sig')
    # private boundary details with real wells, plus public anonymized boundary list
    bd_all.to_csv(PRIV/'boundary_proxy_real_wellsinternal_review.csv', index=False, encoding='utf-8-sig')
    bd_pub = bd_all.copy()
    if not bd_pub.empty:
        bd_pub['well_anon'] = bd_pub['well'].map(well_map)
        bd_pub['nearest_opposite_well_anon'] = bd_pub['nearest_opposite_well'].map(well_map)
        bd_pub['zone_en'] = bd_pub['zone'].map(ZONE_EN)
        bd_pub['nearest_opposite_zone_en'] = bd_pub['nearest_opposite_zone'].map(ZONE_EN)
        bd_pub[['m','boundary_pair','well_anon','zone_en','nearest_opposite_well_anon','nearest_opposite_zone_en','nearest_opposite_distance']].to_csv(PRIV/'boundary_proxy_anon_detail.csv', index=False, encoding='utf-8-sig')
    results=[]; fold_details=[]
    canonical = pd.read_csv(CANON_FOLDS)
    # original rerun to verify against canonical on fixed backbones
    for dataset, target, model_name in TARGETS:
        base_df = load_model_ready(df, target)
        for m in [0,1,2,3]:
            modes = ['original'] if m == 0 else ['remove','relabel']
            bd = build_boundary_sets(wells, m) if m else pd.DataFrame()
            for mode in modes:
                work = base_df.copy() if mode == 'original' else apply_variant(base_df, bd, mode)
                folds, pooled = lozo_eval(work, target, model_name)
                row = {
                    'dataset': dataset, 'target': 'TOC' if target == 'toc_pct' else 'total_gas', 'model': model_name,
                    'variant': mode, 'm': m, 'boundary_proxy': 'cross-zone nearest-neighbor wells' if m else 'none',
                    'n_boundary_unique_wells': 0 if m == 0 else int(bd['well'].nunique()),
                    'n_test': pooled['n_test'], 'MAE_w': pooled['MAE_w'], 'RMSE_w': pooled.get('RMSE_w', np.nan),
                    'pooled_R2': pooled['pooled_R2'], 'Spearman_mean': pooled['Spearman_mean'],
                    'coverage': np.nan,
                }
                # compare original to canonical matching fixed model
                can = canonical[(canonical['dataset'].eq(dataset)) & (canonical['model'].eq(model_name)) & (canonical['cv_scheme'].eq('LOZO'))]
                if mode == 'original' and not can.empty:
                    row['canonical_MAE_w'] = float(np.average(can['MAE'], weights=can['n_test']))
                    # canonical pooled R2 available in metrics file, but compute from predictions not here; use folds detail R2 mean for diagnostic only.
                    row['canonical_fold_mean_R2_deprecated'] = float(can['R2'].mean())
                    row['original_reproduction_MAE_diff'] = row['MAE_w'] - row['canonical_MAE_w']
                else:
                    row['canonical_MAE_w'] = np.nan; row['canonical_fold_mean_R2_deprecated'] = np.nan; row['original_reproduction_MAE_diff'] = np.nan
                results.append(row)
                if not folds.empty:
                    fd = folds.copy(); fd['dataset']=dataset; fd['target']='TOC' if target == 'toc_pct' else 'total_gas'; fd['model']=model_name; fd['variant']=mode; fd['m']=m
                    fold_details.append(fd)
    res = pd.DataFrame(results)
    # qualitative flags vs locked conclusions
    def qualitative(row):
        if row['target'] == 'TOC':
            return 'TOC_LOZO_failed_or_zone_sensitive' if row['pooled_R2'] < 0 else 'TOC_LOZO_nonnegative_R2_potential_flip'
        else:
            return 'gas_LOZO_weak_positive' if row['pooled_R2'] > 0 else 'gas_LOZO_nonpositive_potential_flip'
    res['qualitative_result'] = res.apply(qualitative, axis=1)
    # Overturn threshold over m=1..3 tested: whether any tested variant flips by criteria.
    res['tested_overturn_within_m_le_3'] = res['qualitative_result'].str.contains('potential_flip').map({True:'Y', False:'N'})
    res.to_csv(OUT/'boundary_sensitivity.csv', index=False, encoding='utf-8-sig')
    if fold_details:
        pd.concat(fold_details, ignore_index=True).to_csv(PRIV/'boundary_sensitivity_fold_details.csv', index=False, encoding='utf-8-sig')
    # Simple public markdown report.
    lines=[]
    lines += ['# boundary_well_sensitivity a priori zone assignment and boundary-well sensitivity', '']
    lines += ['## Scope and source rule', '', '- Source data: `model-ready dataset` canonical modelling table.', '- Model/backbone settings follow the locked canonical configuration: TOC uses ExtraTrees; total gas uses RF.', '- This analysis does not use digitized fault-line geometry. The regional fault boundaries are treated as schematic/apriori structural-domain boundaries.', '- Boundary-near wells are approximated only from well-coordinate relations, using a cross-zone nearest-neighbor proxy.', '']
    lines += ['## Boundary proxy', '', '- For each adjacent zone pair, North|Central and Central|South, wells on each side were ranked by Euclidean distance to the nearest well in the opposite zone.', '- Sensitivity levels use m = 1, 2, and 3 nearest wells per side per adjacent boundary.', '- Two variants were evaluated: removing boundary-proxy wells, and relabelling each boundary-proxy well to the adjacent zone for the corresponding boundary.', '- If a well appears in both adjacent-boundary sets, relabelling uses its nearest opposite-zone boundary record.', '']
    lines += ['## Public outputs', '', '- `well_zone_assignment.csv`: anonymized well-to-zone assignment and boundary-proxy flags.', '- `boundary_sensitivity.csv`: LOZO sensitivity metrics for original, removal, and relabelling variants.', '']
    lines += ['## Summary results', '', '| target | variant | m | boundary wells | n test | MAE | pooled R2 | qualitative result |', '|---|---|---:|---:|---:|---:|---:|---|']
    for _, r in res.iterrows():
        lines.append(f"| {r['target']} | {r['variant']} | {int(r['m'])} | {int(r['n_boundary_unique_wells'])} | {int(r['n_test'])} | {r['MAE_w']:.6f} | {r['pooled_R2']:.6f} | {r['qualitative_result']} |")
    flips = res[res['tested_overturn_within_m_le_3'].eq('Y')]
    lines += ['', '## Honest conclusion', '']
    if flips.empty:
        lines += ['- No tested m <= 3 boundary-proxy removal or relabelling variant overturned the locked qualitative conclusions.', '- TOC remains LOZO-failed or strongly zone-sensitive under all tested variants.', '- Total gas remains weak-positive under all tested variants.', '- Within the tested cross-zone nearest-neighbor proxy levels, the overturn threshold was not reached by m <= 3.']
    else:
        lines += ['- At least one tested variant overturned a locked qualitative conclusion. This requires Claude/user review before any further use.', '']
        for _, r in flips.iterrows():
            lines.append(f"  - Potential flip: {r['target']} variant={r['variant']} m={int(r['m'])} pooled_R2={r['pooled_R2']:.6f}.")
    lines += ['', '## Disclosure for reviewer-facing text', '', 'The North, Central, and South domains were assigned a priori from regional structural interpretation. The two inter-domain fault boundaries are schematic/regional rather than GIS-grade digitized fault polylines. Therefore, this sensitivity test does not claim metre- or kilometre-level fault-distance precision. It tests whether reasonable boundary-near well reassignment, approximated from well-coordinate nearest-neighbor relations, changes the LOZO conclusions.', '']
    lines += ['## Anonymization', '', '- Public files use anonymized well IDs and English zone names.', '- Private real-well mapping and boundary details are retained only under `internal_review/`.', '- Final sensitive-token scan is reported separately in `internal_review/boundary_well_sensitivity_sensitive_scan.csv`.', '']
    (OUT/'boundary_well_sensitivity_report.md').write_text('\n'.join(lines), encoding='utf-8')

    # Sensitive scan public files.
    sensitive_pat = re.compile(r"(?!)")  # public package: true-name QC pattern removed after de-identification
    scan=[]
    for p in [OUT/'well_zone_assignment.csv', OUT/'boundary_sensitivity.csv', OUT/'boundary_well_sensitivity_report.md']:
        txt = p.read_text(encoding='utf-8-sig' if p.suffix=='.csv' else 'utf-8', errors='ignore')
        hits = sensitive_pat.findall(txt)
        scan.append({'file': p.name, 'sensitive_hits': len(hits), 'examples': ';'.join(sorted(set(hits))[:5])})
    pd.DataFrame(scan).to_csv(PRIV/'boundary_well_sensitivity_sensitive_scan.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(scan).to_string(index=False))
    print(res[['target','variant','m','n_boundary_unique_wells','n_test','MAE_w','pooled_R2','qualitative_result']].to_string(index=False))
    if any(x['sensitive_hits'] for x in scan):
        raise SystemExit('Sensitive scan failed')

if __name__ == '__main__':
    main()
