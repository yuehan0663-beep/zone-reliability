"""Input schema and label filtering (paper §3.1–3.3)."""
from pathlib import Path
import numpy as np
import pandas as pd
from .constants import LOG_COLS, ZONE_EN

def validate_dataset(df, group_col='zone', well_col='well'):
    """Return a validated copy, mapping North/Central/South Chinese zone labels.

    Required: group, well, depth_md, GR/AC/DEN/CNL/RT, pressure_coeff,
    porosity_pct, and at least one target. Optional geological descriptors are
    selected when present. Original identities are never rewritten on disk.
    Configurable group and well columns are renamed only in the returned copy.
    """
    required=[group_col,well_col,'depth_md',*LOG_COLS,'pressure_coeff','porosity_pct']
    missing=[c for c in required if c not in df]
    if missing:raise ValueError('Missing required columns: '+', '.join(missing))
    if not any(c in df for c in ['toc_pct','total_gas_m3_t']):raise ValueError('At least one target column is required')
    out=df.copy().rename(columns={group_col:'zone',well_col:'well'})
    if out.columns.duplicated().any():raise ValueError('Grouping-column rename collides with existing names')
    if out[['zone','well']].isna().any().any():raise ValueError('Zone and well identities cannot be missing')
    for c in ['well','zone','layer','lithology','diagenesis_category']:
        if c in out:out[c]=out[c].astype(str).replace({'nan':np.nan})
    out['zone']=out.zone.map(lambda z:ZONE_EN.get(z,z))
    for c in ['toc_pct','total_gas_m3_t']:
        if c not in out:out[c]=np.nan
    for c in [*LOG_COLS,'depth_md','pressure_coeff','porosity_pct','toc_pct','total_gas_m3_t','formation_top_md','formation_bottom_md','max_burial_top']:
        if c in out:out[c]=pd.to_numeric(out[c],errors='raise')
    if 'sample_id' not in out:out['sample_id']=['S%06d'%i for i in range(len(out))]
    if out.sample_id.duplicated().any():raise ValueError('sample_id must identify rows uniquely')
    if 'layer' not in out:out['layer']='unspecified'
    return out

def read_dataset(path, sheet_name=None, group_col='zone', well_col='well'):
    """Read CSV or an explicitly selected workbook sheet; return validated rows.

    No file is written. For confidential workbook use the explicitly named source data sheet.
    """
    path=Path(path)
    if path.suffix.lower()=='.csv':df=pd.read_csv(path)
    elif path.suffix.lower() in ['.xlsx','.xls']:
        if sheet_name is None:raise ValueError('Workbook reads require an explicit sheet_name')
        df=pd.read_excel(path,sheet_name=sheet_name)
    else:raise ValueError('Supported formats: CSV, XLSX, XLS')
    return validate_dataset(df,group_col,well_col)

def target_datasets(df):
    """Select labelled rows with >=3 nonmissing logs, preserving source order.

    Targets are filtered independently: gas wells need not be a TOC subset.
    Returns (dataset name, target column, reset-index DataFrame) tuples (§3.3).
    """
    out=[]
    for name,target in [('TOC_high_medium','toc_pct'),('total_gas_high_medium','total_gas_m3_t')]:
        d=df[df[target].notna()].copy()
        d=d[d[[c for c in LOG_COLS if c in d.columns]].notna().sum(axis=1)>=3].reset_index(drop=True)
        d['cleaning_confidence']='high_medium'
        if len(d):out.append((name,target,d))
    return out
