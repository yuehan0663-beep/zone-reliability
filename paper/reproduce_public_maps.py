"""Recompute S27, corrected S28 and S29 using released derived inputs only."""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load_script(path,name):
    """Import an archived, reviewed script without invoking its CLI main."""
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    """Render no figures; compare regenerated tables with shipped values."""
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    root=Path(__file__).resolve().parent;a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'corrected_scripts'))
    base=load_script(root/'corrected_scripts'/'run_table_s27_to_s29_well_level_risk.py','_map_base')
    correction=load_script(root/'corrected_scripts'/'run_table_s28_s29_risk_correction.py','_map_correction')
    correction.plot_variogram=lambda *args,**kwargs:None
    inputs=root/'map_reproduction_inputs'
    s25=pd.read_csv(root/'data_derived'/'Table_S25_coring_priority.csv')
    s27=base.build_s27(s25)
    prior28=pd.read_csv(inputs/'Table_S28_well_level_conformal_risk.csv')
    prior29=pd.read_csv(inputs/'Table_S29_spatial_autocorrelation_test.csv')
    audit=pd.read_csv(inputs/'risk_component_audit.csv')
    s28,_=correction.build_s28_v2(prior28,audit)
    widths=correction.width_summary(s28)
    assert widths.cv_gate_passed.all()
    s29,_=correction.build_spatial_tests(s28,prior29,a.out)
    reports=[]
    for name,table in [('Table_S27_well_level_map_inputs.csv',s27),('Table_S28_well_level_conformal_risk.csv',s28),('Table_S29_spatial_autocorrelation_test.csv',s29)]:
        path=a.out/name;table.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.6f')
        expected=pd.read_csv(root/'data_derived'/name)
        actual=pd.read_csv(path)
        ok=list(actual)==list(expected) and len(actual)==len(expected)
        mismatches=[]
        for c in expected:
            if c not in actual:mismatches.append(c);continue
            if pd.api.types.is_numeric_dtype(expected[c]) and pd.api.types.is_numeric_dtype(actual[c]):
                equal=np.allclose(actual[c],expected[c],rtol=0,atol=1e-6,equal_nan=True)
            else:equal=actual[c].fillna('').astype(str).equals(expected[c].fillna('').astype(str))
            if not equal:mismatches.append(c)
        ok=ok and not mismatches
        reports.append(dict(table=name,passed=ok,differing_columns='; '.join(mismatches)))
    pd.DataFrame(reports).to_csv(a.out/'map_reproduction_gate.csv',index=False)
    print(pd.DataFrame(reports).to_string(index=False))
    if not all(r['passed'] for r in reports):raise SystemExit(2)
    print('PUBLIC MAP REPRODUCTION PASS')


if __name__=='__main__':main()
