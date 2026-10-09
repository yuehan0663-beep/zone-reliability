"""Check released current-manuscript tables without confidential inputs."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data_derived'
def rows(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pick(table,**conditions):
    return next(r for r in rows(table) if all(r[k]==v for k,v in conditions.items()))
def check(value,expected,tolerance):
    assert abs(float(value)-expected)<=tolerance,(value,expected,tolerance)
def main():
    assert len(list((ROOT/'supplementary_tables').glob('Table_S*.csv')))==36
    check(pick('Table_S1_full_model_rank.csv',target='toc_pct',model='ExtraTrees',cv_scheme='LOZO')['pooled_R2'],-.786,.005)
    check(pick('Table_S1_full_model_rank.csv',target='total_gas_m3_t',model='RF',cv_scheme='LOZO')['pooled_R2'],.092,.005)
    check(pick('Table_S14_rejection_vs_random_baseline.csv',target='total_gas_m3_t',method='global')['risk_improvement_pct_same_eligible_units'],6.62913952092778,1e-6)
    check(pick('Table_S19_rejection_weight_sensitivity.csv',target='total_gas_m3_t',cv_scheme='LOWO',method='global',weight_scheme='current_0.55_0.25_0.10_0.10')['relative_improvement_pct'],21.509181247201752,.05)
    for r in rows('Table_S19_rejection_weight_sensitivity.csv'):
        assert r['informative_ranking']==('False' if r['weight_scheme']=='quality_only' else 'True')
    adaptive=[r for r in rows('Table_S21b_adaptive_rejection_results.csv') if r['target_role']=='gas_main']
    assert adaptive and all(r['win']=='False' for r in adaptive)
    for r in adaptive:check(r['mde_pct_points'],16.887,.01)
    for target,features,scheme,expected in [
        ('TOC','full','LOWO',.948),('total_gas','full','LOWO',.837),
        ('TOC','full','LOZO',.851),('total_gas','full','LOZO',.497),
        ('TOC','logs-only','LOWO',.933),('total_gas','logs-only','LOWO',.943)]:
        check(pick('Table_S22b_stability_summary.csv',target=target,feature_set=features,cv_scheme=scheme)['pairwise_spearman_mean'],expected,.001)
    ablation=pick('Table_S18_logs_only_vs_full.csv',target='TOC',cv_scheme='LOZO')
    check(ablation['pooled_R2_logs_only'],-.361,.001);check(ablation['pooled_R2_full'],-.786,.001)
    toc=pick('Table_S24c_schmoker_baseline_sensitivity.csv',zone_en='North',variant='baseline')
    check(toc['relative_improvement_pct'],7.3887045863,.01)
    check(toc['improvement_ci_low_pct'],1.1735790462,.01)
    check(toc['improvement_ci_high_pct'],25.2249995572,.01)
    for p in (ROOT/'supplementary_tables').glob('*.csv'):
        assert p.read_bytes()==(DATA/p.name).read_bytes(),p.name
    print('PUBLIC PAPER TABLE CHECK PASS: 36 current CSVs, corrected rejection and eight attribution scalars')
if __name__=='__main__':main()
