"""Four-action rule from paper Phase III and Fig. 11."""
import pandas as pd

def decision_action(coverage_ok,support_ok,skill_class,local_correction_robust):
    """Return the published four-way action from explicitly supplied evidence.

    Positive directional skill is sufficient for the model-skill gate; local
    correction requires CI-robust evidence. Poor coverage overrides skill.
    """
    if coverage_ok and support_ok:
        return 'use with interval' if skill_class!='no_skill' else 'zone-level only'
    if local_correction_robust:return 'recalibrate locally'
    return 'acquire more data'

def decision_matrix(coverage,skill,benefits,support,coverage_threshold=.80):
    """Resolve target × zone decisions using protocol S24b's >=0.80 convention.

    support is an explicit {(target, zone): bool} assessment; no numerical
    minimum-well requirement is invented. The supplied coverage uses the selected global-variant protocols; local corrective evidence is a selected CI-robust
    result. Returns evidence and action for every available target-zone.

    The legacy zone summary pools global LOWO and LOZO records, as does
    S24b. Its >=0.80 threshold comes from run_protocol_sensitivity.py.
    The retained default is reported explicitly; a caller may request a labelled
    LOWO-only sensitivity through decision_protocols.
    """
    rows=[]
    for r in coverage.to_dict('records'):
        target,zone=r['target'],r['zone']
        s=skill[(skill.target==target)&(skill.zone_en==zone)&skill.predictor_group.str.startswith('A_')]
        cls=s.iloc[0].skill_class if len(s) else 'no_skill'
        tname='TOC' if target=='toc_pct' else 'total_gas'
        b=benefits[(benefits.target==tname)&(benefits.zone_en==zone)]
        robust=bool((b.benefit_class=='CI-robust').any())
        adequate=bool(support.get((target,zone),False))
        rows.append(dict(target=target,zone=zone,n_wells=r.get('n_wells'),coverage=r['coverage'],coverage_threshold=coverage_threshold,support_ok=adequate,skill_class=cls,local_correction_robust=robust,action=decision_action(r['coverage']>=coverage_threshold,adequate,cls,robust)))
    return pd.DataFrame(rows)
