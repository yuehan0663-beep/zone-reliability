"""Generate an entirely simulated, partially overlapping three-zone example."""
import numpy as np
import pandas as pd

def generate_synthetic(seed=20261009, samples_per_well=6):
    """Return simulated logs, descriptors and labels, without reading source data.

    Labelled well counts North/Central/South are TOC 8/9/15 and gas 5/11/10.
    Overlap counts are 3/5/7; neither target's wells form a subset of the other.
    Six samples per well support the five-retained-row rejection rule at 20%.
    Synthetic zone shifts demonstrate diagnostics without reproducing the paper.
    """
    if samples_per_well < 6:
        raise ValueError('Use at least six rows per well to demonstrate rejection')
    rng = np.random.default_rng(seed)
    records=[]
    for zi,(zone,nt,ng,overlap) in enumerate(zip(
            ['North','Central','South'],[8,9,15],[5,11,10],[3,5,7])):
        memberships = [('both',True,True)]*overlap + [('toc',True,False)]*(nt-overlap) + [('gas',False,True)]*(ng-overlap)
        for wi,(kind,toc_label,gas_label) in enumerate(memberships):
            well=f'SYN_{zone[0]}_{wi+1:03d}'
            effect=rng.normal(0,.12)
            for j in range(samples_per_well):
                latent=rng.normal()
                depth=2200+450*zi+18*wi+4*j+rng.normal(0,2)
                gr=80+13*latent+12*zi+rng.normal(0,3)
                ac=235+14*latent-9*zi+rng.normal(0,3)
                den=2.57-.07*latent+.025*zi+rng.normal(0,.015)
                cnl=19+3*latent+zi+rng.normal(0,1)
                rt=np.exp(1.7+.30*latent-.12*zi+rng.normal(0,.1))
                toc=max(.05,1.2+.25*latent+.25*zi+effect+rng.normal(0,.08))
                gas=max(.05,.9+.20*latent+.08*zi+effect+rng.normal(0,.09))
                records.append(dict(sample_id=f'SYN_S{len(records)+1:05d}',well=well,zone=zone,
                    depth_md=depth,GR=gr,AC=ac,DEN=den,CNL=cnl,RT=rt,
                    pressure_coeff=.85+.1*zi+rng.uniform(0,.12),
                    porosity_pct=3.5+.35*latent+rng.normal(0,.15),
                    formation_top_md=depth-40,formation_bottom_md=depth+60,max_burial_top=depth+500,
                    layer=f'Member_{j%2+1}',lithology='shale' if j%3 else 'silty_shale',
                    diagenesis_category=f'Stage_{zi+1}',
                    toc_pct=toc if toc_label else np.nan,total_gas_m3_t=gas if gas_label else np.nan))
    data=pd.DataFrame(records)
    # Preserve at least three available logs per labelled sample.
    for col in ['GR','AC']:
        data.loc[rng.choice(data.index,size=max(1,len(data)//20),replace=False),col]=np.nan
    data.loc[rng.choice(data.index,size=len(data)//10,replace=False),'porosity_pct']=np.nan
    return data
