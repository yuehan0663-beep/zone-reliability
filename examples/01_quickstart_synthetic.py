"""Run all three phases on generated data; pass --fast for five model families."""
from pathlib import Path
import argparse
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zonerel.synthetic import generate_synthetic
from zonerel.pipeline import run_workflow

def plot_results(results,out):
    """Show protocol diagnostics and rejection trajectories on synthetic data."""
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    order=['random','LOWO','LOZO']
    colors={'toc_pct':'#0072B2','total_gas_m3_t':'#D55E00'}
    markers={'toc_pct':'o','total_gas_m3_t':'s'}
    summary=results['validation_summary']
    for target,backbone in [('toc_pct','ExtraTrees'),('total_gas_m3_t','RF')]:
        g=summary[(summary.target==target)&(summary.model==backbone)].set_index('cv_scheme').reindex(order)
        x=np.arange(3)+(-.08 if target=='toc_pct' else .08)
        axes[0].scatter(x,g.pooled_R2,color=colors[target],marker=markers[target],label=target)
    axes[0].set(xticks=range(3),xticklabels=order,ylabel='Pooled R²',title='Validation by protocol')
    axes[0].axhline(0,color='.6',lw=.7)
    axes[0].legend(fontsize=7)
    reject=results['rejection_curve']
    # Column names retain the research implementation.
    pct='reject_frac'
    err=next(c for c in ['MAE_w','MAE','retained_MAE_w','retained_MAE'] if c in reject)
    for target in colors:
        g=reject[(reject.target==target)&(reject.cv_scheme=='LOWO')&(reject.method=='global')].groupby(pct,as_index=False)[err].mean().sort_values(pct)
        axes[1].plot(g[pct],g[err],color=colors[target],marker=markers[target],label=target)
    axes[1].set(xlabel='Rejection fraction',ylabel='Retained MAE',title='Risk rejection (LOWO)')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'synthetic_diagnostics.png',dpi=300)
    fig.savefig(out/'synthetic_diagnostics.pdf')
    fig.savefig(out/'synthetic_diagnostics.svg')
    plt.close(fig)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast',action='store_true',help='Use five validation models; retain attribution backbones')
    parser.add_argument('--output',type=Path,default=Path('outputs/synthetic'))
    args=parser.parse_args(argv)
    started=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    data=generate_synthetic()
    data.to_csv(args.output/'synthetic_dataset.csv',index=False)
    profile=data.groupby('zone').agg(rows=('sample_id','size'),wells=('well','nunique'))
    profile.to_csv(args.output/'synthetic_profile.csv')
    print(profile.to_string(),flush=True)
    warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMRegressor was fitted with feature names')
    results=run_workflow(data,{'fast':args.fast,'run_attribution':True,
                            'progress':lambda dataset,model:print(dataset,model,flush=True)})
    for name,table in results.items():
        table.to_csv(args.output/(name+'.csv'),index=False)
    plot_results(results,args.output)
    metadata={'mode':'fast' if args.fast else 'full','seconds':time.perf_counter()-started,
              'rows':len(data),'wells':data.well.nunique(),'seed':20261009,
              'models':int(results['validation_summary'].model.nunique()),
              'synthetic_only':True,'outputs':sorted(results)}
    (args.output/'runtime.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)
    return results

if __name__=='__main__':
    main()
