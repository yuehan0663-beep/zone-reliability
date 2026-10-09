"""Draw current-numbered evidence companions from released tables or local logs.

These reproduce the numerical evidence, with a portable layout. Author artwork
and exact manuscript page layout are outside this renderer. Raw logs are needed
only for Fig. 3 and Fig. S2 and are never written as a public CSV by this script.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zonerel.io import read_dataset, target_datasets

ROOT=Path(__file__).resolve().parents[1]
COLORS={'toc_pct':'#0072B2','total_gas_m3_t':'#D55E00','TOC':'#0072B2','total_gas':'#D55E00'}
ZONES=['North','Central','South']

def table(name):
    return pd.read_csv(ROOT/'data_derived'/name)

def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'{name}.{ext}',dpi=300)
    plt.close(fig)

def draw(number,out,raw=None):
    """Render one evidence figure with current manuscript display numbering."""
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    if number=='03':
        if raw is None:raise ValueError('Fig. 3 requires an authorized local log dataset')
        d=raw.copy()
        rt0=d.RT.median(skipna=True);ac0=d.AC.median(skipna=True)
        d['delta_logR_vis']=np.log10(d.RT/rt0)+.02*(d.AC-ac0)
        d=d[d.toc_pct.notna()]
        fig,axes=plt.subplots(1,2,figsize=(7.48,3.35),layout='constrained',sharey=True)
        for ax,log,label in zip(axes,['DEN','delta_logR_vis'],['DEN (g/cm³)','Delta-logR-style index']):
            for zone,color,marker in zip(ZONES,['#0072B2','#E69F00','#009E73'],['o','s','^']):
                g=d[d.zone==zone];ax.scatter(g[log],g.toc_pct,s=9,alpha=.65,color=color,marker=marker,label=zone)
            ax.set(xlabel=label,ylabel='TOC (wt.%)')
        axes[0].legend(fontsize=7)
        name='fig03_log_toc_crossplots'
    elif number=='04':
        d=table('Table_S7_feature_drift_metrics.csv')
        fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained')
        features=list(d.feature.drop_duplicates());positions=np.arange(len(features))
        for zone,offset,color,marker in [('North',-.12,'#0072B2','o'),('South',.12,'#D55E00','s')]:
            g=d[(d.reference_zone=='Central')&(d.comparison_zone==zone)].set_index('feature').reindex(features)
            ax.scatter(g.PSI,positions+offset,color=color,marker=marker,label=f'{zone} vs Central')
        ax.set(yticks=positions,yticklabels=features,xlabel='Population stability index',title='Zonal feature drift')
        ax.legend(fontsize=8);name='fig04_zonal_drift'
    elif number=='05':
        d=table('Table_S1_full_model_rank.csv');residual=table('Table_S12_toc_lozo_residual_by_zone.csv')
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained')
        for target,model,offset,marker in [('toc_pct','ExtraTrees',-.08,'o'),('total_gas_m3_t','RF',.08,'s')]:
            g=d[(d.target==target)&(d.model==model)].set_index('cv_scheme').reindex(['random','LOWO','LOZO'])
            axes[0].scatter(np.arange(3)+offset,g.pooled_R2,color=COLORS[target],marker=marker,label=target)
        axes[0].set(xticks=range(3),xticklabels=['random','LOWO','LOZO'],ylabel='Pooled R²')
        axes[0].axhline(0,color='.6',lw=.7);axes[0].legend(fontsize=7)
        r=residual.set_index('zone_en').reindex(ZONES)
        axes[1].scatter(range(3),r.mean_residual,color='#0072B2')
        axes[1].axhline(0,color='.6',lw=.7)
        axes[1].set(xticks=range(3),xticklabels=ZONES,ylabel='Mean TOC LOZO residual (wt.%)')
        name='fig05_validation_residuals'
    elif number=='06':
        d=table('Table_S2_conformal_summary_by_protocol.csv');reject=table('Table_S3_rejection_curves.csv')
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained')
        for target,marker in [('toc_pct','o'),('total_gas_m3_t','s')]:
            g=d[(d.target==target)&(d.cv_scheme=='LOWO')]
            axes[0].scatter(g.mean_width,g.coverage,color=COLORS[target],marker=marker,label=target)
            g=reject[(reject.target==target)&(reject.cv_scheme=='LOWO')&(reject.method=='global')]
            g=g.groupby('reject_frac',as_index=False).MAE.mean().sort_values('reject_frac')
            axes[1].plot(g.reject_frac,g.MAE,color=COLORS[target],marker=marker,label=target)
        axes[0].set(xlabel='Mean interval width (target units)',ylabel='Coverage',ylim=(0,1))
        axes[1].set(xlabel='Rejection fraction',ylabel='Retained MAE (target units)')
        axes[0].legend(loc='lower right',fontsize=7);name='fig06_intervals_rejection'
    elif number=='07':
        d=table('zone_reliability_summary.csv')
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        for target,offset,marker in [('toc_pct',-.08,'o'),('total_gas_m3_t',.08,'s')]:
            g=d[d.target==target].set_index('zone').reindex(ZONES)
            for ax,col in zip(axes,['coverage','reject_rate']):
                ax.scatter(np.arange(3)+offset,g[col],color=COLORS[target],marker=marker,label=target)
        for ax,label in zip(axes,['Coverage','Rejection rate']):
            ax.set(xticks=range(3),xticklabels=ZONES,ylabel=label,ylim=(0,1))
        axes[0].legend(fontsize=7);name='fig07_zone_reliability'
    elif number=='08':
        d=table('Table_S4_recalibration_all_methods.csv');curve=table('Table_S8_learning_curve_guidance.csv')
        fig,axes=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
        for target,proxy,offset,marker in [('TOC','schmoker_toc_proxy',-.08,'o'),('total_gas','gas_total_proxy_quant',.08,'s')]:
            g=d[(d.target==target)&(d.proxy==proxy)&(d.zone=='North')]
            methods=list(g.method)
            axes[0].scatter(np.arange(len(g))+offset,pd.to_numeric(g.method_MAE),color=COLORS[target],marker=marker,label=target)
            c=curve[(curve.target==target)&(curve.proxy==proxy)&(curve.zone=='North')].sort_values('training_wells')
            axes[1].plot(c.training_wells,c.mae_mean,color=COLORS[target],marker=marker,label=target)
        axes[0].set(xticks=range(len(methods)),xticklabels=[m.split('_')[0] for m in methods],ylabel='North within-zone MAE')
        axes[1].set(xlabel='Sampled training wells',ylabel='Resampled-error guidance MAE')
        axes[0].legend(fontsize=7);name='fig08_recalibration_learning'
    elif number=='09':
        d=table('Table_S22a_perfold_importance.csv');d=d[d.feature_set=='full']
        matrix=d.groupby(['target','cv_scheme','feature']).importance.mean().unstack('feature')
        matrix=matrix.div(matrix.sum(axis=1),axis=0)
        fig,ax=plt.subplots(figsize=(8,3.7),layout='constrained')
        im=ax.imshow(matrix,aspect='auto',cmap='cividis',vmin=0)
        labels={'diagenesis_category':'Diagenesis','formation_bottom_md':'Formation bottom','formation_top_md':'Formation top',
                'max_burial_top':'Max burial','pressure_coeff':'Pressure coeff.'}
        ax.set(xticks=range(len(matrix.columns)),xticklabels=[labels.get(c,c) for c in matrix.columns],yticks=range(len(matrix)),
               yticklabels=[f'{target} {scheme}' for target,scheme in matrix.index])
        plt.setp(ax.get_xticklabels(),rotation=90,ha='center',fontsize=8)
        fig.colorbar(im,ax=ax,label='Normalized fold-mean absolute SHAP')
        name='fig09_shap_matrix'
    elif number=='10':
        d=table('Table_S23_residual_geology_correlations.csv')
        d=d[(d.level=='point')&(d.analysis=='Spearman')&(d.metric=='signed residual')]
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        ax.scatter(d.rho,range(len(d)),color='#0072B2')
        ax.axvline(0,color='.6',lw=.7)
        ax.set(yticks=range(len(d)),yticklabels=d.variable,xlabel='Point-level Spearman ρ',xlim=(-1,1),title='TOC LOZO residual and geological covariates')
        name='fig10_residual_geology'
    elif number=='S2':
        if raw is None:raise ValueError('Fig. S2 requires an authorized local log dataset')
        fig,axes=plt.subplots(3,7,figsize=(12,7),layout='constrained')
        requested=table('Table_S11_figure4_representative_wells.csv')
        for zi,zone in enumerate(ZONES):
            public=requested[requested.zone==zone].well_original.iloc[0]
            d=raw[(raw.zone==zone)&(raw.well==public)].sort_values('depth_md')
            if d.empty:raise ValueError('Representative well requires an approved identity mapping for '+zone)
            columns=['GR','AC','DEN','CNL','RT','toc_pct','total_gas_m3_t']
            units=['GR (API)','AC (µs/m)','DEN (g/cm³)','CNL (%)','RT (ohm m)','TOC (wt.%)','Gas (m³/t)']
            for ax,col,label in zip(axes[zi],columns,units):
                if col in ['toc_pct','total_gas_m3_t']:
                    ax.scatter(d[col],d.depth_md,color=COLORS[col],s=12,marker='o' if col=='toc_pct' else 's')
                else:ax.plot(d[col],d.depth_md,color='#0072B2',lw=.8)
                padding=.02*(d.depth_md.max()-d.depth_md.min())
                ax.set(xlabel=label,ylabel='Depth (m)',ylim=(d.depth_md.max()+padding,d.depth_md.min()-padding))
            axes[zi,0].set_title(f'{zone} ({public})')
        name='figS2_representative_tracks'
    else:raise ValueError(number)
    save(fig,Path(out),name)

def main(number=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/paper_figures'))
    p.add_argument('--input',type=Path)
    p.add_argument('--sheet',default=None)
    p.add_argument('--well-map',type=Path,help='Authorized local mapping CSV: original ID in first column, public ID in third')
    p.add_argument('--figure',choices=['03','04','05','06','07','08','09','10','S2'])
    a=p.parse_args();raw=read_dataset(a.input,a.sheet) if a.input else None
    if a.well_map:
        if raw is None:raise ValueError('--well-map requires --input')
        mapping=pd.read_csv(a.well_map,dtype=str)
        aliases=dict(zip(mapping.iloc[:,0],mapping.iloc[:,2]))
        mapped=raw.well.map(aliases)
        if mapped.isna().any():raise ValueError('The authorized mapping does not cover every input well')
        raw['well']=mapped
    figures=[number or a.figure] if number or a.figure else ['04','05','06','07','08','09','10']
    for n in figures:draw(n,a.output,raw)
    print('Rendered current-numbered evidence companion figures:',', '.join(figures))

if __name__=='__main__':main()
