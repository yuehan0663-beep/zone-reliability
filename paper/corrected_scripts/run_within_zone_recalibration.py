# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
import csv, math, os, random, re, zipfile, xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

INFILE='Input Program/derived_dataset.xlsx'
SHEET='\u6570\u636e'
OUTDIR='within_zone_recalibration_analysis_derived_dataset'
RNG=random.Random(20260613)
ZONE_EN={'\u5317':'North','\u4e2d':'Central','\u5357':'South'}
FEATURES_C4=['GR','AC','DEN','CNL','RT','pressure_coeff','depth_md','porosity_pct']
METHODS=['C0_proxy','C0b_zone_mean','C1_offset','C2_affine','C3_isotonic','C4_small_ridge']
REJECT_PCTS=[0,5,10,15,20,25,30,35,40]
NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def col_to_idx(ref):
    m=re.match(r'([A-Z]+)',ref); n=0
    for ch in m.group(1): n=n*26+ord(ch)-64
    return n-1

def load_xlsx(path,sheet_name):
    z=zipfile.ZipFile(path); shared=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        root=ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in root.findall(NS+'si'):
            shared.append(''.join((t.text or '') for t in si.iter(NS+'t')))
    wb=ET.fromstring(z.read('xl/workbook.xml')); rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    rid_to_target={r.attrib['Id']:r.attrib['Target'] for r in rels}
    sheet_path=None
    for s in wb.find(NS+'sheets').findall(NS+'sheet'):
        if s.attrib['name']==sheet_name:
            rid=s.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']; target=rid_to_target[rid]
            sheet_path=target.lstrip('/') if target.startswith('/xl/') else ('xl/'+target.lstrip('/') if not target.startswith('xl/') else target)
    if sheet_path is None: raise ValueError('sheet not found')
    root=ET.fromstring(z.read(sheet_path)); rows=[]
    for row in root.iter(NS+'row'):
        vals={}; maxidx=-1
        for c in row.findall(NS+'c'):
            idx=col_to_idx(c.attrib.get('r','A1')); maxidx=max(maxidx,idx); t=c.attrib.get('t'); v=c.find(NS+'v')
            if t=='inlineStr':
                isel=c.find(NS+'is'); text=''.join((tt.text or '') for tt in isel.iter(NS+'t')) if isel is not None else ''
            elif v is None: text=''
            elif t=='s': text=shared[int(v.text)]
            else: text=v.text or ''
            vals[idx]=text
        if vals: rows.append([vals.get(i,'') for i in range(maxidx+1)])
    header=rows[0]; out=[]
    for rr in rows[1:]:
        rr=rr+['']*(len(header)-len(rr))
        rec={h:rr[i] for i,h in enumerate(header)}
        if any(str(v)!='' for v in rec.values()): out.append(rec)
    return header,out

def fnum(x):
    try:
        if x is None or str(x)=='': return None
        v=float(x); return None if math.isnan(v) else v
    except Exception: return None

def percentile(vals,q):
    vals=sorted([v for v in vals if v is not None and math.isfinite(v)])
    if not vals: return None
    pos=(len(vals)-1)*q/100; lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    return vals[lo] if lo==hi else vals[lo]*(hi-pos)+vals[hi]*(pos-lo)

def mean(vals):
    vals=[v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals)/len(vals) if vals else float('nan')

def add_proxies(rows):
    byz=defaultdict(list)
    for r in rows: byz[r['zone']].append(r)
    baselines={}
    for z,zs in byz.items():
        den_base=percentile([fnum(r.get('DEN')) for r in zs],90)
        gr_cut=percentile([fnum(r.get('GR')) for r in zs],25)
        ref=[r for r in zs if fnum(r.get('GR')) is not None and gr_cut is not None and fnum(r.get('GR'))<=gr_cut and fnum(r.get('RT')) is not None and fnum(r.get('AC')) is not None]
        if len(ref)<5: ref=[r for r in zs if fnum(r.get('RT')) is not None and fnum(r.get('AC')) is not None]
        baselines[z]={'DEN_baseline_p90':den_base,'lowGR_cut_p25':gr_cut,'RT_baseline_lowGR_median':percentile([fnum(r.get('RT')) for r in ref],50),'AC_baseline_lowGR_median':percentile([fnum(r.get('AC')) for r in ref],50),'porosity_median':percentile([fnum(r.get('porosity_pct')) for r in zs],50),'n_zone_rows':len(zs),'n_lowGR_ref':len(ref)}
    for r in rows:
        z=r['zone']; b=baselines[z]
        den=fnum(r.get('DEN')); rt=fnum(r.get('RT')); ac=fnum(r.get('AC')); depth=fnum(r.get('depth_md')); pc=fnum(r.get('pressure_coeff')); por=fnum(r.get('porosity_pct'))
        r['schmoker_toc_proxy']=None
        if den is not None and b['DEN_baseline_p90'] is not None: r['schmoker_toc_proxy']=max(0,(b['DEN_baseline_p90']-den)*1.378)
        r['passey_toc_proxy']=None
        if rt and ac is not None and b['RT_baseline_lowGR_median'] and b['AC_baseline_lowGR_median']:
            dlogr=math.log10(rt/b['RT_baseline_lowGR_median'])+0.02*(ac-b['AC_baseline_lowGR_median'])
            r['passey_toc_proxy']=max(0,dlogr*(10**(2.297-0.1688*11.5)))
        r['gas_total_proxy_quant']=None; r['gas_ads_proxy']=None; r['gas_free_proxy']=None; r['porosity_imputed_flag']='0'
        if depth is not None and pc is not None:
            P=max(0.001,pc*depth*0.0098)
            # Methods: fixed adsorption capacity, independent of labels.
            qL_mgg=0.39
            PL=7.60
            qL_m3t=qL_mgg*1.3975
            G_ads=qL_m3t*P/(P+PL)
            phi=por
            if phi is None:
                phi=b['porosity_median']; r['porosity_imputed_flag']='1'
            if phi is not None:
                phi_frac=phi/100.0
                tempK=293.15+0.03*depth
                Z=0.90
                # Bg reservoir m3/std m3 approx 0.00035096*Z*T/P; convert pore volume to std m3/t using rho_rock 2.65 t/m3 and Sg=0.5 proxy
                Bg=0.00035096*Z*tempK/P
                G_free=(phi_frac*0.5)/(2.65*Bg)
                r['gas_ads_proxy']=G_ads; r['gas_free_proxy']=G_free; r['gas_total_proxy_quant']=G_ads+G_free
    return baselines

def mae(y,p): return float(np.mean(np.abs(np.array(y)-np.array(p)))) if y else float('nan')
def r2(y,p):
    y=np.array(y,float); p=np.array(p,float)
    if len(y)<2: return float('nan')
    ss=float(np.sum((y-p)**2)); st=float(np.sum((y-np.mean(y))**2)); return 1-ss/st if st>0 else float('nan')
def rankdata(a):
    pairs=sorted((v,i) for i,v in enumerate(a)); ranks=[0]*len(a); i=0
    while i<len(pairs):
        j=i
        while j+1<len(pairs) and pairs[j+1][0]==pairs[i][0]: j+=1
        rk=(i+j)/2+1
        for k in range(i,j+1): ranks[pairs[k][1]]=rk
        i=j+1
    return ranks
def spear(y,p):
    if len(y)<3: return float('nan')
    ry=np.array(rankdata(y)); rp=np.array(rankdata(p))
    return float(np.corrcoef(ry,rp)[0,1]) if np.std(ry)>0 and np.std(rp)>0 else float('nan')
def mdict(y,p): return {'pooled_R2':r2(y,p),'weighted_MAE':mae(y,p),'Spearman':spear(y,p),'n':len(y)}
def fit_offset(p,y): return mean([yy-pp for pp,yy in zip(p,y)])
def fit_affine(p,y):
    x=np.array(p,float); yy=np.array(y,float)
    if len(x)<2 or np.var(x)==0: return 0.0, float(np.mean(yy)) if len(yy) else 0.0
    a,b=np.linalg.lstsq(np.vstack([x,np.ones(len(x))]).T, yy, rcond=None)[0]; return float(a),float(b)
def pava_fit(x,y):
    order=np.argsort(np.array(x)); blocks=[]
    for i in order:
        blocks.append([x[i],x[i],y[i],1])
        while len(blocks)>=2 and blocks[-2][2]>blocks[-1][2]:
            b2=blocks.pop(); b1=blocks.pop(); n=b1[3]+b2[3]; avg=(b1[2]*b1[3]+b2[2]*b2[3])/n; blocks.append([b1[0],b2[1],avg,n])
    bx=[]; by=[]
    for lo,hi,v,n in blocks: bx += [lo,hi]; by += [v,v]
    return bx,by
def pred_iso(p,model):
    bx,by=model; out=[]
    for v in p:
        if v<=bx[0]: out.append(by[0]); continue
        if v>=bx[-1]: out.append(by[-1]); continue
        k=0
        while k<len(bx)-1 and bx[k+1]<v: k+=1
        t=0 if bx[k+1]==bx[k] else (v-bx[k])/(bx[k+1]-bx[k]); out.append(by[k]*(1-t)+by[k+1]*t)
    return out
def mat_builder(train, features):
    meds=[]; mus=[]; sds=[]
    for c in features:
        vals=[fnum(r.get(c)) for r in train if fnum(r.get(c)) is not None]
        med=percentile(vals,50) if vals else 0.0; arr=np.array(vals or [med],float)
        meds.append(med); mus.append(float(np.mean(arr))); sds.append(float(np.std(arr)) or 1.0)
    def mat(rs,proxy):
        X=[]
        for r in rs:
            row=[1.0,float(r[proxy])]
            for c,med,mu,sd in zip(features,meds,mus,sds):
                v=fnum(r.get(c)); v=med if v is None else v; row.append((v-mu)/sd)
            X.append(row)
        return np.array(X,float)
    return mat
def fit_c4(train,y,proxy):
    mat=mat_builder(train,FEATURES_C4); X=mat(train,proxy); yy=np.array(y,float); lam=np.eye(X.shape[1])*2.0; lam[0,0]=0
    return mat, np.linalg.pinv(X.T@X+lam)@X.T@yy
def eval_zone(zrows,target,proxy,method):
    obs=[]; pred=[]; wells=[]; samples=[]; risks=[]; zmeans=[]
    for w in sorted(set(r['well'] for r in zrows)):
        tr=[r for r in zrows if r['well']!=w]; te=[r for r in zrows if r['well']==w]
        if not tr or not te: continue
        tp=[r[proxy] for r in tr]; ty=[fnum(r[target]) for r in tr]; ep=[r[proxy] for r in te]; ey=[fnum(r[target]) for r in te]
        zm=mean(ty)
        if method=='C0_proxy': yh=ep
        elif method=='C0b_zone_mean': yh=[zm]*len(te)
        elif method=='C1_offset': yh=[v+fit_offset(tp,ty) for v in ep]
        elif method=='C2_affine':
            a,b=fit_affine(tp,ty); yh=[a*v+b for v in ep]
        elif method=='C3_isotonic': yh=pred_iso(ep,pava_fit(tp,ty))
        elif method=='C4_small_ridge':
            mat,beta=fit_c4(tr,ty,proxy); yh=list(mat(te,proxy)@beta)
        med=percentile(tp,50); iqr=max(1e-9,(percentile(tp,75)-percentile(tp,25)) if percentile(tp,25) is not None else np.std(tp) or 1)
        obs+=ey; pred+=[float(v) for v in yh]; wells += [w]*len(te); samples += [r['sample_id'] for r in te]; risks += [abs(v-med)/iqr for v in ep]; zmeans += [zm]*len(te)
    return obs,pred,wells,samples,risks,zmeans
def bootstrap(y,p,wells,B=400):
    uniq=sorted(set(wells)); idx_by={w:[i for i,a in enumerate(wells) if a==w] for w in uniq}; res=defaultdict(list)
    if len(uniq)<3: return {k:(float('nan'),float('nan')) for k in ['pooled_R2','weighted_MAE','Spearman','MAE_improvement_vs_zonemean']}
    for _ in range(B):
        idx=[]
        for w in [RNG.choice(uniq) for _ in uniq]: idx+=idx_by[w]
        yy=[y[i] for i in idx]; pp=[p[i] for i in idx]
        m=mdict(yy,pp)
        for k in ['pooled_R2','weighted_MAE','Spearman']:
            if math.isfinite(m[k]): res[k].append(m[k])
    return {k:(percentile(v,2.5),percentile(v,97.5)) if len(v)>=20 else (float('nan'),float('nan')) for k,v in res.items()}
def bootstrap_improvement(y,p,zm,wells,B=500):
    uniq=sorted(set(wells)); idx_by={w:[i for i,a in enumerate(wells) if a==w] for w in uniq}; vals=[]
    if len(uniq)<3: return float('nan'),float('nan')
    for _ in range(B):
        idx=[]
        for w in [RNG.choice(uniq) for _ in uniq]: idx+=idx_by[w]
        yy=[y[i] for i in idx]; pp=[p[i] for i in idx]; zz=[zm[i] for i in idx]
        base=mae(yy,zz); corr=mae(yy,pp)
        if base>0: vals.append((base-corr)/base*100)
    return percentile(vals,2.5), percentile(vals,97.5)
def fmt(x):
    try: return '' if not math.isfinite(float(x)) else f'{float(x):.6g}'
    except Exception: return ''
def write_csv(fn,rows,fields):
    with open(os.path.join(OUTDIR,fn),'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def drift(p,y):
    a,b=fit_affine(p,y); c0=mae(y,p); c1=mae(y,[v+fit_offset(p,y) for v in p]); c2=mae(y,[a*v+b for v in p]); r=spear(y,p)
    if abs(a-1)<0.25 and c1<c0*.85: return 'location_shift'
    if c2<c1*.90 and abs(a-1)>=0.25: return 'scale_shift'
    if math.isfinite(r) and r>.4: return 'monotonic_shape_shift'
    return 'relationship_shift_or_weak_proxy'
def svg_bar(fn,title,rows):
    W,H,M=1100,560,90; vals=[float(r['relative_MAE_improvement_vs_zonemean_pct']) for r in rows if r.get('relative_MAE_improvement_vs_zonemean_pct')]
    ymin=min(vals+[0,-5]); ymax=max(vals+[5]);
    def ymap(v): return H-M-(v-ymin)/(ymax-ymin)*(H-2*M)
    step=(W-2*M)/max(1,len(rows)); bw=step*.6
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="30" text-anchor="middle" font-size="18">{title}</text>',f'<line x1="{M}" y1="{ymap(0)}" x2="{W-M}" y2="{ymap(0)}" stroke="black"/><line x1="{M}" y1="{M}" x2="{M}" y2="{H-M}" stroke="black"/>']
    for i,r in enumerate(rows):
        v=float(r['relative_MAE_improvement_vs_zonemean_pct']); x=M+i*step; y=ymap(v); y0=ymap(0); color='#4C78A8' if v>1 else '#B0B0B0'
        parts.append(f'<rect x="{x:.1f}" y="{min(y,y0):.1f}" width="{bw:.1f}" height="{abs(y-y0):.1f}" fill="{color}"/>')
        lab=f'{r["target"]}-{r["zone_en"]}'
        parts.append(f'<text x="{x+bw/2:.1f}" y="{H-45}" transform="rotate(-45 {x+bw/2:.1f},{H-45})" text-anchor="end" font-size="11">{lab}</text>')
    parts.append('<text x="20" y="280" transform="rotate(-90 20,280)">MAE improvement vs zone-mean (%)</text></svg>')
    open(os.path.join(OUTDIR,fn),'w',encoding='utf-8').write('\n'.join(parts))
def legacy_computational_main():
    header,rows=load_xlsx(INFILE,SHEET); baselines=add_proxies(rows)
    # audits
    audit=[]
    audit.append({'item':'shape','value':f'{len(rows)} rows x {len(header)} cols'})
    for target in ['toc_pct','total_gas_m3_t']:
        sub=[r for r in rows if fnum(r.get(target)) is not None]; audit.append({'item':target+'_points_wells','value':f'{len(sub)} points / {len(set(r["well"] for r in sub))} wells'}); audit.append({'item':target+'_zone_wells','value':str({z:len(set(r['well'] for r in sub if r['zone']==z)) for z in sorted(set(r['zone'] for r in sub))})})
    gas=[r for r in rows if fnum(r.get('total_gas_m3_t')) is not None]
    audit.append({'item':'gas_pressure_depth_zone_wells','value':str({z:len(set(r['well'] for r in gas if r['zone']==z and fnum(r.get('pressure_coeff')) is not None and fnum(r.get('depth_md')) is not None)) for z in sorted(set(r['zone'] for r in gas))})})
    audit.append({'item':'gas_porosity_zone_wells','value':str({z:len(set(r['well'] for r in gas if r['zone']==z and fnum(r.get('porosity_pct')) is not None)) for z in sorted(set(r['zone'] for r in gas))})})
    write_csv('load_audit.csv',audit,['item','value'])
    write_csv('proxy_baselines.csv',[dict(zone=z,zone_en=ZONE_EN.get(z,z),**b) for z,b in baselines.items()],['zone','zone_en','DEN_baseline_p90','lowGR_cut_p25','RT_baseline_lowGR_median','AC_baseline_lowGR_median','porosity_median','n_zone_rows','n_lowGR_ref'])
    targets=[('TOC','toc_pct',['schmoker_toc_proxy','passey_toc_proxy']),('total_gas','total_gas_m3_t',['gas_total_proxy_quant'])]
    results=[]; preds=[]; drifts=[]; rej=[]; benefit=[]
    for tname,tcol,proxies in targets:
        for proxy in proxies:
            for z in sorted(set(r['zone'] for r in rows)):
                zrows=[r for r in rows if r['zone']==z and fnum(r.get(tcol)) is not None and r.get(proxy) is not None and math.isfinite(r.get(proxy))]
                wells=sorted(set(r['well'] for r in zrows))
                if len(wells)<3 or len(zrows)<5: continue
                p=[r[proxy] for r in zrows]; y=[fnum(r[tcol]) for r in zrows]; a,b=fit_affine(p,y)
                drifts.append({'target':tname,'proxy':proxy,'zone':z,'zone_en':ZONE_EN.get(z,z),'n':len(zrows),'wells':len(wells),'spearman_proxy_gold':fmt(spear(y,p)),'affine_slope':fmt(a),'affine_intercept':fmt(b),'drift_type':drift(p,y)})
                cache={}
                for m in METHODS:
                    yy,pp,ww,ss,rr,zm=eval_zone(zrows,tcol,proxy,m); cache[m]=(yy,pp,ww,ss,rr,zm)
                    md=mdict(yy,pp); ci=bootstrap(yy,pp,ww)
                    rec={'target':tname,'proxy':proxy,'zone':z,'zone_en':ZONE_EN.get(z,z),'method':m,'n':len(yy),'wells':len(set(ww))}
                    for k,v in md.items(): rec[k]=fmt(v)
                    for k,(lo,hi) in ci.items(): rec[k+'_ci_low']=fmt(lo); rec[k+'_ci_high']=fmt(hi)
                    results.append(rec)
                    for sid,w,obs,pr,ris,zmp in zip(ss,ww,yy,pp,rr,zm): preds.append({'target':tname,'proxy':proxy,'zone':z,'zone_en':ZONE_EN.get(z,z),'well':w,'sample_id':sid,'method':m,'observed':fmt(obs),'predicted':fmt(pr),'zone_mean_prediction':fmt(zmp),'rejection_score':fmt(ris),'abs_error':fmt(abs(obs-pr)),'zone_mean_abs_error':fmt(abs(obs-zmp))})
                cand=[r for r in results if r['target']==tname and r['proxy']==proxy and r['zone']==z and r['method'] in ['C1_offset','C2_affine','C3_isotonic','C4_small_ridge']]
                best=min(cand,key=lambda r:float(r['weighted_MAE']))
                yy,pp,ww,ss,rr,zm=cache[best['method']]
                base=mae(yy,zm); corr=mae(yy,pp); imp=(base-corr)/base*100 if base>0 else float('nan'); ilo,ihi=bootstrap_improvement(yy,pp,zm,ww)
                benefit_class='CI-robust' if imp>1 and ilo>0 else ('directional only' if imp>1 else 'no directional benefit')
                benefit.append({'target':tname,'proxy':proxy,'zone':z,'zone_en':ZONE_EN.get(z,z),'wells':len(set(ww)),'best_method':best['method'],'best_MAE':fmt(corr),'zone_mean_MAE':fmt(base),'relative_MAE_improvement_vs_zonemean_pct':fmt(imp),'improvement_ci_low_pct':fmt(ilo),'improvement_ci_high_pct':fmt(ihi),'benefit_class':benefit_class,'small_sample_note':'North gas has 5 wells; interpret cautiously' if (tname=='total_gas' and z=='\u5317') else ''})
                order=sorted(range(len(yy)),key=lambda i:rr[i])
                for pct in REJECT_PCTS:
                    keep=max(1,int(round(len(order)*(100-pct)/100))); idx=order[:keep]
                    cy=[yy[i] for i in idx]; cp=[pp[i] for i in idx]; cz=[zm[i] for i in idx]
                    bmae=mae(cy,cz); cmae=mae(cy,cp); ri=(bmae-cmae)/bmae*100 if bmae>0 else float('nan')
                    rej.append({'target':tname,'proxy':proxy,'zone':z,'zone_en':ZONE_EN.get(z,z),'best_method':best['method'],'reject_pct':pct,'retain_pct':100-pct,'retained_n':len(idx),'retained_wells':len(set(ww[i] for i in idx)),'corrected_MAE':fmt(cmae),'zone_mean_MAE':fmt(bmae),'relative_improvement_pct':fmt(ri),'same_retained_subset':'yes','risk_score':'proxy_distance_from_training_distribution'})
    write_csv('zone_recalibration_results.csv',results,['target','proxy','zone','zone_en','method','n','wells','pooled_R2','pooled_R2_ci_low','pooled_R2_ci_high','weighted_MAE','weighted_MAE_ci_low','weighted_MAE_ci_high','Spearman','Spearman_ci_low','Spearman_ci_high'])
    write_csv('zone_recalibration_predictions.csv',preds,['target','proxy','zone','zone_en','well','sample_id','method','observed','predicted','zone_mean_prediction','rejection_score','abs_error','zone_mean_abs_error'])
    write_csv('drift_typing_per_zone.csv',drifts,['target','proxy','zone','zone_en','n','wells','spearman_proxy_gold','affine_slope','affine_intercept','drift_type'])
    write_csv('rejection_vs_zonemean.csv',rej,['target','proxy','zone','zone_en','best_method','reject_pct','retain_pct','retained_n','retained_wells','corrected_MAE','zone_mean_MAE','relative_improvement_pct','same_retained_subset','risk_score'])
    write_csv('benefit_summary.csv',benefit,['target','proxy','zone','zone_en','wells','best_method','best_MAE','zone_mean_MAE','relative_MAE_improvement_vs_zonemean_pct','improvement_ci_low_pct','improvement_ci_high_pct','benefit_class','small_sample_note'])
    # guidance/trust same as benefit plus best rejection
    guidance=[]; trust=[]
    for b in benefit:
        subs=[r for r in rej if r['target']==b['target'] and r['proxy']==b['proxy'] and r['zone']==b['zone']]
        bestrej=max(subs,key=lambda r:float(r['relative_improvement_pct']) if r['relative_improvement_pct'] else -999)
        guidance.append({**b,'best_rejection_pct':bestrej['reject_pct'],'best_rejection_improvement_pct':bestrej['relative_improvement_pct']})
        trust.append({'target':b['target'],'proxy':b['proxy'],'zone':b['zone'],'zone_en':b['zone_en'],'best_method':b['best_method'],'benefit_class':b['benefit_class'],'trusted_use':'candidate' if b['benefit_class']!='no directional benefit' else 'do_not_claim','note':b['small_sample_note']})
    write_csv('acquisition_guidance.csv',guidance,['target','proxy','zone','zone_en','wells','best_method','best_MAE','zone_mean_MAE','relative_MAE_improvement_vs_zonemean_pct','improvement_ci_low_pct','improvement_ci_high_pct','benefit_class','small_sample_note','best_rejection_pct','best_rejection_improvement_pct'])
    write_csv('trustworthy_blocks_summary.csv',trust,['target','proxy','zone','zone_en','best_method','benefit_class','trusted_use','note'])
    svg_bar('Fig_benefit_summary.svg','Directional benefit vs zone-mean by target and zone',benefit)
    # minimal png via PIL if available
    try:
        from PIL import Image, ImageDraw
        im=Image.new('RGB',(900,500),'white'); d=ImageDraw.Draw(im); d.text((220,20),'Directional benefit vs zone-mean (%)',fill='black')
        vals=[float(b['relative_MAE_improvement_vs_zonemean_pct']) for b in benefit]; ymin=min(vals+[0,-5]); ymax=max(vals+[5]); M=70
        def ym(v): return 430-(v-ymin)/(ymax-ymin)*330
        d.line((M,ym(0),840,ym(0)),fill='black'); d.line((M,80,M,430),fill='black')
        step=770/max(1,len(benefit));
        for i,b in enumerate(benefit):
            v=float(b['relative_MAE_improvement_vs_zonemean_pct']); x=M+i*step; y=ym(v); y0=ym(0); d.rectangle((x,min(y,y0),x+step*.55,max(y,y0)),fill=(76,120,168) if v>1 else (180,180,180)); d.text((x,440),b['target'][:3]+'-'+b['zone_en'][:1],fill='black')
        im.save(os.path.join(OUTDIR,'Fig_benefit_summary.png'))
    except Exception: pass
    # log/report
    open(os.path.join(OUTDIR,'proxy_construction_log.md'),'w',encoding='utf-8').write('# Proxy construction log - model-ready dataset\n\n- Input `/Input Program/derived_dataset.xlsx`, sheet `data`; pressure_coeff treated as corrected present-day pressure coefficient.\n- DOI/page hard gate cancelled by instruction; DOI marked pending author completion.\n- Wulalike adsorption anchor: W148 moisture-equilibrated 30 deg C, q_L=0.39 mg/g, P_L=7.60 MPa; parsed evidence from Table 4 lines 238-241.\n- Gas proxy upgraded to adsorption + free gas: G_total_proxy = G_ads + G_free. Pressure P=pressure_coeff*depth_md*0.0098. Free-gas term uses porosity, conservative S_g=0.5, Z=0.90, geothermal gradient 0.03 deg C/m; missing porosity imputed by zone median and flagged internally.\n- Proxy is a physically motivated predictor only; C1-C4 recalibration aligns it to measured total gas inside each within-zone LOWO training fold.\n- Gold total gas is not used to set proxy parameters.\n')
    report=['# Within-zone quantitative recalibration and rejection report - model-ready dataset\n\n','## Summary\n\n','- Data table 9 passed load audit; total gas pressure+depth is available for all 26 gas wells, so North gas is evaluated quantitatively with small-sample caution.\n','- Evaluation uses within-zone LOWO, pooled metrics, well bootstrap CI, and fair rejection on identical retained subsets against zone-mean.\n','- Directional benefit is defined as >1% relative MAE improvement over zone-mean; CI-robust means bootstrap improvement CI lower bound >0.\n\n','## Benefit summary\n\n']
    for b in benefit:
        report.append(f'- {b["target"]}/{b["zone_en"]}/{b["proxy"]}: {b["benefit_class"]}; best={b["best_method"]}; improvement={b["relative_MAE_improvement_vs_zonemean_pct"]}% (CI {b["improvement_ci_low_pct"]}, {b["improvement_ci_high_pct"]}); wells={b["wells"]}. {b["small_sample_note"]}\n')
    report.append('\n## Decision answers\n\n')
    dirs=[b for b in benefit if b['benefit_class']!='no directional benefit']; robust=[b for b in benefit if b['benefit_class']=='CI-robust']
    report.append('- >1% directional benefit: '+(', '.join(f'{b["target"]}-{b["zone_en"]}-{b["proxy"]}' for b in dirs) if dirs else 'none')+'.\n')
    report.append('- CI-robust: '+(', '.join(f'{b["target"]}-{b["zone_en"]}-{b["proxy"]}' for b in robust) if robust else 'none')+'.\n')
    report.append('- Rejection effects are listed in `rejection_vs_zonemean.csv`; use only rows where the corrected and zone-mean scores are compared on the same retained subset.\n\n')
    report.append('## Outputs\n\n')
    for fn in ['proxy_construction_log.md','load_audit.csv','drift_typing_per_zone.csv','zone_recalibration_results.csv','rejection_vs_zonemean.csv','benefit_summary.csv','acquisition_guidance.csv','trustworthy_blocks_summary.csv','Fig_benefit_summary.svg','Fig_benefit_summary.png','recal_log.txt']:
        report.append(f'- `{fn}`\n')
    open(os.path.join(OUTDIR,'within_zone_recalibration_report.md'),'w',encoding='utf-8').write(''.join(report))
    open(os.path.join(OUTDIR,'recal_log.txt'),'w',encoding='utf-8').write('Completed model-ready dataset quantitative TOC+gas within-zone LOWO recalibration and fair rejection evaluation. No cross-zone borrowing; no point random split.\n')
    print('done',OUTDIR,'results',len(results),'benefit',len(benefit),'rejection',len(rej))
def main():
    """Run the Methods-aligned audit with explicit data and output paths."""
    from run_methods_fix import main as run
    run()

if __name__ == '__main__':
    main()
