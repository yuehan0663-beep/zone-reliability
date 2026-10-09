# Gas proxy implementation changes

Historical computational statements are retained in `*_v_prev.py`; their
comments and explanatory strings were translated to English under v2. Byte-original
Online Resource sources remain unchanged outside this repository. The diff below
records the earlier Methods correction, before the commentary-only translation.
The main entry points delegate to a parameterized runner; it executes the
original computational statements in their original order, without plotting.

## run_within_zone_recalibration.py

```diff
--- run_within_zone_recalibration.py (previous)

+++ run_within_zone_recalibration.py (Methods)

@@ -6,7 +6,6 @@

 INFILE='Input Program/derived_dataset.xlsx'
 SHEET='\u6570\u636e'
 OUTDIR='within_zone_recalibration_analysis_derived_dataset'
-os.makedirs(OUTDIR, exist_ok=True)
 RNG=random.Random(20260613)
 ZONE_EN={'\u5317':'North','\u4e2d':'Central','\u5357':'South'}
 FEATURES_C4=['GR','AC','DEN','CNL','RT','pressure_coeff','depth_md','porosity_pct']
@@ -72,7 +71,6 @@

     byz=defaultdict(list)
     for r in rows: byz[r['zone']].append(r)
     baselines={}
-    # global TOC->qL anchor from Ordos rows in Li Table 1: qL/VL approx linear around qL=0.70*TOC+1.16, clipped to Wulalike moisture anchor scale.
     for z,zs in byz.items():
         den_base=percentile([fnum(r.get('DEN')) for r in zs],90)
         gr_cut=percentile([fnum(r.get('GR')) for r in zs],25)
@@ -81,7 +79,7 @@

         baselines[z]={'DEN_baseline_p90':den_base,'lowGR_cut_p25':gr_cut,'RT_baseline_lowGR_median':percentile([fnum(r.get('RT')) for r in ref],50),'AC_baseline_lowGR_median':percentile([fnum(r.get('AC')) for r in ref],50),'porosity_median':percentile([fnum(r.get('porosity_pct')) for r in zs],50),'n_zone_rows':len(zs),'n_lowGR_ref':len(ref)}
     for r in rows:
         z=r['zone']; b=baselines[z]
-        den=fnum(r.get('DEN')); rt=fnum(r.get('RT')); ac=fnum(r.get('AC')); depth=fnum(r.get('depth_md')); pc=fnum(r.get('pressure_coeff')); por=fnum(r.get('porosity_pct')); toc=fnum(r.get('toc_pct'))
+        den=fnum(r.get('DEN')); rt=fnum(r.get('RT')); ac=fnum(r.get('AC')); depth=fnum(r.get('depth_md')); pc=fnum(r.get('pressure_coeff')); por=fnum(r.get('porosity_pct'))
         r['schmoker_toc_proxy']=None
         if den is not None and b['DEN_baseline_p90'] is not None: r['schmoker_toc_proxy']=max(0,(b['DEN_baseline_p90']-den)*1.378)
         r['passey_toc_proxy']=None
@@ -91,10 +89,8 @@

         r['gas_total_proxy_quant']=None; r['gas_ads_proxy']=None; r['gas_free_proxy']=None; r['porosity_imputed_flag']='0'
         if depth is not None and pc is not None:
             P=max(0.001,pc*depth*0.0098)
-            # qL from W148 moisture-equilibrated 30C anchor plus TOC dependence; use measured TOC only as proxy construction auxiliary, not target gas.
-            toc_for_q = toc if toc is not None else r['schmoker_toc_proxy']
-            if toc_for_q is None: toc_for_q=2.0
-            qL_mgg=max(0.12, min(0.80, 0.39*(toc_for_q/2.0)))  # conservative Wulalike moisture-scale mg/g
+            # Methods: fixed adsorption capacity, independent of labels.
+            qL_mgg=0.39
             PL=7.60
             qL_m3t=qL_mgg*1.3975
             G_ads=qL_m3t*P/(P+PL)
@@ -102,14 +98,12 @@

             if phi is None:
                 phi=b['porosity_median']; r['porosity_imputed_flag']='1'
             if phi is not None:
-                phi_frac=max(0.0001, phi/100.0)
+                phi_frac=phi/100.0
                 tempK=293.15+0.03*depth
                 Z=0.90
                 # Bg reservoir m3/std m3 approx 0.00035096*Z*T/P; convert pore volume to std m3/t using rho_rock 2.65 t/m3 and Sg=0.5 proxy
                 Bg=0.00035096*Z*tempK/P
                 G_free=(phi_frac*0.5)/(2.65*Bg)
-                # keep physically rough free gas from dominating order by conservative scaling; C1-C4 re-calibrates to measured total gas.
-                G_free=G_free*0.01
                 r['gas_ads_proxy']=G_ads; r['gas_free_proxy']=G_free; r['gas_total_proxy_quant']=G_ads+G_free
     return baselines
 
@@ -237,7 +231,7 @@

         parts.append(f'<text x="{x+bw/2:.1f}" y="{H-45}" transform="rotate(-45 {x+bw/2:.1f},{H-45})" text-anchor="end" font-size="11">{lab}</text>')
     parts.append('<text x="20" y="280" transform="rotate(-90 20,280)">MAE improvement vs zone-mean (%)</text></svg>')
     open(os.path.join(OUTDIR,fn),'w',encoding='utf-8').write('\n'.join(parts))
-def main():
+def legacy_computational_main():
     header,rows=load_xlsx(INFILE,SHEET); baselines=add_proxies(rows)
     # audits
     audit=[]
@@ -323,4 +317,10 @@

     open(os.path.join(OUTDIR,'within_zone_recalibration_report.md'),'w',encoding='utf-8').write(''.join(report))
     open(os.path.join(OUTDIR,'recal_log.txt'),'w',encoding='utf-8').write('Completed model-ready dataset quantitative TOC+gas within-zone LOWO recalibration and fair rejection evaluation. No cross-zone borrowing; no point random split.\n')
     print('done',OUTDIR,'results',len(results),'benefit',len(benefit),'rejection',len(rej))
-if __name__=='__main__': main()
+def main():
+    """Run the Methods-aligned audit with explicit data and output paths."""
+    from run_methods_fix import main as run
+    run()
+
+if __name__ == '__main__':
+    main()
```

## run_gas_proxy_reproduction.py

```diff
--- run_gas_proxy_reproduction.py (previous)

+++ run_gas_proxy_reproduction.py (Methods)

@@ -30,7 +30,6 @@

     "Z": 0.90,
     "geothermal_gradient_C_per_m": 0.03,
     "rho_rock_t_m3": 2.65,
-    "free_gas_conservative_scale": 0.01,
 }
 
 GATE_EXPECTED = [
@@ -249,7 +248,6 @@

         depth = fnum(r.get("depth_md"))
         pc = fnum(r.get("pressure_coeff"))
         por = fnum(r.get("porosity_pct"))
-        toc = fnum(r.get("toc_pct"))
         r["schmoker_toc_proxy"] = None
         if den is not None and b["DEN_baseline_p90"] is not None:
             r["schmoker_toc_proxy"] = max(0, (b["DEN_baseline_p90"] - den) * 1.378)
@@ -263,10 +261,8 @@

         r["porosity_imputed_flag"] = "0"
         if depth is not None and pc is not None:
             pressure_mpa = max(0.001, pc * depth * 0.0098)
-            toc_for_q = toc if toc is not None else r["schmoker_toc_proxy"]
-            if toc_for_q is None:
-                toc_for_q = 2.0
-            ql_mgg = max(0.12, min(0.80, params["q_L_mg_g"] * (toc_for_q / 2.0)))
+            # Methods: fixed capacity (or its specified sensitivity multiplier).
+            ql_mgg = params["q_L_mg_g"]
             ql_m3t = ql_mgg * 1.3975
             g_ads = ql_m3t * pressure_mpa / (pressure_mpa + params["P_L_MPa"])
             phi = por
@@ -274,11 +270,10 @@

                 phi = b["porosity_median"]
                 r["porosity_imputed_flag"] = "1"
             if phi is not None:
-                phi_frac = max(0.0001, phi / 100.0)
+                phi_frac = phi / 100.0
                 temp_k = 293.15 + params["geothermal_gradient_C_per_m"] * depth
                 bg = 0.00035096 * params["Z"] * temp_k / pressure_mpa
                 g_free = (phi_frac * params["S_g"]) / (params["rho_rock_t_m3"] * bg)
-                g_free = g_free * params["free_gas_conservative_scale"]
                 r["gas_ads_proxy"] = g_ads
                 r["gas_free_proxy"] = g_free
                 r["gas_total_proxy_quant"] = g_ads + g_free
@@ -462,7 +457,7 @@

             "unit": "mg/g",
             "source": "W148 moisture-equilibrated 30 C Langmuir anchor; literature/DOI pending author confirmation",
             "gold_label_used": "N",
-            "note": "TOC only scales adsorption capacity during proxy construction; gold total gas is not used for parameter fitting.",
+            "note": "Fixed capacity; proxy uses no TOC or total-gas labels.",
         },
         {
             "parameter": "P_L",
@@ -664,7 +659,7 @@

     return pd.DataFrame(rows)
 
 
-def main():
+def legacy_computational_main():
     ensure_dirs()
     backup = backup_existing()
     write_params()
@@ -691,5 +686,10 @@

     print("North gas C4 sensitivity range", float(focus["benefit_pct"].min()), float(focus["benefit_pct"].max()))
 
 
-if __name__ == "__main__":
+def main():
+    """Run the Methods-aligned audit with explicit data and output paths."""
+    from run_methods_fix import main as run
+    run()
+
+if __name__ == '__main__':
     main()
```

