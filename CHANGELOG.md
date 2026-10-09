# Changelog

## 0.1.1 — 2026-10-09

Documentation, archive naming and metadata cleanup. Computational code and
numerical data are unchanged from 0.1.0. The existing 0.1.0 tag is preserved.

- Renamed the change log and previous-release archive/manifest paths.
- Rewrote the landing README and moved the original verification/resource notes
  verbatim to `docs/reproducibility.md`.
- Corrected the synthetic-seed metadata to 20261009 without changing the generator.
- Updated software/citation version metadata to 0.1.1.

## 0.1.0 — original computational release

The current manuscript has been updated to the corrected A reference results.
Original Online Resource files and the previous public release (v5) were not modified. Superseded public
CSV versions are retained in `paper/previous_release_v5/`.

1. Gas proxy: fixed qL=.39 mg/g and PL=7.60 MPa, no TOC capacity scaling, no
   free-gas .01 factor, porosity percentage divided by 100 without a floor.
   Label-independent full-zone porosity median fills missing rows and is flagged.
   North gas remains directional (C4, +22.8350%, CI crosses zero), not CI-robust.
2. Risk: round each component to nine decimals, then average percentile ranks.
   Original .55/.25/.10/.10 weights, seeds and model parameters are retained.
   A/B 8028 scores match exactly. Matched eligible-unit A gas rejection benefit
   is +6.629140%; TOC is -1.341475%. S19 has a distinct historical aggregation.
3. S19 adds `informative_ranking`: quality_only False, all other schemes True.
   Constant quality carries no ranking information; original numerical columns
   and aggregation design are retained.
4. S21's four variants collapse to identical row selection under average ranks.
   TOC 4/4 false-positive flags are one false positive repeated four times, not
   four independent events. MDE is 16.887467 percentage points; no robust gas gain.
5. S27 retains original numeric values; S28 quality average-rank convention causes
   a common target-level score offset, with unchanged flags/order. S29 gas/North/R
   positive Moran p is .655 (previously .654); interpolation decisions do not change.
6. Reusable attribution (Methods 3.4) retains training-fold TreeExplainer and
   signed one-hot aggregation. All eight added A scalar comparisons pass ±.001.
   Together with the existing 22, these form 28 gate groups: logs-only TOC/gas
   correlations share group 27, and the two TOC ablation values share group 28.
7. English documentation, API adapters, synthetic tutorial, tests, CI and licenses
   added. Action wording now matches the four v2 manuscript actions; conditions
   and thresholds are unchanged. XGBoost parameters/threads/seeds are unchanged.
8. Legacy comments and explanatory report/plot strings are translated only;
   Chinese data values/path literals are retained. The normalized-AST and diff
   audit is in `paper/legacy_scripts/COMMENT_TRANSLATION.md`.

## Figure display-number mapping

| Historical script/output name | Current display | Contents |
|---|---|---|
| Figure_05_log_toc_crossplots | Fig. 3 | Log–TOC crossplots |
| Figure_06_feature_drift | Fig. 4 | Zonal log drift |
| Figure_08_validation_degradation plus TOC residual panels | Fig. 5 | Validation decline / TOC LOZO residuals |
| Figure_09_conformal_coverage_width + Figure_10_rejection_curves | Fig. 6 | Conformal intervals / rejection |
| Figure_11_zone_reliability | Fig. 7 | Zone coverage / rejection |
| Figure_12_proxy_gold_correction + Figure_13_learning_curve_guidance | Fig. 8 | Within-zone correction / learning curve |
| Figure_14_shap_feature_importance / shap_matrix/Figure_SHAP_matrix | Fig. 9 | SHAP matrix |
| build_residual_geology.py / S23 | Fig. 10 | TOC LOZO residual / geological covariates |
| Historical Figure 13 decision-map prose | Fig. 11 | Author artwork; CSV inputs only |
| Figure_04_representative_logs | Fig. S2 | Representative log tracks |
| Figure_07_well_location_scatter | Removed from current sequence | Not the current author decision map |

Historical Table S11 filename `Table_S11_figure4_representative_wells.csv` is
retained as a table source identifier (not a figure output); its README describes
Fig. S2. New figure scripts/output names use current display numbers exclusively.
Original research artwork is not recreated by these CSV companion plots.
