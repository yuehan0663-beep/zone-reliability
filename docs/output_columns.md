# Output column dictionary

Names and units follow the retained research outputs. Targets ending `_pct` are
percent, total gas is m³/t, and MAE/RMSE/interval widths use the target unit.
R² and Spearman are dimensionless; coverage is a fraction; benefit fields
ending `_pct` are percentages. `risk_inflated` means distance-inflated.

## attribution_perfold.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| feature_set | full / logs-only (or full_logs_plus_descriptors / logs_only) |
| cv_scheme | random / LOWO / LOZO validation protocol |
| fold_id | Anonymized attribution held-out group |
| feature | Original feature name after categorical SHAP aggregation |
| importance | Training-fold mean absolute SHAP in target units |
| rank | Descending minimum importance rank; ties retained |

## attribution_stability.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| feature_set | full / logs-only (or full_logs_plus_descriptors / logs_only) |
| cv_scheme | random / LOWO / LOZO validation protocol |
| pairwise_spearman_mean | Mean finite pairwise Spearman over fold importance vectors |
| pairwise_spearman_sd | Sample SD of finite pairwise fold correlations |
| n_folds | Eligible outer-fold count |
| top5_membership_json | Fraction of folds with minimum rank <=5 per feature |
| top8_features | Eight smallest mean ranks in order |
| method | Interval variant, attribution method or correction class as appropriate |
| seed | Fixed stochastic seed |
| model | Fixed model family |
| n_features | Number of aggregated original features |
| full_vs_logs_only_common_feature_spearman | Spearman of mean fold ranks for common log features |

## conformal_coverage.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| cv_scheme | random / LOWO / LOZO validation protocol |
| method | Interval variant, attribution method or correction class as appropriate |
| zone | Structural zone |
| coverage | Fraction inside prediction intervals |
| mean_width | Mean upper-minus-lower interval width |
| n | Samples entering the record/group |

## conformal_folds.csv

| Column | Meaning |
|---|---|
| dataset | Target-specific filtered dataset identifier |
| target | Target column or display identifier |
| model | Fixed model family |
| cv_scheme | random / LOWO / LOZO validation protocol |
| fold | Held-out group or random-fold identifier |
| method | Interval variant, attribution method or correction class as appropriate |
| n_train | Training row count |
| n_cal | Rows in the fold-contained conformal calibration subset |
| n_test | Held-out row count summed where aggregated |
| coverage | Fraction inside prediction intervals |
| mean_width | Mean upper-minus-lower interval width |
| MAE | Mean absolute error in the record/group |
| RMSE | Root mean square error |
| R2 | R² for the record/group |
| Spearman | Rank correlation for the record/group |
| domain_abs_error_spearman | Spearman correlation between feature-domain distance and absolute held-out error |

## conformal_predictions.csv

| Column | Meaning |
|---|---|
| dataset | Target-specific filtered dataset identifier |
| target | Target column or display identifier |
| model | Fixed model family |
| cv_scheme | random / LOWO / LOZO validation protocol |
| fold | Held-out group or random-fold identifier |
| method | Interval variant, attribution method or correction class as appropriate |
| sample_id | Unique input sample row |
| well | Input well identity |
| zone | Structural zone |
| layer | Pre-modelling stratigraphic member |
| y_true | Observed held-out response |
| y_pred | Predicted held-out response |
| lo | Prediction-interval lower endpoint |
| hi | Prediction-interval upper endpoint |
| covered | Held-out observation inside interval indicator |
| interval_width | Upper-minus-lower prediction interval width |
| domain_knn | Five-neighbour feature-domain distance component |
| missing_risk | Fraction of missing selected logs |
| quality_risk | Constant 0.1 quality component |
| abs_error | Absolute held-out residual |
| composite_risk | Weighted nine-decimal average percentile ranks |

## decision_matrix.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| zone | Structural zone |
| n_wells | Distinct wells entering record/group |
| coverage | Fraction inside prediction intervals |
| coverage_threshold | Decision coverage threshold (default .80) |
| support_ok | Caller-supplied calibration-support assessment |
| skill_class | no_skill / skill_directional / skill_CI_robust |
| local_correction_robust | At least one selected correction has CI-robust evidence |
| action | One of the four evidence-gated action classes |

## drift_typing.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| proxy | Label-independent physical/log proxy name |
| zone | Structural zone |
| zone_en | English structural-zone label |
| n | Samples entering the record/group |
| wells | Distinct wells entering record/group |
| spearman_proxy_gold | Within-zone proxy/response rank correlation |
| affine_slope | Within-zone proxy/response fitted slope |
| affine_intercept | Within-zone fitted intercept |
| drift_type | Diagnosis from proxy/response affine slope and ordering |

## logs_only_ablation.csv

| Column | Meaning |
|---|---|
| dataset | Target-specific filtered dataset identifier |
| target | Target column or display identifier |
| model | Fixed model family |
| feature_set | full / logs-only (or full_logs_plus_descriptors / logs_only) |
| cv_scheme | random / LOWO / LOZO validation protocol |
| folds | Number of eligible validation folds |
| n_test | Held-out row count summed where aggregated |
| MAE_w | Sample-weighted outer-fold MAE |
| RMSE_w | Sample-weighted outer-fold RMSE |
| fold_mean_R2_deprecated | Historical mean fold R²; deprecated for pooled performance |
| Spearman_mean | Unweighted mean of finite fold correlations |
| pooled_R2 | R² over pooled held-out predictions, not average fold R² |
| pooled_n | Number of pooled held-out prediction rows |
| pooled_Spearman | Spearman over all held-out prediction rows |

## oof_predictions.csv

| Column | Meaning |
|---|---|
| sample_id | Unique input sample row |
| well | Input well identity |
| zone | Structural zone |
| layer | Pre-modelling stratigraphic member |
| depth_md | Measured depth in metres |
| cleaning_confidence | Retained model-ready selection class |
| dataset | Target-specific filtered dataset identifier |
| target | Target column or display identifier |
| model | Fixed model family |
| cv_scheme | random / LOWO / LOZO validation protocol |
| fold | Held-out group or random-fold identifier |
| y_true | Observed held-out response |
| y_pred | Predicted held-out response |
| abs_error | Absolute held-out residual |

## random_rejection_control.csv

| Column | Meaning |
|---|---|
| dataset | Target-specific filtered dataset identifier |
| target | Target column or display identifier |
| cv_scheme | random / LOWO / LOZO validation protocol |
| method | Interval variant, attribution method or correction class as appropriate |
| reject_frac | Fraction rejected before retaining >=5 rows |
| baseline_MAE_same_eligible_units | Mean model-fold MAE using the same eligible units as risk rejection |
| risk_MAE_20pct_same_eligible_units | Retained MAE at 20% risk rejection on those same units |
| risk_improvement_pct_same_eligible_units | Fair matched-unit relative risk-rejection benefit |
| random_improvement_mean_pct | Mean true-uniform rejection benefit over 5000 draws |
| random_improvement_p95_pct | 95th percentile true-uniform benefit |
| empirical_p_random_ge_risk | Fraction of uniform benefits at least the risk benefit |

## recalibration_benefits.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| proxy | Label-independent physical/log proxy name |
| zone | Structural zone |
| zone_en | English structural-zone label |
| wells | Distinct wells entering record/group |
| best_method | Lowest-MAE selected C1–C4 method |
| best_MAE | Selected within-zone correction MAE |
| zone_mean_MAE | Same-sample zone-mean reference MAE |
| relative_MAE_improvement_vs_zonemean_pct | 100 × (reference MAE − corrected MAE)/reference MAE |
| improvement_ci_low_pct | Lower 95% well-bootstrap bound on relative MAE improvement, in percent |
| improvement_ci_high_pct | Upper 95% well-bootstrap bound on relative MAE improvement, in percent |
| benefit_class | CI-robust / directional only / no directional benefit |
| small_sample_note | Explicit small-well-count interpretation note |

## recalibration_ladder.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| proxy | Label-independent physical/log proxy name |
| zone | Structural zone |
| zone_en | English structural-zone label |
| method | Interval variant, attribution method or correction class as appropriate |
| n | Samples entering the record/group |
| wells | Distinct wells entering record/group |
| pooled_R2 | R² over pooled held-out predictions, not average fold R² |
| weighted_MAE | Within-zone pooled/sample-weighted MAE |
| Spearman | Rank correlation for the record/group |
| pooled_R2_ci_low | Lower well-bootstrap 95% bound for pooled_R2 |
| pooled_R2_ci_high | Upper well-bootstrap 95% bound for pooled_R2 |
| weighted_MAE_ci_low | Lower well-bootstrap 95% bound for weighted_MAE |
| weighted_MAE_ci_high | Upper well-bootstrap 95% bound for weighted_MAE |
| Spearman_ci_low | Lower well-bootstrap 95% bound for Spearman |
| Spearman_ci_high | Upper well-bootstrap 95% bound for Spearman |

## recalibration_predictions.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| proxy | Label-independent physical/log proxy name |
| zone | Structural zone |
| zone_en | English structural-zone label |
| well | Input well identity |
| sample_id | Unique input sample row |
| method | Interval variant, attribution method or correction class as appropriate |
| observed | Observed within-zone response |
| predicted | Within-zone LOWO corrected response |
| zone_mean_prediction | Within-zone training-well response mean |
| rejection_score | Distance from training proxy distribution |
| abs_error | Absolute held-out residual |
| zone_mean_abs_error | Absolute observed-minus-training-zone-mean residual, in target units |

## rejection_curve.csv

| Column | Meaning |
|---|---|
| dataset | Target-specific filtered dataset identifier |
| target | Target column or display identifier |
| model | Fixed model family |
| cv_scheme | random / LOWO / LOZO validation protocol |
| fold | Held-out group or random-fold identifier |
| method | Interval variant, attribution method or correction class as appropriate |
| risk_score | Risk component/rule used for ordering |
| reject_frac | Fraction rejected before retaining >=5 rows |
| kept_n | Retained sample count |
| coverage | Fraction inside prediction intervals |
| mean_width | Mean upper-minus-lower interval width |
| MAE | Mean absolute error in the record/group |
| RMSE | Root mean square error |
| R2 | R² for the record/group |
| Spearman | Rank correlation for the record/group |

## validation_folds.csv

| Column | Meaning |
|---|---|
| dataset | Target-specific filtered dataset identifier |
| target | Target column or display identifier |
| model | Fixed model family |
| cv_scheme | random / LOWO / LOZO validation protocol |
| fold | Held-out group or random-fold identifier |
| n_train | Training row count |
| n_test | Held-out row count summed where aggregated |
| test_zones | Comma-separated held-out zones |
| test_wells | Comma-separated held-out wells |
| MAE | Mean absolute error in the record/group |
| RMSE | Root mean square error |
| R2 | R² for the record/group |
| Spearman | Rank correlation for the record/group |

## validation_summary.csv

| Column | Meaning |
|---|---|
| dataset | Target-specific filtered dataset identifier |
| target | Target column or display identifier |
| model | Fixed model family |
| cv_scheme | random / LOWO / LOZO validation protocol |
| folds | Number of eligible validation folds |
| n_test | Held-out row count summed where aggregated |
| MAE_w | Sample-weighted outer-fold MAE |
| RMSE_w | Sample-weighted outer-fold RMSE |
| fold_mean_R2_deprecated | Historical mean fold R²; deprecated for pooled performance |
| Spearman_mean | Unweighted mean of finite fold correlations |
| pooled_R2 | R² over pooled held-out predictions, not average fold R² |
| pooled_n | Number of pooled held-out prediction rows |

## zone_skill.csv

| Column | Meaning |
|---|---|
| target | Target column or display identifier |
| zone_en | English structural-zone label |
| predictor_group | Fixed backbone or best-model performance envelope |
| model | Fixed model family |
| n_points | Held-out samples entering zone-skill comparison |
| n_wells | Distinct wells entering record/group |
| MAE_model | Model MAE on the matched zone LOWO rows |
| MAE_zone_mean | Reference MAE on those same rows |
| delta_MAE | Model MAE minus matched zone-mean reference MAE; negative means improvement |
| relative_improvement_pct | Relative MAE reduction as percent |
| ci_low_pct | Well-bootstrap lower relative-improvement bound |
| ci_high_pct | Well-bootstrap upper relative-improvement bound |
| spearman_model | Within-zone model rank correlation |
| skill_class | no_skill / skill_directional / skill_CI_robust |
