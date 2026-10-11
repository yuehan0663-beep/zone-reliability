# User guide

## Input schema

`read_dataset(path, sheet_name=None, group_col='zone', well_col='well')`
reads CSV or an explicitly selected XLSX/XLS sheet. `validate_dataset` returns
a copy; the caller's file is unchanged. Custom names are renamed only in memory.
For example, `read_dataset('logs.csv', group_col='fault_zone', well_col='well_id')`.
Arbitrary zone names and counts are supported, including four fault blocks
`F1`–`F4`. `North`, `Central`, `South` and the corresponding Chinese labels retain
the paper's canonical ordering for reproduction; other labels are sorted from
the data. LOZO requires at least two distinct zones in each labelled target dataset;
two zones produce a warning because there are only two folds. Well identities must be
stable and nonmissing. Repeated depth rows within a well are valid samples;
`sample_id` must be unique (generated if absent).

| Column | Required | Unit / meaning |
|---|---|---|
| well / zone (or configured names) | yes | Well identity / structural-zone group |
| depth_md | yes | Measured depth, m |
| GR | yes | Gamma ray, API |
| AC | yes | Acoustic slowness, microseconds/m; convert other units before use |
| DEN | yes | Bulk density, g/cm³ |
| CNL | yes | Neutron porosity, percent |
| RT | yes | Resistivity, ohm m |
| pressure_coeff | optional | Dimensionless pressure coefficient; required for gas physical proxy |
| porosity_pct | optional | Porosity, percent; required for gas physical proxy; NaN rows use label-independent full-zone median |
| toc_pct | at least one target | TOC, wt.% |
| total_gas_m3_t | at least one target | Total gas, m³/t |
| formation_top_md / formation_bottom_md / max_burial_top | optional | Boundary/burial depths, m |
| layer / lithology / diagenesis_category | optional | Pre-modelling categorical geological descriptors |
| sample_id | optional | Unique row identifier |
| sample_anon / well_anon | optional | Author-approved bootstrap ordering aliases for exact paper reproduction |

Target filtering is independent: nonmissing target and at least three of the
five logs. An absent target column becomes NaN. Optional absent descriptors are
omitted, except absent `layer` gets an unspecified category. Do not supply
target-derived interpretation fields as features. Imputation/scaling/one-hot
encoding fit on each training fold. Infinity and nonnumeric log values must be
corrected in the source data; NaN log values are supported.

The target names remain fixed: `toc_pct` and `total_gas_m3_t`. This release
generalizes regressors and grouping labels, not arbitrary target columns or
physical proxies. If `pressure_coeff` is absent it is omitted from model features;
if present it remains a feature. An absent pressure or porosity column skips
the gas physical proxy and gas C0–C4 with one explicit warning, leaving gas
`local_correction_robust=False`. TOC physical proxies can still run using their
own available inputs; their evidence gates remain unchanged. Missing columns
and NaN values in a supplied porosity column have different meanings.

## Phase I: diagnosis

Methods 3.3: `target_datasets(df)` returns `(dataset, target, frame)` tuples;
`eval_baselines(datasets, model_names=None, progress=None, extra_models=None)` returns fold metrics,
out-of-fold predictions and model-error status. `split_iter(df, scheme)` supports
random five-fold (shuffle, seed 42), LOWO and LOZO, with at least ten training rows.
The default 16-model pool is retained in `models.regressor_pool`; fast models are train_mean,
zone_mean, Ridge, RF and ExtraTrees. Missing optional model libraries are errors
in `run_workflow`, never surrogate replacements.

`regressor_pool(model_names=None, extra_models=None)` accepts an optional mapping
`{name: estimator}` of unfitted, scikit-learn-compatible regressors. Names must
not collide with any of the 16 built-ins, even if an optional library is absent.
Callable `fit`/`predict` and `sklearn.base.clone` are required; each fold receives
a fresh clone and training-only imputation, scaling and one-hot encoding.
`model_names` selects the built-in subset (and may name supplied custom models);
all supplied `extra_models` are appended once and participate in random, LOWO,
LOZO and the unchanged LOWO/LOZO ranking used for top-three interval/rejection
selection. With custom models supplied, Phase III evaluates the first-ranked
model for each target against the paired LOWO zone-mean reference, labelled
`A_user_selected`. A missing `zone_mean` is added to that validation pool for
this diagnostic. With no custom models, the original paper skill backbones
and bootstrap order are retained.

Methods 3.4: `foldwise_shap_importance(df, feature_sets=('full','logs-only'),
schemes=('LOWO','LOZO'), group_col='zone', well_col='well')` fits ExtraTrees for TOC
and RF for gas (180 trees, leaf size 3, seed 42). TreeExplainer runs on training
rows, and signed one-hot contributions are summed into original categorical
features before mean absolute SHAP. `ranking_stability(perfold, top_k=5)` reports
pairwise Spearman mean/sample SD, minimum-rank top-k membership frequencies and
full-versus-logs-only mean-rank correlation. Ties may put more than k features
in top-k. `logs_only_ablation(df, ...)` refits the same fixed backbones with
five logs versus full descriptors under all three protocols and reports pooled
R², weighted MAE, fold-mean and pooled Spearman. SHAP describes model reliance;
it does not establish causal geological controls. SHAP is mandatory for this step;
no permutation-importance substitution is made. The optional
`attribution_backbones={'toc_pct': estimator, 'total_gas_m3_t': estimator}` overrides
either target's SHAP and logs-only ablation backbone; omitted targets keep
the original defaults. TreeExplainer-compatible tree ensembles (ExtraTrees,
RandomForest, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM and
CatBoost regressors) are supported subject to the installed SHAP version.
Non-tree models fail explicitly; runtime incompatibilities also report the
backbone and cause, with no substitute explainer. The ordinary validation and
interval/rejection stages accept non-tree custom regressors.

## Phase II: remediation

Methods 3.5: `conformal_rejection(datasets, summary, finite_sample=False)` uses
the top three models selected by the retained LOWO/LOZO ranking. Calibration uses
25% of training wells (seed 42); the ordered row fallback is used where required
by the original implementation. Fit/calibration minima are 10/5. Empirical q90
is the default; `finite_sample=True` is an explicitly labelled sensitivity option.
The `risk_inflated` code value corresponds to paper distance-inflated intervals.
Global interval width is constant within a fold and supplies no ranking signal.
LOZO coverage is an observed diagnostic, not an exchangeability guarantee.

`composite_risk(frame, weights=(.55,.25,.10,.10))` rounds each component to nine
decimals and uses average percentile ranks for width, five-neighbour feature
distance, log missingness and constant quality risk (0.1). `random_control(cpreds)`
retains 5000 true-uniform draws with seed 20260629 and matches baseline/retained
MAE to the same eligible units. Folds retaining fewer than five rows are excluded;
at 20% rejection the rejected count must also be positive. Rejection curves and
fair random controls have distinct aggregation conventions; never substitute
S19 weight-sensitivity values for S14 fair-comparison values.

Methods 3.6: `load_gas_params(path=None)` reads the fixed Methods JSON;
`recalibration_ladder(df, params=None)` returns metrics, benefits, drift,
predictions and rejection tables. C0 proxy, C1 offset, C2 affine, C3 PAVA isotonic
and C4 small ridge use within-zone LOWO only. `evaluate_zone(rows,target,proxy,method)`
rejects mixed zones. C4 uses intercept + unscaled proxy + eight standardized
covariates, lambda=2, with unpenalized intercept. No cross-zone labels are borrowed.
Full-zone density/porosity reference medians are label-independent inputs, not
held-out response statistics. Python Random(20260613) is a continuous bootstrap
stream: 400 metric replicates and 500 selected-benefit replicates, retained order.
Gas pressure P=max(0.001, coefficient×depth×0.0098), fixed qL=0.39 mg/g,
adsorption conversion 1.3975, PL=7.60 MPa, Sg=.5, Z=.9, rock density=2.65 t/m³;
T=293.15+.03×depth, Bg=.00035096×Z×T/P; total=adsorbed+free. There is no TOC
capacity scaling or free-gas 0.01 factor. Porosity is divided by 100 without a floor.

These physical proxies cover TOC and total gas only. `run_recalibration=True`
is the default. Setting it to `False` skips physical proxy construction,
gas-parameter loading, drift typing and the entire C0–C4 recalibration ladder;
the corresponding returned tables are empty with headers. Prediction intervals
and risk rejection still run, providing the coverage evidence needed by Phase III.
Every `local_correction_robust` value is then False, and decisions are limited to
`use with interval`, `zone-level only`, and `acquire more data`.

## Phase III: decision and metrics

`decision_action(coverage_ok,support_ok,skill_class,local_correction_robust)` and
`decision_matrix(coverage,skill,benefits,support,coverage_threshold=.80)` return
four action labels. Coverage and explicit support must pass before model skill
permits use with interval; no_skill then yields zone-level only. Otherwise a
CI-robust within-zone correction permits recalibrate locally; remaining zone–target pairs
require acquire more data. A positive directional model-skill estimate passes
the published skill gate; local correction requires its interval above zero.
The retained map convention pools global LOWO and LOZO sample records. The
workflow `decision_protocols` option labels any deliberate LOWO-only sensitivity.

Methods 3.7: `metrics.summarize(folds,preds)` reports sample-weighted fold MAE,
weighted fold RMSE, pooled out-of-fold R² and fold-mean Spearman. The deprecated
fold-mean R² is labelled and must not substitute for pooled R². `reference.zone_skill`
compares identical within-zone LOWO rows with the zone-mean reference and reports
relative improvement and well bootstrap intervals. Benefit classes distinguish
CI-robust from directional only and no directional benefit.

## Orchestration and options

`run_workflow(df,config=None)` returns named DataFrames. Defaults: full 16 models,
original empirical conformal convention, fixed gas JSON, global LOWO+LOZO map
coverage, threshold .80, random control enabled, attribution disabled. Options:
`fast`, `model_names`, `progress`, `well_col`, `group_col`, `finite_sample`,
`gas_params`, `gas_params_path`, `support_assessment` keyed by `(target,zone)`,
`coverage_threshold`, `decision_protocols`, `random_control`, `run_attribution`,
`extra_models`, `attribution_backbones`, `run_recalibration`.
Fast mode reduces the validation pool only; it never changes tree hyperparameters.
Omitted support assessments are False. Synthetic decisions therefore illustrate
evidence-gated action rules rather than claim adequate field calibration support.

## Outputs

The quickstart writes all returned DataFrames as `<key>.csv`, generated input,
profile, runtime metadata and a diagnostics figure. Column dictionaries are in
`docs/output_columns.md`. `attribution_stability.csv` and `logs_only_ablation.csv`
are produced in both modes. Figure graphics are PNG/PDF/SVG; tutorial figures
describe simulated results. Output paths are supplied by the caller. Notebook
outputs can be regenerated from its final execution cell.

## Troubleshooting

* Missing columns: check configured names and the schema above; fix input units
  before fitting. Identity-rename collisions and duplicate sample IDs raise errors.
* No eligible folds: provide enough independent wells, zones and labelled rows.
  At least ten training rows and five retained rejection rows are required.
* Model evaluation errors: install the pinned research dependencies; inspect the
  reported model/fold exception. Do not silently replace XGBoost/LightGBM/CatBoost.
* SHAP missing: install pinned shap and its transitive dependencies. Attribution
  does not fall back to another method.
* Native-library differences: use Python 3.11.9 reference pins. B differences are
  documented; do not tune parameters to obtain a preferred result.
* Undefined Spearman/R²: constant or very small groups legitimately produce NaN.
* Long runtime: use --fast for a tutorial; bootstrap and random-control iterations
  remain fixed. GPU is unnecessary.
* Private-data reproduction: request authorized source data and pass its explicit
  path/sheet. Public table checks need no source workbook.
