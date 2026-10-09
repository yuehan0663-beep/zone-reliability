# zone-reliability

Code, a fully simulated example, and anonymous derived data supporting:

**A workflow for bounding the reliability of well-log machine-learning predictions across fault-bounded structural zones: Application to shale TOC and gas content in the Ordos Basin, China**

Code is MIT licensed; public derived
CSV tables and synthetic data are CC BY 4.0 licensed (see LICENSE-data).

## Install

Reference Python: **3.11.9**, with all package versions pinned in
`requirements.txt` and `environment.yml`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Alternatively, `conda env create -f environment.yml` then activate
`zone-reliability` and run `python -m pip install -e . --no-deps`.

## Typical use

```bash
python examples/01_quickstart_synthetic.py --fast --output outputs/fast
python examples/01_quickstart_synthetic.py --output outputs/full
python paper/reproduce_paper_tables.py
python paper/reproduce_public_maps.py --out outputs/maps
```

The fast tutorial evaluates five model families; full mode evaluates all 16.
Both include the fixed 180-tree attribution backbones and all three workflow
phases. `examples/01_quickstart_synthetic.ipynb` contains the same tutorial code.
All tutorial rows are generated from distributions with seed 20261009: TOC
well counts are 8/9/15 and gas counts 5/11/10 across North/Central/South, with
3/5/7 overlapping wells. Gas-labelled wells are not a TOC subset.

```python
from zonerel.io import read_dataset
from zonerel.pipeline import run_workflow
data = read_dataset('my_logs.csv', group_col='zone', well_col='well')
tables = run_workflow(data, {'fast': True, 'run_attribution': True})
tables['decision_matrix'].to_csv('decisions.csv', index=False)
```

## Three phases

1. **Phase I: diagnosis.** Leakage-controlled features and fold-contained
   preprocessing; random five-fold, leave-one-well-out (LOWO) and
   leave-one-zone-out (LOZO) evaluation; a fixed 16-model pool; fold-wise SHAP,
   logs-only ablation and cross-fold ranking stability (Methods 3.3–3.4).
2. **Phase II: remediation.** Split-conformal global, zone-conditional and
   distance-inflated intervals; composite-risk rejection; within-zone drift
   typing and C0–C4 recalibration (Methods 3.5–3.6). `risk_inflated` in output
   columns means the paper's **distance-inflated** variant.
3. **Phase III: decision.** Assign each zone × target block to **use with
   interval**, **zone-level only**, **recalibrate locally**, or **acquire more
   data**. Calibration support is explicit caller evidence, never a guessed
   minimum-well threshold. See `docs/user_guide.md` for rules and metrics (3.7).

## Reproducibility notes

A (Python 3.11.9) is the paper's reference environment: 28/28 gate groups
pass, including all eight added attribution scalar comparisons. Original model
parameters, thread counts and seeds are retained. Exact installed versions:

```text
catboost==1.2.10
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
et-xmlfile==2.0.0
fonttools==4.62.1
graphviz==0.21
iniconfig==2.3.1
joblib==1.5.3
kiwisolver==1.5.0
lightgbm==4.6.0
llvmlite==0.47.0
matplotlib==3.10.9
narwhals==2.22.1
numba==0.65.1
numpy==2.4.4
openpyxl==3.1.5
packaging==26.2
pandas==3.0.2
pillow==12.2.0
plotly==6.8.0
pluggy==1.6.0
pygments==2.20.0
pyparsing==3.3.2
pytest==8.4.2
python-dateutil==2.9.0.post0
scikit-learn==1.8.0
scipy==1.17.1
shap==0.51.0
six==1.17.0
slicer==0.0.8
threadpoolctl==3.6.0
tqdm==4.68.2
typing-extensions==4.15.0
tzdata==2026.2
xgboost==3.2.0
```

XGBoost can vary with platform and native-library build. In diagnostic B
(Python 3.12, single-thread library limit), maximum prediction difference was
0.117, related coverage differences were at most 0.009, and matched eligible
gas rejection benefit was 6.63% in A versus 6.51% in B. Conclusions agree.
B is a reporting environment and does not block release. Components are
rounded to nine decimals before average percentile ranking; all 8028 tested
composite-risk scores agree exactly across A/B. A constant global width or
quality component contributes no within-fold ordering information.

S19 retains its original aggregation convention and adds `informative_ranking`:
`quality_only=False`, all other schemes True. S14's fair eligible-unit benefit
and S19's differently aggregated sensitivity statistic must not be interchanged.
In S21 the four adaptive variants select the same rows under average ranks:
the TOC 4/4 flags repeat **one false positive four times**, rather than providing
four independent observations. No robust adaptive gas gain is claimed.

## Data and paper reproduction

Raw logs and core data belong to **Changqing Oilfield** and are restricted by
a confidentiality agreement. Request access from the corresponding author of
the associated manuscript. The confidential workbook and private caches are
excluded from this repository and Git history. Approved anonymous derived CSVs,
including author-approved well abbreviations and coordinates, are provided in
`paper/data_derived/`; corrected Tables S1–S30 (36 CSVs) are in
`paper/supplementary_tables/`. Replaced published versions are archived in
`paper/superseded_zenodo_v5/`. Tables S31–S33 are PDF-only and outside this package.

`paper/reproduce_paper_tables.py` checks public table anchors without raw data.
`paper/reproduce_from_private_data.py` reruns the reusable workflow when an
authorized local dataset is supplied. No confidential-data anchors are in CI.
Historical research programs remain in `paper/legacy_scripts/`; portable current
entry points live in `src/zonerel/`. See their translation audit and provenance.
The figure map in `paper/figures/README.md` uses manuscript display numbers;
Figs. 1, 2, 11 and S1 are author artwork. Map inputs are CSVs, not a new decision map.

## Computing resources

Measured on an AMD Ryzen 7 8745HS (8 cores/16 logical processors), approximately
31.3 GiB installed RAM, CPU execution; no GPU required. Measured synthetic wall
times were **191.114 s for --fast** and **176.611 s for full mode**. The added
real-data attribution check took **67.786 s**. The previously completed paper
workflow took approximately **110 s**, estimated from historical file timestamps.
These are individual observations under different process loads; see
`docs/runtime_results.md` for scope and interpretation. Final distribution size,
excluding `.git` and ignored regenerable caches, is approximately **28.07 MiB**.
Model families can emit warnings
on small synthetic folds; see the guide for missing libraries and schema errors.

## Tests

`python -m pytest -q` tests fold separation, risk ties, proxy independence,
attribution aggregation and within-zone constraints using synthetic inputs.
GitHub Actions (`.github/workflows/tests.yml`) installs the pinned environment on
Ubuntu and runs tests and a synthetic tutorial. Remote Actions has not run before
publication. XGBoost assertions use numerical tolerance and no private-paper anchor.
