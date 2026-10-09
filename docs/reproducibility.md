# Reproducibility details

The sections below retain the reference verification and measured-resource notes
from the README before the 0.1.1 documentation cleanup. They record the original
computational release; the cleanup does not rerun or alter its reference results.
The previously measured distribution size is retained as a historical observation.

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


## Supporting evidence

Reference comparisons are in `../paper/acceptance/`; measured runtime scope is
also documented in `runtime_results.md`. The source workflow and numerical
outputs are unchanged. Current CI status is shown by the README status badge.
