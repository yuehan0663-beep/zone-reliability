# zone-reliability

[![Synthetic workflow tests](https://github.com/yuehan0663-beep/zone-reliability/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/yuehan0663-beep/zone-reliability/actions/workflows/tests.yml)

A Python workflow for assessing the reliability of well-log predictions across
structural zones, accompanying:

**A workflow for bounding the reliability of well-log machine-learning predictions across fault-bounded structural zones: Application to shale TOC and gas content in the Wulalike Formation, Ordos Basin, China**

## Overview

1. **Diagnosis:** leakage-controlled preprocessing and random, leave-one-well-out
   (LOWO), and leave-one-zone-out (LOZO) validation; fold-wise SHAP, logs-only
   ablation, and feature-ranking stability.
2. **Remediation:** prediction intervals, risk-based rejection, and within-zone
   recalibration. The output value `risk_inflated` corresponds to the paper's
   distance-inflated interval variant.
3. **Decision:** combine coverage, calibration support, predictive skill, and
   local correction evidence to choose use with interval, zone-level only,
   recalibrate locally, or acquire more data.

## Installation

Use Python **3.11.9** with the pinned dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Alternatively, `conda env create -f environment.yml`, activate
`zone-reliability`, and run `python -m pip install -e . --no-deps`.

## Quick start (synthetic data)

```bash
python examples/01_quickstart_synthetic.py --fast --output outputs/fast
python examples/01_quickstart_synthetic.py --output outputs/full
```

The example is generated entirely from distributions with seed **20261009**.
TOC-labelled well counts are **8/9/15** and gas-labelled counts **5/11/10** across
North/Central/South, with partial overlap. Fast mode uses five model families;
full mode uses all 16. Both demonstrate the three phases and attribution.
The same tutorial is available in `examples/01_quickstart_synthetic.ipynb`.

## Using your own data

```python
from zonerel.io import read_dataset
from zonerel.pipeline import run_workflow
data = read_dataset('my_logs.csv', group_col='zone', well_col='well')
tables = run_workflow(data, {'fast': True, 'run_attribution': True})
tables['decision_matrix'].to_csv('decisions.csv', index=False)
```

See [the user guide](docs/user_guide.md) for input columns, units, grouping,
options, outputs, and expected behavior.

## Reproducing the paper tables

```bash
python paper/reproduce_paper_tables.py
python paper/reproduce_public_maps.py --out outputs/maps
```

These commands use the distributed derived tables. Rerunning the workflow from
confidential raw data requires authorization; see
[`paper/reproduce_from_private_data.py`](paper/reproduce_from_private_data.py).

## Data availability

Raw well logs and core data belong to **Changqing Oilfield** and are restricted
by a confidentiality agreement. Access may be requested from the corresponding
author of the associated manuscript.

This repository provides anonymized derived data in
[`paper/data_derived/`](paper/data_derived/) and the CSV files for supplementary
Tables S1–S30 and S34 in [`paper/supplementary_tables/`](paper/supplementary_tables/).
Tables S31–S33 are available only in the supplementary-material PDF.
The synthetic example contains no real log or core rows.

## Reproducibility

The reference environment is Python **3.11.9** with exact dependency versions
in `requirements.txt` and `environment.yml`. XGBoost results may vary slightly
with platform and native-library builds; details and reference checks are in
[the reproducibility notes](docs/reproducibility.md).

## Repository layout

```text
src/zonerel/   Workflow modules
examples/      Synthetic dataset, tutorial, and example results
docs/          User guide and reproducibility details
paper/         Derived data, supplementary tables, and reproduction tools
tests/         Synthetic workflow tests
```

## Testing

```bash
python -m pytest -q
```

[GitHub Actions](https://github.com/yuehan0663-beep/zone-reliability/actions/workflows/tests.yml) runs the tests and the
synthetic quick start on Ubuntu.

## Citation

See [`CITATION.cff`](CITATION.cff) for citation metadata and
[the releases](https://github.com/yuehan0663-beep/zone-reliability/releases) for fixed software versions.

## License

Code: [MIT](LICENSE). Public derived and synthetic data:
[CC BY 4.0](LICENSE-data).
