# Manuscript display figures

These portable companions draw the numerical evidence behind the paper figures.
They do not reproduce author artwork or guarantee identical editorial layout.
Default execution uses released derived tables only:

```bash
python paper/figures/generate_paper_figures.py --output outputs/paper_figures
```

Each figure also has a numbered entry point (`fig09_shap_matrix.py`, for example).
Fig. 3 crossplots and Fig. S2 tracks require an authorized local log file, supplied
with `--input` and an explicit `--sheet` for a workbook. Fig. S2 requires the
approved representative-well identity mapping used by Table S11; no replacement
well is chosen if identities do not match. Raw rows are never exported by these scripts.

| Display | Contents | Inputs / scope |
|---|---|---|
| Fig. 1 | Study area | Author artwork |
| Fig. 2 | Three-phase workflow | Author artwork |
| Fig. 3 | Log–TOC crossplots | Local raw logs, model-ready TOC filtering |
| Fig. 4 | Zonal log drift | S7 metrics; S10 statistics also released |
| Fig. 5 | Validation decline / TOC LOZO residuals | S1 and S12 |
| Fig. 6 | Conformal intervals / rejection | Corrected S2 and S3 |
| Fig. 7 | Zonal coverage / rejection | Corrected zone_reliability_summary |
| Fig. 8 | Recalibration / learning-curve guidance | Corrected S4 and S8 |
| Fig. 9 | SHAP matrix | S22a, normalized fold-mean training SHAP |
| Fig. 10 | TOC LOZO residual / geology correlations | S23; no causal inference |
| Fig. 11 | Four-action decision map | Author artwork; S25/S27/S28/S29/S30 CSV inputs |
| Fig. S1 | Stratigraphic column | Author artwork |
| Fig. S2 | Representative well tracks | Table S11 selections and authorized local logs |

Fig. 8 learning guidance resamples existing recalibration errors; it is not a
new model-refitting learning curve. Cross-target MAE/width values retain distinct
target units. Figure companions use distinct markers as well as color.
