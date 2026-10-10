# Paper companion

Corrected public CSVs reproduce the current manuscript results; A is the
reference environment. The reference acceptance evidence is in `acceptance/`.
The confidential workbook and manuscript sources are not included.

Run `python paper/reproduce_paper_tables.py` for a derived-table audit.
Table S34 summarizes the existing global conformal predictions by target, zone
and CV scheme; run `python paper/tables/table_s34_zone_coverage_by_scheme.py`
to reproduce it without fitting models.
Run `python paper/reproduce_public_maps.py` for public map-input reconstruction.
With an authorized local dataset, use `paper/reproduce_from_private_data.py`.
Historical scripts preserve numerical logic; paths are historical and may require
their documented research layout. Use the reusable package for portable runs.
Corrected snapshots retain the Methods gas formula and risk tie rule; archived
CSV versions are in `previous_release_v5/`. Figures use display numbering.
