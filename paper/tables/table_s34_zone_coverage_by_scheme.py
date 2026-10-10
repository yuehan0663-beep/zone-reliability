"""Summarize global intervals in the released predictions for Table S34.

Counts are distinct samples and wells across CV schemes. Pooled coverage,
interval width and MAE are row means across all released schemes; LOWO and
LOZO coverage are separate row means. Width and MAE retain each target's units.
No models are fitted and the input predictions are never modified.
"""
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NAME = 'Table_S34_zone_coverage_by_scheme.csv'


def build_table(predictions):
    """Return one row per target and zone using method == 'global'."""
    data = predictions.loc[predictions['method'].eq('global')].copy()
    required = ['target', 'zone_en', 'sample_anon', 'well_anon', 'cv_scheme',
                'covered', 'interval_width', 'abs_error']
    if data.empty or data[required].isna().any().any():
        raise ValueError('Global predictions must contain complete S34 inputs')
    keys = ['target', 'zone_en']
    result = data.groupby(keys, sort=False).agg(
        samples=('sample_anon', 'nunique'),
        wells=('well_anon', 'nunique'),
        coverage_pooled=('covered', 'mean'),
        mean_interval_width=('interval_width', 'mean'),
        MAE=('abs_error', 'mean'),
    )
    coverage = data.groupby(keys + ['cv_scheme'])['covered'].mean().unstack('cv_scheme')
    result = result.join(coverage[['LOWO', 'LOZO']].rename(
        columns={'LOWO': 'coverage_LOWO', 'LOZO': 'coverage_LOZO'}))
    order = pd.MultiIndex.from_product(
        [data['target'].drop_duplicates(), ['North', 'Central', 'South']], names=keys)
    result = result.reindex(order).reset_index()
    if result.isna().any().any():
        raise ValueError('Both LOWO and LOZO predictions are required in each zone')
    return result[['target', 'zone_en', 'samples', 'wells', 'coverage_pooled',
                   'coverage_LOWO', 'coverage_LOZO', 'mean_interval_width', 'MAE']]


def main():
    source = ROOT / 'data_derived/conformal_predictions.csv'
    predictions = pd.read_csv(source, dtype={'covered': 'boolean'})
    result = build_table(predictions)
    output = ROOT / 'supplementary_tables' / NAME
    result.to_csv(output, index=False, encoding='utf-8')
    shutil.copyfile(output, ROOT / 'data_derived' / NAME)
    print(result.to_string(index=False, float_format=lambda value: f'{value:.3f}'))
    print('Table S34 written to supplementary_tables/ and data_derived/')


if __name__ == '__main__':
    main()
