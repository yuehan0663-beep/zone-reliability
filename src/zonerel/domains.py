"""Data-derived domain order, retaining the paper's canonical ordering."""
import warnings
from .constants import ZONE_ORDER, ZONE_EN


def ordered_zones(df, group_col='zone', validate_lozo=False):
    groups = sorted(df[group_col].dropna().unique())
    if validate_lozo:
        if len(groups) < 2:
            raise ValueError('LOZO requires at least 2 distinct zones in each target dataset')
        if len(groups) == 2:
            warnings.warn('LOZO has only 2 folds; interpret results cautiously.', UserWarning, stacklevel=2)
    if set(groups) <= set(ZONE_ORDER):
        return list(ZONE_ORDER)
    chinese = [k for k, v in ZONE_EN.items() if k != v]
    if set(groups) <= set(chinese):
        return chinese
    return groups
