"""Fixed author-approved risk component precision and average ties."""
import numpy as np
import pandas as pd
RANK_DECIMALS=9
def rounded_component(values):
    """Author-approved fixed nine-decimal rounding, before risk ranking.

    Preserves a Series index and missing values. Precision is fixed before
    running either environment and must never be selected from results.
    """
    return values.round(RANK_DECIMALS) if isinstance(values, pd.Series) else np.round(np.asarray(values, float), RANK_DECIMALS)

def percentile_rank(values):
    """Nine-decimal components ranked with average ties, divided by n.

    For global intervals, within-fold width is constant and contributes no
    ordering information; the other components determine the risk order.
    pandas average percentile rank is equivalent to rankdata(average)/n for
    finite inputs. Missing values retain the historical pandas behavior.
    """
    values = values if isinstance(values, pd.Series) else pd.Series(values)
    return rounded_component(values).rank(pct=True, method='average')

def composite_risk(frame,weights=(.55,.25,.10,.10)):
    """Fold-local risk after fixed rounding and average ties (§3.5 erratum).

    Weights remain .55/.25/.10/.10, quality remains constant 0.1. A global
    interval's width is constant within its fold, so it supplies no ordering
    information; the remaining components determine the ranking. Interval
    bounds and width arithmetic retain the original implementation.
    """
    return sum(percentile_rank(frame[c])*w for c,w in zip(['interval_width','domain_knn','missing_risk','quality_risk'],weights))

