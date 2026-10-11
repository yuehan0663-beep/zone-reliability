"""Original random/LOWO/LOZO folds (paper §3.3)."""
import numpy as np
from sklearn.model_selection import KFold
from .domains import ordered_zones

def split_iter(df, scheme, group_col='zone', well_col='well', zone_order=None):
    """Yield (fold label, train positions, test positions).

    random: shuffled fivefold KFold(seed=42). LOWO and LOZO require at
    least ten training rows. Canonical LOZO order is North/Central/South;
    other grouping labels use sorted groups unless zone_order is supplied.
    """
    if scheme=='random':
        for i,(tr,te) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42).split(df),1):yield f'random_{i}',np.asarray(tr),np.asarray(te)
    elif scheme in ['LOWO','LOZO']:
        col=well_col if scheme=='LOWO' else group_col
        groups=sorted(df[col].dropna().unique())
        if scheme=='LOZO':
            inferred=ordered_zones(df, group_col, validate_lozo=True)
            groups=zone_order or inferred
        for group in groups:
            te=np.where(df[col].to_numpy()==group)[0];tr=np.where(df[col].to_numpy()!=group)[0]
            if len(te) and len(tr)>=10:yield str(group),tr,te
    else:raise ValueError('Unknown validation scheme: '+str(scheme))
