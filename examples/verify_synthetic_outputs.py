"""Verify tutorial outputs independently of confidential paper anchors."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def verify(out):
    out=Path(out)
    metadata=json.loads((out/'runtime.json').read_text())
    assert metadata['synthetic_only'] and metadata['seconds']>0
    data=pd.read_csv(out/'synthetic_dataset.csv')
    assert data.well.str.startswith('SYN_').all()
    predictions=pd.read_csv(out/'oof_predictions.csv')
    assert np.isfinite(predictions.y_pred).all()
    assert set(predictions.cv_scheme)=={'random','LOWO','LOZO'}
    assert predictions.model.nunique()==metadata['models']
    for (_,target,model,scheme),g in predictions.groupby(['dataset','target','model','cv_scheme']):
        assert len(g)==data[target].notna().sum()
        assert g.sample_id.nunique()==len(g)
    importance=pd.read_csv(out/'attribution_perfold.csv')
    assert set(importance.feature_set)=={'full','logs-only'}
    assert np.isfinite(importance.importance).all()
    stability=pd.read_csv(out/'attribution_stability.csv')
    assert len(stability)==8 and stability.method.eq('shap').all()
    assert stability.pairwise_spearman_mean.between(-1,1).all()
    ablation=pd.read_csv(out/'logs_only_ablation.csv')
    assert len(ablation)==12 and np.isfinite(ablation.pooled_R2).all()
    decisions=pd.read_csv(out/'decision_matrix.csv')
    assert len(decisions)==6
    assert set(decisions.action)<= {'use with interval','zone-level only','recalibrate locally','acquire more data'}
    assert (out/'synthetic_diagnostics.png').stat().st_size>10000
    print(f"Synthetic integration PASS: {metadata['models']} models, three phases, eight attribution groups")

if __name__=='__main__':
    verify(sys.argv[1])
