from __future__ import annotations
import math,random
from collections import defaultdict
import numpy as np
import pandas as pd
from .models import GroupMeanRegressor
BOOTSTRAP_REPLICATES=500
BOOTSTRAP_SEED=20260613
ZONES=("North","Central","South")
KEY_FIELDS=("sample_anon","well_anon","zone_en","target","cv_scheme","fold_label_anon")
# Source: run_zone_skill_criterion.py:60
def percentile(values: list[float], q: float) -> float:
    """percentile: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    values = sorted((v for v in values if math.isfinite(v)))
    if not values:
        return float('nan')
    pos = (len(values) - 1) * q / 100
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)

# Source: run_zone_skill_criterion.py:72
def rankdata(values: list[float]) -> list[float]:
    """rankdata: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end + 2) / 2
        for position in range(cursor, end + 1):
            ranks[order[position]] = rank
        cursor = end + 1
    return ranks

# Source: run_zone_skill_criterion.py:87
def spearman(y_true: list[float], y_pred: list[float]) -> float:
    """spearman: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    if len(y_true) < 3:
        return float('nan')
    rank_true = np.asarray(rankdata(y_true), dtype=float)
    rank_pred = np.asarray(rankdata(y_pred), dtype=float)
    if np.std(rank_true) == 0 or np.std(rank_pred) == 0:
        return float('nan')
    return float(np.corrcoef(rank_true, rank_pred)[0, 1])

# Source: run_zone_skill_criterion.py:97
def mae(y_true: list[float], y_pred: list[float]) -> float:
    """mae: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

# Source: run_zone_skill_criterion.py:101
def key(row: dict[str, str]) -> tuple[str, ...]:
    """key: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return tuple((row[field] for field in KEY_FIELDS))

# Source: run_zone_skill_criterion.py:105
def paired_rows(lowo_rows: list[dict[str, str]], target: str, model: str, zone: str | None=None) -> list[dict[str, object]]:
    """paired_rows: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    baseline: dict[tuple[str, ...], dict[str, str]] = {}
    for row in lowo_rows:
        if row['target'] == target and row['model'] == 'zone_mean':
            if zone is None or row['zone_en'] == zone:
                if key(row) in baseline:
                    raise ValueError(f'Duplicate zone_mean row for key {key(row)}')
                baseline[key(row)] = row
    paired: list[dict[str, object]] = []
    for row in lowo_rows:
        if row['target'] != target or row['model'] != model:
            continue
        if zone is not None and row['zone_en'] != zone:
            continue
        base = baseline.get(key(row))
        if base is None:
            raise ValueError(f'Missing zone_mean match for key {key(row)}')
        if not math.isclose(float(row['y_true']), float(base['y_true']), abs_tol=1e-12):
            raise ValueError(f'y_true mismatch for key {key(row)}')
        paired.append({'sample_anon': row['sample_anon'], 'well_anon': row['well_anon'], 'zone_en': row['zone_en'], 'fold_label_anon': row['fold_label_anon'], 'y_true': float(row['y_true']), 'y_pred_model': float(row['y_pred']), 'y_pred_baseline': float(base['y_pred'])})
    if len(paired) != len(baseline):
        raise ValueError(f'Model/baseline row count mismatch for {target}/{model}/{zone}: {len(paired)} vs {len(baseline)}')
    return paired

# Source: run_zone_skill_criterion.py:146
def bootstrap_improvement(rows: list[dict[str, object]], rng: random.Random, replicates: int=BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    """bootstrap_improvement: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    wells = [str(row['well_anon']) for row in rows]
    unique_wells = sorted(set(wells))
    indices_by_well = {well: [index for index, value in enumerate(wells) if value == well] for well in unique_wells}
    values: list[float] = []
    for _ in range(replicates):
        indices: list[int] = []
        for well in [rng.choice(unique_wells) for _ in unique_wells]:
            indices.extend(indices_by_well[well])
        y_true = [float(rows[index]['y_true']) for index in indices]
        model = [float(rows[index]['y_pred_model']) for index in indices]
        baseline = [float(rows[index]['y_pred_baseline']) for index in indices]
        baseline_mae = mae(y_true, baseline)
        model_mae = mae(y_true, model)
        if baseline_mae > 0:
            values.append((baseline_mae - model_mae) / baseline_mae * 100)
    return (percentile(values, 2.5), percentile(values, 97.5))

# Source: run_zone_skill_criterion.py:171
def classify(improvement: float, ci_low: float) -> str:
    """classify: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    if improvement > 0 and ci_low > 0:
        return 'skill_CI_robust'
    if improvement > 0:
        return 'skill_directional'
    return 'no_skill'

# Source: run_zone_skill_criterion.py:179
def calculate_row(rows: list[dict[str, object]], target: str, zone: str, predictor_group: str, model: str, rng: random.Random) -> tuple[dict[str, object], float]:
    """calculate_row: retained research implementation; paper §5.5, S30. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    y_true = [float(row['y_true']) for row in rows]
    model_pred = [float(row['y_pred_model']) for row in rows]
    baseline_pred = [float(row['y_pred_baseline']) for row in rows]
    model_mae = mae(y_true, model_pred)
    baseline_mae = mae(y_true, baseline_pred)
    improvement = (baseline_mae - model_mae) / baseline_mae * 100
    ci_low, ci_high = bootstrap_improvement(rows, rng)
    folds: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        folds[str(row['fold_label_anon'])].append(index)
    weighted_fold_mae = sum((mae([y_true[index] for index in indices], [model_pred[index] for index in indices]) * len(indices) for indices in folds.values())) / len(rows)
    equivalence_error = abs(weighted_fold_mae - model_mae)
    result = {'target': target, 'zone_en': zone, 'predictor_group': predictor_group, 'model': model, 'n_points': len(rows), 'n_wells': len({str(row['well_anon']) for row in rows}), 'MAE_model': f'{model_mae:.6f}', 'MAE_zone_mean': f'{baseline_mae:.6f}', 'delta_MAE': f'{model_mae - baseline_mae:.6f}', 'relative_improvement_pct': f'{improvement:.6f}', 'ci_low_pct': f'{ci_low:.6f}', 'ci_high_pct': f'{ci_high:.6f}', 'spearman_model': f'{spearman(y_true, model_pred):.6f}', 'skill_class': classify(improvement, ci_low)}
    return (result, equivalence_error)

def zone_skill(predictions):
    """Recompute S30 paired zone-mean skill with its original RNG sequence.

    Optional sample_anon/well_anon columns control the published bootstrap
    identities without changing model folds. Returns all fixed-backbone and
    available TOC envelope rows; synthetic fast mode omits the SVR envelope.
    """
    frame=predictions.copy()
    for new,old in [('sample_anon','sample_id'),('well_anon','well'),('zone_en','zone'),('fold_label_anon','fold')]:
        if new not in frame:frame[new]=frame[old]
    lowo=frame[frame.cv_scheme=='LOWO'].to_dict('records')
    available=set(frame.model)
    specs=[('toc_pct','A_fixed_backbone','ExtraTrees'),('toc_pct','B_table3_best_envelope','SVR_RBF'),('total_gas_m3_t','A_fixed_backbone_and_B_table3_best','RF')]
    rng=random.Random(BOOTSTRAP_SEED);out=[]
    zones=ZONES if set(frame.zone)<=set(ZONES) else tuple(sorted(frame.zone.unique()))
    for target,group,model in specs:
        if model not in available or not any(r['target']==target for r in lowo):continue
        for zone in zones:
            paired=paired_rows(lowo,target,model,zone)
            if paired:
                row,error=calculate_row(paired,target,zone,group,model,rng);out.append(row)
    return pd.DataFrame(out)
