"""Retained Table S22 numerical functions; adapters live in attribution.py."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
SEED = 42
METHOD = 'shap'
LOG_COLS = ['GR', 'AC', 'DEN', 'CNL', 'RT']
NUM_COLS = LOG_COLS + ['pressure_coeff', 'formation_top_md', 'formation_bottom_md', 'max_burial_top']
CAT_COLS = ['layer', 'lithology', 'diagenesis_category']
from .domains import ordered_zones
from .constants import ZONE_EN as _ZONE_EN
ZONE_EN = {k:v for k,v in _ZONE_EN.items() if k != v}
def preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")

def feature_config(df: pd.DataFrame, feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set == "logs-only":
        return [col for col in LOG_COLS if col in df.columns], []
    if feature_set == "full":
        return [col for col in NUM_COLS if col in df.columns], [
            col for col in CAT_COLS if col in df.columns
        ]
    raise ValueError(feature_set)

def split_iter(df: pd.DataFrame, scheme: str):
    if scheme == "LOWO":
        for well in sorted(df["well"].dropna().unique()):
            te = np.where(df["well"].to_numpy() == well)[0]
            tr = np.where(df["well"].to_numpy() != well)[0]
            if len(te) and len(tr) >= 10:
                yield str(well), tr, te
    elif scheme == "LOZO":
        for zone in ordered_zones(df, validate_lozo=True):
            te = np.where(df["zone"].to_numpy() == zone)[0]
            tr = np.where(df["zone"].to_numpy() != zone)[0]
            if len(te) and len(tr) >= 10:
                yield str(zone), tr, te
    else:
        raise ValueError(scheme)

def anonymizers(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    wells = sorted(str(w) for w in df["well"].dropna().unique())
    well_map = {well: f"well_anon_{i:03d}" for i, well in enumerate(wells, 1)}
    zone_map = {zone: f"zone_en_{ZONE_EN.get(zone, zone)}" for zone in ordered_zones(df)}
    return well_map, zone_map

def aggregate_shap(
    pre: ColumnTransformer,
    shap_trans: np.ndarray,
    x_orig: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=x_orig.index)
    idx = 0
    for col in num_cols:
        out[col] = shap_trans[:, idx]
        idx += 1
    if cat_cols:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        for col, categories in zip(cat_cols, ohe.categories_):
            n = len(categories)
            out[col] = shap_trans[:, idx : idx + n].sum(axis=1)
            idx += n
    return out

def importance_for_fold(
    model,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.Series:
    pipe = Pipeline(
        [("pre", preprocessor(num_cols, cat_cols)), ("model", clone(model))]
    )
    pipe.fit(x_train, y_train)
    if METHOD == "shap":
        x_trans = pipe.named_steps["pre"].transform(x_train)
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        values = explainer.shap_values(x_trans)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values, dtype=float)
        shap_orig = aggregate_shap(
            pipe.named_steps["pre"], values, x_train, num_cols, cat_cols
        )
        return shap_orig.abs().mean().astype(float)

    result = permutation_importance(
        pipe,
        x_train,
        y_train,
        n_repeats=10,
        random_state=SEED,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=list(x_train.columns), dtype=float)

def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
        return np.nan
    rho = spearmanr(a[ok], b[ok], nan_policy="omit").correlation
    return float(rho) if np.isfinite(rho) else np.nan
